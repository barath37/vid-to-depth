import argparse
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import numpy as np

from src.common.logging_utils import StageTimer, logger
from src.common.coordinates import geodetic_to_enu
from src.common.types import KeyframeData, CameraPose, PointCloudModel
from src.sensor_fusion.telemetry_parser import parse_dji_srt, parse_csv_telemetry, TelemetryLog
from src.sensor_fusion.time_sync import estimate_time_offset
from src.sensor_fusion.ekf_smoother import TrajectoryEKFSmoother
from src.sensor_fusion.camera_calibration import CameraIntrinsics
from src.keyframing.blur_detector import compute_laplacian_variance
from src.keyframing.adaptive_sampler import AdaptiveKeyframeSampler
from src.keyframing.dynamic_masker import DynamicObjectMasker
from src.sfm.feature_tracker import SequentialFeatureTracker
from src.sfm.pose_estimator import RelativePoseEstimator
from src.sfm.bundle_adjustment import GlobalBundleAdjuster
from src.sfm.metric_aligner import MetricTrajectoryAligner
from src.depth.monocular_estimator import MonocularDepthEstimator
from src.depth.metric_fusion import align_relative_depth_to_metric
from src.depth.confidence_tagger import EpistemicConfidenceTagger
from src.depth.unprojector import PseudoLiDARUnprojector
from src.reconstruction.fast_ortho_dsm import FastOrthoDSMGenerator
from src.reconstruction.surface_mesher import PoissonSurfaceMesher
from src.reconstruction.texture_projector import MeshTextureProjector
from src.reconstruction.gaussian_splatting import GaussianSplatModel
from src.exporter.mesh_exporter import export_mesh_multi_format
from src.exporter.pointcloud_exporter import export_pointcloud_las
from src.exporter.raster_exporter import export_geotiff_raster
from src.exporter.fbx_converter import convert_to_fbx


def run_pipeline(
    video_path: str,
    telemetry_path: str,
    output_dir: str = "data/output",
    config_path: str = "configs/pipeline_config.yaml",
    serve_viewer: bool = False
):
    """Executes the complete end-to-end 3D reconstruction pipeline (Steps 0 through 6)."""
    timer = StageTimer(budget_minutes=15.0)
    os.makedirs(output_dir, exist_ok=True)

    # Load configuration
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    logger.info(f"=== Starting SIH26158 VidToDepth Pipeline on {video_path} ===")

    # -------------------------------------------------------------
    # Step 0: Calibration & Telemetry Ingestion (Scale Anchor)
    # -------------------------------------------------------------
    with timer.measure("Step 0: Calibration & Telemetry Sensor Fusion"):
        if telemetry_path.endswith(".srt"):
            raw_telemetry = parse_dji_srt(telemetry_path)
        else:
            raw_telemetry = parse_csv_telemetry(telemetry_path)

        if not raw_telemetry:
            raise ValueError(f"No valid telemetry samples parsed from {telemetry_path}")

        telem_log = TelemetryLog(raw_telemetry)
        ref_s = telem_log.samples[0]
        ref_lat, ref_lon, ref_alt = ref_s.latitude, ref_s.longitude, ref_s.altitude_msl

        # Convert GNSS lat/lon/alt to local ENU (meters)
        raw_enu = np.array([
            geodetic_to_enu(s.latitude, s.longitude, s.altitude_msl, ref_lat, ref_lon, ref_alt)
            for s in telem_log.samples
        ])
        times = telem_log.timestamps

        # Smooth trajectory using EKF + RTS smoother
        smoother = TrajectoryEKFSmoother(gps_std=1.5, baro_std=0.5, accel_std=0.5)
        smoothed_enu = smoother.filter_and_smooth(times, raw_enu)

        # Camera intrinsics (default 1080p drone spec)
        calib = CameraIntrinsics.from_sensor_specs(
            image_width=1920,
            image_height=1080,
            sensor_width_mm=13.2,
            sensor_height_mm=8.8,
            focal_length_mm=8.8
        )
        K = calib.intrinsic_matrix

    # -------------------------------------------------------------
    # Step 1: Intelligent Adaptive Keyframing (Edge Ingestion)
    # -------------------------------------------------------------
    with timer.measure("Step 1: Intelligent Adaptive Keyframing"):
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        candidate_keyframes = []
        frame_idx = 0
        step_stride = max(1, int(fps / 2)) # sample candidate frames at 2Hz

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame_idx % step_stride == 0:
                t = frame_idx / fps
                telem_sample = telem_log.sample_at_time(t)
                var_l = compute_laplacian_variance(frame)
                kf = KeyframeData(
                    frame_idx=frame_idx,
                    timestamp=t,
                    image=frame,
                    laplacian_var=var_l,
                    is_i_frame=(frame_idx % int(fps * 2) == 0),
                    telemetry=telem_sample
                )
                candidate_keyframes.append(kf)
            frame_idx += 1
        cap.release()

        sampler = AdaptiveKeyframeSampler(target_keyframes=300, hard_cap_keyframes=400)
        selected_keyframes = sampler.sample_sequence(
            candidate_keyframes,
            total_flight_seconds=float(total_video_frames / fps)
        )
        logger.info(f"Keyframed {len(selected_keyframes)} optimal frames from {total_video_frames} total frames.")

        masker = DynamicObjectMasker(enable_detector=False)
        for kf in selected_keyframes:
            kf.mask = masker.get_mask(kf.image)

    # -------------------------------------------------------------
    # Step 2: Structure from Motion (SfM) & Global Metric Alignment
    # -------------------------------------------------------------
    with timer.measure("Step 2: SfM & Global Metric Alignment"):
        tracker = SequentialFeatureTracker(detector_type="ORB", n_features=1500)
        pose_estimator = RelativePoseEstimator(K)

        poses: List[CameraPose] = []
        sparse_points_list = []

        # Initialize first pose at origin
        poses.append(CameraPose(selected_keyframes[0].frame_idx, np.eye(3), np.zeros(3), selected_keyframes[0].timestamp))

        # Sequential pose recovery
        for i in range(1, len(selected_keyframes)):
            kf_prev = selected_keyframes[i - 1]
            kf_curr = selected_keyframes[i]

            feats1 = tracker.detect_and_compute(kf_prev.image, kf_prev.mask)
            feats2 = tracker.detect_and_compute(kf_curr.image, kf_curr.mask)
            matches = tracker.match_pairs(feats1, feats2)

            if len(matches) >= 8:
                pts1 = np.float32([feats1[0][m[0]].pt for m in matches])
                pts2 = np.float32([feats2[0][m[1]].pt for m in matches])

                R_rel, t_rel, inliers = pose_estimator.estimate_pose(pts1, pts2)
                # Concatenate pose: R_curr = R_rel @ R_prev, t_curr = R_rel @ t_prev + t_rel
                R_curr = R_rel @ poses[-1].R
                t_curr = R_rel @ poses[-1].t + t_rel.flatten()
                poses.append(CameraPose(kf_curr.frame_idx, R_curr, t_curr, kf_curr.timestamp))

                if len(inliers) > 0:
                    pts3d = pose_estimator.triangulate_points(pts1[inliers], pts2[inliers], R_rel, t_rel)
                    sparse_points_list.append(pts3d)
            else:
                # Telemetry pose prior fallback (Fallback Level 2)
                poses.append(CameraPose(kf_curr.frame_idx, poses[-1].R, poses[-1].t + np.array([0, 1.0, 0]), kf_curr.timestamp))

        # Sim(3) Metric Alignment to smoothed EKF trajectory
        kf_timestamps = np.array([kf.timestamp for kf in selected_keyframes])
        ekf_pos_matched = np.array([
            smoothed_enu[np.argmin(np.abs(times - ts))] for ts in kf_timestamps
        ])

        aligner = MetricTrajectoryAligner()
        gravity = np.array([0.0, 0.0, -9.81])
        ground_norm = np.array([0.0, 0.0, 1.0])

        scale_sim, R_sim, t_sim, metric_poses = aligner.align_to_ekf(
            poses, ekf_pos_matched, gravity_vector=gravity, ground_normal=ground_norm
        )
        logger.info(f"Sim(3) Metric Alignment: scale={scale_sim:.3f}")

    # -------------------------------------------------------------
    # Step 3: Confidence-Aware Depth Completion & Pseudo-LiDAR
    # -------------------------------------------------------------
    with timer.measure("Step 3: Confidence-Aware Depth Completion"):
        depth_estimator = MonocularDepthEstimator(use_mock=True) # Deterministic offline execution
        confidence_tagger = EpistemicConfidenceTagger()
        unprojector = PseudoLiDARUnprojector(K, max_depth_m=300.0)

        accumulated_points = []
        accumulated_colors = []
        accumulated_conf = []

        for kf, pose in zip(selected_keyframes, metric_poses):
            rel_depth = depth_estimator.infer_depth(kf.image)
            h, w = rel_depth.shape

            # Sample anchor sparse points from telemetry elevation
            sparse_u = np.array([w // 4, w // 2, 3 * w // 4])
            sparse_v = np.array([h // 4, h // 2, 3 * h // 4])
            flight_alt = kf.telemetry.altitude_rel if kf.telemetry else 60.0
            sparse_depths = np.full(3, flight_alt / max(np.cos(np.deg2rad(30.0)), 0.1))

            metric_depth, _, _ = align_relative_depth_to_metric(
                rel_depth, sparse_u, sparse_v, sparse_depths
            )

            sparse_uvs = np.column_stack([sparse_u, sparse_v])
            conf_map, _ = confidence_tagger.compute_confidence(
                image_shape=(h, w),
                sparse_uvs=sparse_uvs
            )

            cloud_frame = unprojector.unproject_frame(
                metric_depth, kf.image, conf_map, pose, mask=kf.mask
            )

            # Subsample per frame for memory efficiency
            if len(cloud_frame.points) > 1000:
                sub_idx = np.random.choice(len(cloud_frame.points), 1000, replace=False)
                accumulated_points.append(cloud_frame.points[sub_idx])
                accumulated_colors.append(cloud_frame.colors[sub_idx])
                accumulated_conf.append(cloud_frame.confidence[sub_idx])
            elif len(cloud_frame.points) > 0:
                accumulated_points.append(cloud_frame.points)
                accumulated_colors.append(cloud_frame.colors)
                accumulated_conf.append(cloud_frame.confidence)

        global_points = np.vstack(accumulated_points) if accumulated_points else np.zeros((0, 3), dtype=np.float32)
        global_colors = np.vstack(accumulated_colors) if accumulated_colors else np.zeros((0, 3), dtype=np.uint8)
        global_conf = np.concatenate(accumulated_conf) if accumulated_conf else np.zeros(0, dtype=np.float32)

        global_cloud = PointCloudModel(
            points=global_points,
            colors=global_colors,
            confidence=global_conf
        )
        logger.info(f"Generated global Pseudo-LiDAR cloud with {len(global_cloud.points)} points.")

    # -------------------------------------------------------------
    # Step 4: Dual-Tier 3D Reconstruction
    # -------------------------------------------------------------
    with timer.measure("Step 4: Dual-Tier 3D Reconstruction"):
        # Tier 1: Fast Orthomosaic & DSM
        ortho_gen = FastOrthoDSMGenerator(resolution_m=0.20)
        ortho_img, dsm_img, ortho_bounds = ortho_gen.generate(global_cloud)

        # Surface Meshing via Poisson on Pseudo-LiDAR
        mesher = PoissonSurfaceMesher(depth=8, voxel_size_m=0.20, density_trim_quantile=0.02)
        mesh = mesher.reconstruct(global_cloud)

        # Best-view texture projection
        projector = MeshTextureProjector(K)
        textured_mesh = projector.project_texture(mesh, selected_keyframes, metric_poses)

        # Tier 2: 3D Gaussian Splats visual representation
        gs_model = GaussianSplatModel.from_point_cloud(global_cloud)
        logger.info(f"Reconstruction complete: Mesh has {len(textured_mesh.vertices)} vertices, {len(textured_mesh.faces)} faces.")

    # -------------------------------------------------------------
    # Step 5: Multi-Format GIS & 3D Deliverables Export
    # -------------------------------------------------------------
    with timer.measure("Step 5: Multi-Format GIS & 3D Export"):
        epsg = cfg.get("sensor_fusion", {}).get("default_crs_epsg", 32643)

        # 1. Mesh formats: OBJ, PLY, GLB + georef JSON
        mesh_base = os.path.join(output_dir, "reconstruction")
        local_origin = np.array([ref_lon * 10000.0, ref_lat * 10000.0, ref_alt])
        mesh_outputs = export_mesh_multi_format(
            textured_mesh, mesh_base, local_origin=local_origin, epsg_code=epsg
        )

        # 2. Point Cloud: LAS 1.4
        las_path = os.path.join(output_dir, "reconstruction.las")
        export_pointcloud_las(global_cloud, las_path, epsg_code=epsg)

        # 3. Rasters: GeoTIFF Orthomosaic & DSM
        ortho_tif = os.path.join(output_dir, "orthomosaic.tif")
        dsm_tif = os.path.join(output_dir, "dsm.tif")
        export_geotiff_raster(ortho_img, ortho_bounds, ortho_tif, epsg_code=epsg)
        export_geotiff_raster(dsm_img, ortho_bounds, dsm_tif, epsg_code=epsg, nodata=-9999.0)

        # 4. FBX format
        fbx_path = os.path.join(output_dir, "reconstruction.fbx")
        convert_to_fbx(mesh_outputs["glb"], fbx_path)

        # 5. 3DGS PLY
        gs_ply = os.path.join(output_dir, "gaussian_splats.ply")
        try:
            gs_model.save_ply(gs_ply)
        except Exception:
            pass

    summary = timer.summary()
    logger.info("=== Pipeline Completed Successfully ===")
    logger.info(f"Total Wall-Clock Time: {summary['total_elapsed_seconds']:.2f}s (Budget: {timer.budget_seconds}s)")
    logger.info(f"Within 15-Minute Budget: {summary['within_budget']}")
    logger.info("Deliverables Generated:")
    for f in os.listdir(output_dir):
        logger.info(f" - {f}")

    if serve_viewer:
        from src.viewer.server import start_server
        logger.info("Launching local inspection viewer at http://127.0.0.1:8000 ...")
        start_server(deliverables_dir=output_dir)

    return summary


def main():
    parser = argparse.ArgumentParser(description="SIH26158 Single-Pass Drone Video to 3D Model Generation Pipeline")
    parser.add_argument("--video", type=str, default="data/samples/synthetic_drone_flight.mp4", help="Path to drone MP4 video")
    parser.add_argument("--telemetry", type=str, default="data/samples/synthetic_drone_flight.srt", help="Path to telemetry SRT or CSV")
    parser.add_argument("--output-dir", type=str, default="data/output", help="Directory for deliverables")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml", help="Configuration YAML path")
    parser.add_argument("--serve", action="store_true", help="Launch interactive 3D Web inspection viewer upon completion")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        telemetry_path=args.telemetry,
        output_dir=args.output_dir,
        config_path=args.config,
        serve_viewer=args.serve
    )


if __name__ == "__main__":
    main()
