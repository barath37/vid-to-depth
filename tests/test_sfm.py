import numpy as np
import pytest

from src.common.types import CameraPose
from src.sfm.feature_tracker import SequentialFeatureTracker
from src.sfm.pose_estimator import RelativePoseEstimator
from src.sfm.bundle_adjustment import GlobalBundleAdjuster
from src.sfm.metric_aligner import MetricTrajectoryAligner


def test_sequential_feature_tracker_matching():
    # Two synthetic images with overlapping textured blocks
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)

    # Put distinct corners
    np.random.seed(42)
    corners = np.random.randint(30, 170, (40, 2))
    for x, y in corners:
        cv2_rect1 = (x, y)
        img1[y-2:y+3, x-2:x+3] = 255
        # Shift in second image by dx=10, dy=5 (optical flow / camera shift)
        if 0 <= x+10 < 195 and 0 <= y+5 < 195:
            img2[y+3:y+8, x+8:x+13] = 255

    tracker = SequentialFeatureTracker(detector_type="ORB", n_features=500)
    feats1 = tracker.detect_and_compute(img1)
    feats2 = tracker.detect_and_compute(img2)

    matches = tracker.match_pairs(feats1, feats2)
    # Should find at least some valid correspondences
    assert len(matches) > 5


def test_relative_pose_and_triangulation():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

    # 20 3D points in front of camera
    np.random.seed(42)
    pts3d = np.random.uniform(-5.0, 5.0, (20, 3))
    pts3d[:, 2] = np.random.uniform(15.0, 25.0, 20) # Z positive (in front)

    # Camera 1 at origin
    R1 = np.eye(3)
    t1 = np.zeros(3)

    # Camera 2 translated along X by 1.5m
    R2 = np.eye(3)
    t2 = np.array([1.5, 0.0, 0.0])

    # Project to 2D image coordinates
    pts2d_1 = (K @ pts3d.T).T
    pts2d_1 = pts2d_1[:, :2] / pts2d_1[:, 2:]

    pts_cam2 = (pts3d - t2)
    pts2d_2 = (K @ pts_cam2.T).T
    pts2d_2 = pts2d_2[:, :2] / pts2d_2[:, 2:]

    estimator = RelativePoseEstimator(K)
    R_est, t_est, inliers = estimator.estimate_pose(pts2d_1, pts2d_2)

    # Camera 2 optical center c = -R.T @ t in world frame should align with [1, 0, 0]
    t_dir_true = np.array([1.0, 0.0, 0.0])
    cam2_center_dir = (-R_est.T @ t_est).flatten() / np.linalg.norm(t_est)
    assert np.dot(t_dir_true, cam2_center_dir) > 0.90

    # Triangulate points
    triangulated = estimator.triangulate_points(pts2d_1[inliers], pts2d_2[inliers], R_est, t_est)
    assert len(triangulated) == len(inliers)
    assert np.all(triangulated[:, 2] > 0.0) # All in front of camera


def test_global_bundle_adjustment():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

    # True 3D points
    np.random.seed(42)
    true_points = np.random.uniform(-3, 3, (15, 3))
    true_points[:, 2] += 20.0

    # Two camera poses
    R1 = np.eye(3)
    t1 = np.zeros(3)
    R2 = np.eye(3)
    t2 = np.array([1.0, 0.0, 0.0])

    poses = [CameraPose(0, R1, t1), CameraPose(1, R2, t2)]

    # Perfect 2D observations
    obs = []
    for pose in poses:
        pts_c = (pose.R @ true_points.T).T + pose.t
        p2d = (K @ pts_c.T).T
        p2d = p2d[:, :2] / p2d[:, 2:]
        obs.append(p2d)

    # Add small noise to initial 3D points
    noisy_points = true_points + np.random.normal(0, 0.1, true_points.shape)

    gba = GlobalBundleAdjuster(K)
    opt_poses, opt_points = gba.optimize(poses, noisy_points, obs)

    # Optimized points should be closer to true points than noisy points
    err_before = np.mean(np.linalg.norm(noisy_points - true_points, axis=1))
    err_after = np.mean(np.linalg.norm(opt_points - true_points, axis=1))
    assert err_after <= err_before


def test_metric_trajectory_aligner_with_observability_lock():
    # Simulates camera poses and EKF trajectory on straight flight line
    # Tests that roll about flight line is locked using gravity/ground constraints
    N = 10
    true_centers = np.column_stack([np.zeros(N), np.linspace(0, 60, N), np.full(N, 50.0)]) # Straight North

    # SfM poses with arbitrary scale (e.g. 0.2x) and coordinate frame
    sfm_scale = 0.2
    sfm_centers = true_centers * sfm_scale

    poses = [CameraPose(i, np.eye(3), -sfm_centers[i]) for i in range(N)]

    gravity = np.array([0.0, 0.0, -9.81])
    ground_normal = np.array([0.0, 0.0, 1.0])

    aligner = MetricTrajectoryAligner()
    scale, R, t, aligned_poses = aligner.align_to_ekf(
        poses,
        true_centers,
        gravity_vector=gravity,
        ground_normal=ground_normal
    )

    assert pytest.approx(1.0 / sfm_scale, abs=0.05) == scale
    aligned_centers = np.array([p.center for p in aligned_poses])
    rmse = np.sqrt(np.mean(np.sum((aligned_centers - true_centers)**2, axis=1)))
    assert rmse < 0.5 # Sub-meter metric trajectory alignment


def test_metric_trajectory_aligner_orbital_flight():
    # Simulates an orbital circular flight around a target (Point-of-Interest)
    N = 20
    theta = np.linspace(0, 1.5 * np.pi, N)
    radius = 40.0
    true_centers = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(N, 35.0)])

    sfm_scale = 0.5
    sfm_centers = true_centers * sfm_scale

    poses = [CameraPose(i, np.eye(3), -sfm_centers[i]) for i in range(N)]

    gravity = np.array([0.0, 0.0, -9.81])
    ground_normal = np.array([0.0, 0.0, 1.0])

    aligner = MetricTrajectoryAligner()
    scale, R, t, aligned_poses = aligner.align_to_ekf(
        poses,
        true_centers,
        gravity_vector=gravity,
        ground_normal=ground_normal
    )

    assert pytest.approx(1.0 / sfm_scale, abs=0.05) == scale
    aligned_centers = np.array([p.center for p in aligned_poses])
    rmse = np.sqrt(np.mean(np.sum((aligned_centers - true_centers)**2, axis=1)))
    assert rmse < 0.5
