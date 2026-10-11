import open3d as o3d
import numpy as np
import trimesh

from src.common.types import PointCloudModel


class PoissonSurfaceMesher:
    """Decoupled Poisson Surface Reconstruction (§1 Erratum #6 & §10.2 of Spec).
    Meshes the dense pseudo-LiDAR point cloud directly using Open3D.
    Trims low-density boundary vertices to eliminate ballooning over unobserved gaps.
    """

    def __init__(
        self,
        depth: int = 9,
        voxel_size_m: float = 0.15,
        density_trim_quantile: float = 0.02
    ):
        self.depth = depth
        self.voxel_size_m = voxel_size_m
        self.density_trim_quantile = density_trim_quantile

    def reconstruct(self, point_cloud: PointCloudModel) -> trimesh.Trimesh:
        """Runs voxel downsampling, normal orientation, and Screened Poisson Reconstruction.
        Returns: trimesh.Trimesh polygonal mesh.
        """
        pts = point_cloud.points
        if len(pts) < 10:
            return trimesh.Trimesh()

        # Build Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        if point_cloud.colors is not None and len(point_cloud.colors) == len(pts):
            pcd.colors = o3d.utility.Vector3dVector((point_cloud.colors / 255.0).astype(np.float64))

        if point_cloud.normals is not None and len(point_cloud.normals) == len(pts):
            pcd.normals = o3d.utility.Vector3dVector(point_cloud.normals.astype(np.float64))
        else:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_m * 2.5, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

        # Voxel downsample if point count is large
        if len(pts) > 5000:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size_m)

        # Poisson reconstruction
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.depth, linear_fit=True
        )

        # Density trimming (§10.2 of Spec): drop vertices in lowest density quantile
        densities_arr = np.asarray(densities)
        if len(densities_arr) > 0 and self.density_trim_quantile > 0.0:
            density_threshold = np.quantile(densities_arr, self.density_trim_quantile)
            vertices_to_remove = densities_arr < density_threshold
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_non_manifold_edges()

        # Convert to trimesh
        verts = np.asarray(mesh_o3d.vertices, dtype=np.float32)
        faces = np.asarray(mesh_o3d.triangles, dtype=np.int32)
        v_colors = (np.asarray(mesh_o3d.vertex_colors) * 255).astype(np.uint8) if mesh_o3d.has_vertex_colors() else None

        tri_mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_colors=v_colors,
            process=True
        )

        return tri_mesh
