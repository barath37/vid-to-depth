import json
import os
from typing import Dict, Optional
import numpy as np
import trimesh


def export_mesh_multi_format(
    mesh: trimesh.Trimesh,
    base_path: str,
    local_origin: Optional[np.ndarray] = None,
    epsg_code: int = 32643
) -> Dict[str, str]:
    """Exports 3D mesh into OBJ, PLY, and GLB formats (§5.1 & §11 of Spec).
    Subtracts local origin to prevent float32 precision jitter in 3D viewers.
    Generates companion georef.json with coordinate metadata.
    """
    os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
    obj_path = f"{base_path}.obj"
    ply_path = f"{base_path}.ply"
    glb_path = f"{base_path}.glb"
    json_path = f"{base_path}_georef.json"

    # Copy mesh so original is unchanged
    export_mesh = mesh.copy()

    # Precision safeguard (§11 of Spec): subtract local origin
    if local_origin is None:
        local_origin = np.zeros(3, dtype=np.float64)
    else:
        export_mesh.vertices -= local_origin

    # Export standard OBJ & PLY (ENU Z-up)
    export_mesh.export(obj_path)
    export_mesh.export(ply_path)

    # GLTF standard specifies Y-up, meters. Convert from ENU (Z-up) to Y-up
    # Transform: X_gltf = X_enu, Y_gltf = Z_enu, Z_gltf = -Y_enu
    R_enu_to_gltf = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    glb_mesh = export_mesh.copy()
    glb_mesh.apply_transform(R_enu_to_gltf)
    glb_mesh.export(glb_path)

    # Companion georeferencing JSON metadata
    georef_meta = {
        "format": "SIH26158 Georeference Sidecar",
        "epsg_code": epsg_code,
        "local_origin_offset": [float(x) for x in local_origin],
        "coordinate_convention": "ENU (East-North-Up)",
        "gltf_up_axis": "Y-up",
        "unit": "meters"
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(georef_meta, f, indent=2)

    return {
        "obj": obj_path,
        "ply": ply_path,
        "glb": glb_path,
        "georef_json": json_path
    }
