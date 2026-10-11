import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from src.viewer.server import create_app


def test_viewer_server_routes():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy deliverable
        dummy_file = os.path.join(tmpdir, "model.glb")
        with open(dummy_file, "w") as f:
            f.write("glTF_dummy")

        app = create_app(deliverables_dir=tmpdir)
        client = TestClient(app)

        # Index route
        res_idx = client.get("/")
        assert res_idx.status_code == 200

        # Deliverables list
        res_deliv = client.get("/api/deliverables")
        assert res_deliv.status_code == 200
        data = res_deliv.json()
        assert "model.glb" in data["files"]

        # Download file
        res_dl = client.get("/api/download/model.glb")
        assert res_dl.status_code == 200
        assert res_dl.content == b"glTF_dummy"

        # Not found
        res_404 = client.get("/api/download/nonexistent.obj")
        assert res_404.status_code == 404
