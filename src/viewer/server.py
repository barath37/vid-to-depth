import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


def create_app(deliverables_dir: str = "data/output"):
    if deliverables_dir in ("data/output", "data\\output") and not os.path.exists(os.path.join(deliverables_dir, "reconstruction.glb")):
        candidate = os.path.join("data", "output", "Family_run")
        if os.path.exists(os.path.join(candidate, "reconstruction.glb")):
            deliverables_dir = candidate

    app = FastAPI(title="SIH26158 3D Inspection Viewer", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        html_file = os.path.join(static_dir, "index.html")
        if os.path.exists(html_file):
            with open(html_file, "r", encoding="utf-8") as f:
                return f.read()
        return "<h3>SIH26158 Viewer running. static/index.html not found.</h3>"

    @app.get("/api/deliverables")
    def list_deliverables():
        if not os.path.exists(deliverables_dir):
            return {"files": []}
        files = os.listdir(deliverables_dir)
        return {"files": files}

    @app.get("/api/download/{filename}")
    def download_file(filename: str):
        file_path = os.path.join(deliverables_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)

    return app


def start_server(deliverables_dir: str = "data/output", host: str = "127.0.0.1", port: int = 8000):
    app = create_app(deliverables_dir)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
