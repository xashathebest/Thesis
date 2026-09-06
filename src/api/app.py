"""FastAPI application for the local inspection dashboard."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import sleep

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.camera import CameraInspectionService
from src.api.domain import InspectionState, TrackingConfig
from src.api.model import YoloModel, resolve_weights_path
from src.preprocessing.dataset_utils import load_yaml_file, project_root


REPO_ROOT = project_root()
FRONTEND_DIR = REPO_ROOT / "frontend"
YOLO_CONFIG = load_yaml_file(REPO_ROOT / "configs" / "yolov8.yaml")
CONFIDENCE = float(os.getenv("LEMURU_CONFIDENCE", str(YOLO_CONFIG.get("confidence_threshold", 0.25))))
IMAGE_SIZE = int(YOLO_CONFIG.get("imgsz", 640))
MODEL_FAMILY = Path(str(YOLO_CONFIG.get("model", "yolov8n.yaml"))).stem
CAMERA_INDEX = int(os.getenv("LEMURU_CAMERA_INDEX", "0"))
WEIGHTS_PATH = resolve_weights_path(REPO_ROOT, os.getenv("LEMURU_WEIGHTS"))
TRACKING_CONFIG = TrackingConfig(
    tracker=os.getenv("LEMURU_TRACKER", "bytetrack.yaml"),
    line_orientation=os.getenv("LEMURU_LINE_ORIENTATION", "vertical").lower(),
    line_position=float(os.getenv("LEMURU_LINE_POSITION", "0.65")),
    conveyor_direction=os.getenv("LEMURU_CONVEYOR_DIRECTION", "left_to_right").lower(),
    track_timeout=float(os.getenv("LEMURU_TRACK_TIMEOUT", "1.5")),
    history_limit=int(os.getenv("LEMURU_HISTORY_LIMIT", "25")),
)

state = InspectionState(TRACKING_CONFIG)
state.confidence_threshold = CONFIDENCE
model = YoloModel(
    WEIGHTS_PATH,
    confidence_threshold=CONFIDENCE,
    imgsz=IMAGE_SIZE,
    model_family=MODEL_FAMILY,
    tracker=TRACKING_CONFIG.tracker,
)
service = CameraInspectionService(state, model, camera_index=CAMERA_INDEX)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if model.load():
        state.set_model("ready", model.name, str(model.weights_path), f"Model ready on {model.device}.")
    else:
        state.set_model("unavailable", None, None, model.error or "Model is unavailable.")
    yield
    service.stop()


app = FastAPI(title="Sardinella Lemuru Classification API", version="2.0.0", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/status")
def get_status() -> dict[str, object]:
    return state.snapshot()


@app.post("/api/inspection/start")
def start_inspection() -> dict[str, object]:
    started = service.start()
    return {"started": started, **state.snapshot()}


@app.post("/api/inspection/stop")
def stop_inspection() -> dict[str, object]:
    stopped = service.stop()
    return {"stopped": stopped, **state.snapshot()}


@app.post("/api/session/reset")
def reset_session() -> dict[str, object]:
    service.reset_session()
    return {"reset": True, **state.snapshot()}


@app.get("/api/history")
def get_history() -> dict[str, object]:
    snapshot = state.snapshot()
    return {"latest_event": snapshot["latest_event"], "recent_history": snapshot["recent_history"]}


def _mjpeg_frames():
    last_version = -1
    while True:
        frame, version = state.frame()
        if frame is not None and version != last_version:
            last_version = version
            yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
        sleep(0.025 if state.inspection_status == "running" else 0.2)


@app.get("/api/video-feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(_mjpeg_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
