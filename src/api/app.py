"""FastAPI application for the local inspection dashboard."""

from __future__ import annotations

import os
import json
from contextlib import asynccontextmanager
from datetime import datetime
from time import sleep

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.camera import CameraInspectionService
from src.api.image_analysis import ImageUploadError, PartPreviewImageAnalyzer, StillImageAnalyzer, decode_uploaded_image
from src.api.domain import FEATURE_CAPABILITIES, InspectionState, TrackingConfig
from src.api.export import DEFAULT_EXPORT_FIELDS, EXPORT_FIELDS, filter_events, make_csv, make_xlsx
from src.api.runtime import PART_PREVIEW_MODE, resolve_part_weights_path, resolve_runtime_mode
from src.inference.yolo_fish_detector import YoloFishDetector, resolve_yolo_fish_detector_model_path
from src.inference.yolo_quality_model import YoloQualityModel, resolve_yolo_quality_model_path
from src.inference.part_model import YoloPartModel
from src.preprocessing.dataset_utils import load_yaml_file, project_root


REPO_ROOT = project_root()
FRONTEND_DIR = REPO_ROOT / "frontend"
PART_CONFIG = load_yaml_file(REPO_ROOT / "configs" / "yolov8.yaml")
FISH_DETECTION_CONFIDENCE = float(os.getenv("FISH_DETECTION_CONFIDENCE", "0.5"))
FISH_DETECTOR_DEVICE = os.getenv("FISH_DETECTOR_DEVICE", "auto")
FISH_DETECTOR_PATH = resolve_yolo_fish_detector_model_path(REPO_ROOT, os.getenv("FISH_DETECTOR_MODEL_PATH"))
FISH_DETECTOR_IMAGE_SIZE = int(os.getenv("FISH_DETECTOR_IMAGE_SIZE", "640"))
FISH_QUALITY_CONFIDENCE = float(os.getenv("FISH_QUALITY_CONFIDENCE", os.getenv("FISH_SEGMENTATION_CONFIDENCE", "0.25")))
FISH_QUALITY_DEVICE = os.getenv("FISH_QUALITY_DEVICE", os.getenv("FISH_SEGMENTER_DEVICE", FISH_DETECTOR_DEVICE))
FISH_QUALITY_PATH = resolve_yolo_quality_model_path(REPO_ROOT, os.getenv("FISH_QUALITY_MODEL_PATH", os.getenv("FISH_SEGMENTER_MODEL_PATH")))
FISH_QUALITY_IMAGE_SIZE = int(os.getenv("FISH_QUALITY_IMAGE_SIZE", "640"))
FISH_QUALITY_INTERVAL = int(os.getenv("FISH_QUALITY_INTERVAL", os.getenv("FISH_SEGMENTATION_INTERVAL", "3")))
FISH_QUALITY_ROI_PADDING = int(os.getenv("FISH_QUALITY_ROI_PADDING", os.getenv("FISH_SEGMENTATION_ROI_PADDING", "0")))
FISH_UPLOAD_MAX_BYTES = int(os.getenv("FISH_UPLOAD_MAX_BYTES", str(10 * 1024 * 1024)))
FISH_UPLOAD_MAX_DIMENSION = int(os.getenv("FISH_UPLOAD_MAX_DIMENSION", "4096"))
FISH_UPLOAD_MAX_PIXELS = int(os.getenv("FISH_UPLOAD_MAX_PIXELS", str(16 * 1024 * 1024)))
PART_CONFIDENCE = float(os.getenv("LEMURU_PART_CONFIDENCE", os.getenv("LEMURU_CONFIDENCE", "0.25")))
PART_IMAGE_SIZE = int(PART_CONFIG.get("imgsz", 640))
CAMERA_INDEX = int(os.getenv("LEMURU_CAMERA_INDEX", "0"))
RUNTIME_MODE = resolve_runtime_mode(os.getenv("LEMURU_MODE"))
PART_WEIGHTS_PATH = resolve_part_weights_path(REPO_ROOT, os.getenv("LEMURU_PART_WEIGHTS"))
UPLOAD_TEST_WEIGHTS_PATH = (REPO_ROOT / "models" / "yolo_parts" / "yolo26n_seg_exp4" / "weights" / "exp-4.pt").resolve()
UPLOAD_TEST_CONFIDENCE = float(os.getenv("LEMURU_UPLOAD_TEST_CONFIDENCE", str(PART_CONFIDENCE)))
TRACKING_CONFIG = TrackingConfig(
    tracker=os.getenv("LEMURU_TRACKER", "bytetrack.yaml"),
    line_orientation=os.getenv("LEMURU_LINE_ORIENTATION", "vertical").lower(),
    line_position=float(os.getenv("LEMURU_LINE_POSITION", "0.65")),
    conveyor_direction=os.getenv("LEMURU_CONVEYOR_DIRECTION", "left_to_right").lower(),
    track_timeout=float(os.getenv("LEMURU_TRACK_TIMEOUT", "1.5")),
    history_limit=int(os.getenv("LEMURU_HISTORY_LIMIT", "25")),
)

state = InspectionState(TRACKING_CONFIG, runtime_mode=RUNTIME_MODE)
quality_model: YoloQualityModel | None = None
if RUNTIME_MODE == PART_PREVIEW_MODE:
    state.confidence_threshold = PART_CONFIDENCE
    model = YoloPartModel(PART_WEIGHTS_PATH, confidence=PART_CONFIDENCE, imgsz=PART_IMAGE_SIZE)
else:
    state.confidence_threshold = FISH_DETECTION_CONFIDENCE
    state.set_quality_confidence_threshold(FISH_QUALITY_CONFIDENCE)
    model = YoloFishDetector(
        FISH_DETECTOR_PATH,
        confidence_threshold=FISH_DETECTION_CONFIDENCE,
        device=FISH_DETECTOR_DEVICE,
        image_size=FISH_DETECTOR_IMAGE_SIZE,
        tracker_config=TRACKING_CONFIG.tracker,
    )
    quality_model = YoloQualityModel(
        FISH_QUALITY_PATH,
        confidence_threshold=FISH_QUALITY_CONFIDENCE,
        device=FISH_QUALITY_DEVICE,
        image_size=FISH_QUALITY_IMAGE_SIZE,
    )
service = CameraInspectionService(
    state, model, camera_index=CAMERA_INDEX, runtime_mode=RUNTIME_MODE,
    quality_model=quality_model, quality_interval=FISH_QUALITY_INTERVAL,
    quality_roi_padding=FISH_QUALITY_ROI_PADDING,
)
image_analyzer = StillImageAnalyzer(model, quality_model, roi_padding=FISH_QUALITY_ROI_PADDING) if RUNTIME_MODE != PART_PREVIEW_MODE else None
upload_test_model = YoloPartModel(UPLOAD_TEST_WEIGHTS_PATH, confidence=UPLOAD_TEST_CONFIDENCE, imgsz=PART_IMAGE_SIZE) if RUNTIME_MODE == PART_PREVIEW_MODE else None
upload_test_analyzer = PartPreviewImageAnalyzer(upload_test_model) if upload_test_model is not None else None


@asynccontextmanager
async def lifespan(_: FastAPI):
    if model.load():
        mode_message = "Raw part preview ready; fish counting is disabled. " if RUNTIME_MODE == PART_PREVIEW_MODE else ""
        tracker_detail = f" Tracker: {model.tracker_backend}." if hasattr(model, "tracker_backend") else ""
        state.set_model("ready", model.name, str(model.weights_path), f"{mode_message}Model ready on {model.device}.{tracker_detail}")
    else:
        state.set_model("unavailable", None, None, model.error or "Model is unavailable.")
    if quality_model is None:
        state.set_segmenter("disabled", None, None, "Part-preview mode does not run the Model 2 quality pipeline.")
    elif quality_model.load():
        state.set_segmenter("ready", quality_model.name, str(quality_model.weights_path), f"Quality Model ready on {quality_model.device}.")
    else:
        state.set_segmenter("unavailable", None, str(quality_model.weights_path) if quality_model.weights_path else None, quality_model.error or "Quality Model is unavailable.")
    if upload_test_model is not None:
        # This preview-only model never participates in the production pipeline.
        upload_test_model.load()
    yield
    service.stop()


app = FastAPI(title="Sardinella Lemuru Fish Detection API", version="3.0.0", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


def _feature_counters(events: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Count fish with real final Model 2 regional evidence, never fake defects."""

    observed_regions: dict[str, int] = {name: 0 for name in FEATURE_CAPABILITIES}
    for event in events:
        analysis = event.get("analysis")
        part_results = analysis.get("part_results") if isinstance(analysis, dict) and isinstance(analysis.get("part_results"), dict) else {}
        for region in observed_regions:
            result = part_results.get(region) if isinstance(part_results, dict) else None
            if isinstance(result, dict) and result.get("present") is True:
                observed_regions[region] += 1
    counters: dict[str, dict[str, object]] = {}
    for name, capability in FEATURE_CAPABILITIES.items():
        counters[name] = {
            "count": observed_regions[name],
            "available": bool(capability["available"]),
            "source": capability["source"],
        }
    return counters


def _analytics(events: list[dict[str, object]], snapshot: dict[str, object]) -> dict[str, object]:
    """Build dashboard analytics from real current-session inspection events."""

    quality_counters = snapshot["quality_counters"]
    assert isinstance(quality_counters, dict)
    chronological = list(reversed(events))
    throughput = [
        {"timestamp": event.get("timestamp"), "total": index}
        for index, event in enumerate(chronological, start=1)
    ]
    grade_confidences = [
        float(event["quality_confidence"]) * 100
        for event in chronological
        if event.get("quality_confidence") is not None
    ]
    detection_confidences = [float(event["final_confidence"]) * 100 for event in chronological if event.get("final_confidence") is not None]
    processing_times = [
        float(event["processing_time_ms"])
        for event in chronological
        if event.get("processing_time_ms") is not None
    ]
    part_quality: dict[str, list[float]] = {"Head": [], "Body": [], "Tail": []}
    color_trend: list[dict[str, object]] = []
    for event in chronological:
        analysis = event.get("analysis")
        part_results = analysis.get("part_results") if isinstance(analysis, dict) and isinstance(analysis.get("part_results"), dict) else {}
        colors = analysis.get("color") if isinstance(analysis, dict) and isinstance(analysis.get("color"), dict) else {}
        for region in part_quality:
            result = part_results.get(region) if isinstance(part_results, dict) else None
            confidence = result.get("grade_confidence") if isinstance(result, dict) else None
            if confidence is not None:
                part_quality[region].append(float(confidence) * 100)
        whole = colors.get("Whole Fish") if isinstance(colors, dict) else None
        if isinstance(whole, dict) and whole.get("mean_hue_deg") is not None:
            color_trend.append({
                "timestamp": event.get("timestamp"),
                "mean_hue_deg": round(float(whole["mean_hue_deg"]), 2),
                "yellow_ratio_proxy": round(float(whole.get("yellow_ratio_proxy") or 0) * 100, 2),
            })
    return {
        "has_data": bool(events),
        "grade_distribution": {name: int(value) for name, value in quality_counters.items()},
        "feature_counts": _feature_counters(events),
        "throughput": throughput,
        "confidence_over_time": [
            {"timestamp": event.get("timestamp"), "confidence": round(float(event["quality_confidence"]) * 100, 1)}
            for event in chronological
            if event.get("quality_confidence") is not None
        ],
        "average_grade_confidence": round(sum(grade_confidences) / len(grade_confidences), 1) if grade_confidences else None,
        "average_detection_confidence": round(sum(detection_confidences) / len(detection_confidences), 1) if detection_confidences else None,
        "average_processing_time_ms": round(sum(processing_times) / len(processing_times), 1) if processing_times else None,
        "average_part_quality_scores": {
            region: round(sum(values) / len(values), 1) if values else None
            for region, values in part_quality.items()
        },
        "color_trend": color_trend,
        "average_whole_fish_hue": round(sum(item["mean_hue_deg"] for item in color_trend) / len(color_trend), 2) if color_trend else None,
    }


def _status_payload() -> dict[str, object]:
    snapshot = state.snapshot()
    detector_threshold = getattr(model, "confidence_threshold", getattr(model, "confidence", state.confidence_threshold))
    snapshot["detection_confidence_threshold"] = state.confidence_threshold
    snapshot["default_confidence_threshold"] = PART_CONFIDENCE if RUNTIME_MODE == PART_PREVIEW_MODE else FISH_DETECTION_CONFIDENCE
    snapshot["default_quality_confidence_threshold"] = FISH_QUALITY_CONFIDENCE
    snapshot["feature_counters"] = _feature_counters(state.tracking.archive_history())
    snapshot["upload_analysis_available"] = bool((image_analyzer is not None and model.model is not None) or (upload_test_model is not None and upload_test_model.model is not None))
    snapshot["upload_analysis_mode"] = "whole_fish" if image_analyzer is not None and model.model is not None else "part_preview"
    snapshot["upload_analysis_label"] = (
        "Whole-fish image analysis" if image_analyzer is not None and model.model is not None else "Raw part-segmentation test"
    )
    snapshot["model_info"] = {
        "detector": {
            "name": state.model_name or "Fish detector",
            "status": state.model_status,
            "checkpoint_resolved": bool(getattr(model, "weights_path", None)),
            "device": getattr(model, "device", "unresolved"),
            "confidence_threshold": detector_threshold,
        },
        "quality": {
            "name": state.segmenter_name or "Quality Model",
            "status": state.segmenter_status,
            "checkpoint_resolved": bool(getattr(quality_model, "weights_path", None)),
            "device": getattr(quality_model, "device", "unresolved"),
            "confidence_threshold": getattr(quality_model, "confidence_threshold", None),
            "class_names": list(getattr(quality_model, "class_names", {}).values()),
            "supports_masks": bool(getattr(quality_model, "supports_masks", False)),
            "diagnostics": quality_model.diagnostics() if quality_model is not None else {},
        },
    }
    snapshot["model_info"]["detector"].update({
        "class_names": list(getattr(model, "class_names", ())),
        "supports_masks": bool(getattr(model, "supports_masks", False)),
        "diagnostics": model.diagnostics() if hasattr(model, "diagnostics") else {},
    })
    # Retain the old response member for clients on a prior dashboard build.
    snapshot["model_info"]["segmenter"] = snapshot["model_info"]["quality"]
    quality_diagnostics = snapshot["model_info"]["quality"].get("diagnostics", {})
    snapshot["grading_rules"] = quality_diagnostics.get("grading_config", {}) if isinstance(quality_diagnostics, dict) else {}
    snapshot["analytics"] = _analytics(state.tracking.archive_history(), snapshot)
    return snapshot


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/status")
def get_status() -> dict[str, object]:
    return _status_payload()


@app.post("/api/inspection/start")
def start_inspection() -> dict[str, object]:
    started = service.start()
    return {"started": started, **_status_payload()}


@app.post("/api/inspection/stop")
def stop_inspection() -> dict[str, object]:
    stopped = service.stop()
    return {"stopped": stopped, **_status_payload()}


@app.post("/api/session/reset")
def reset_session() -> dict[str, object]:
    service.reset_session()
    return {"reset": True, **_status_payload()}


@app.get("/api/history")
def get_history(
    page: int = 1,
    page_size: int = 25,
    search: str | None = None,
    grade: str | None = None,
    feature: str | None = None,
    min_confidence: float | None = None,
) -> dict[str, object]:
    """Paginate the current session archive without implying disk persistence."""

    if page < 1 or not 1 <= page_size <= 100:
        raise HTTPException(status_code=422, detail="page must be positive and page_size must be between 1 and 100.")
    events = state.tracking.archive_history()
    if search:
        needle = search.casefold().replace("fish", "").replace("#", "").strip()
        events = [event for event in events if needle in str(event.get("track_id", "")).casefold()]
    if grade and grade.casefold() not in {"all", ""}:
        events = [event for event in events if (event.get("quality") or "Ungraded").casefold() == grade.casefold()]
    if feature and feature.casefold() not in {"all", ""}:
        events = [
            event for event in events
            if any(isinstance(part, dict) and str(part.get("region", "")).casefold() == feature.casefold() for part in (event.get("parts") or []))
        ]
    if min_confidence is not None:
        if not 0 <= min_confidence <= 100:
            raise HTTPException(status_code=422, detail="min_confidence must be between 0 and 100.")
        events = [
            event
            for event in events
            if float(event.get("quality_confidence") or event.get("final_confidence") or 0) * 100 >= min_confidence
        ]
    total = len(events)
    start = (page - 1) * page_size
    snapshot = state.snapshot()
    return {
        "items": events[start : start + page_size],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "latest_event": snapshot["latest_event"],
        "recent_history": snapshot["recent_history"],
    }


@app.get("/api/analytics")
def get_analytics() -> dict[str, object]:
    snapshot = state.snapshot()
    return _analytics(state.tracking.archive_history(), snapshot)


@app.get("/api/settings")
def get_settings() -> dict[str, object]:
    return {
        "detection_confidence_threshold": state.confidence_threshold,
        "quality_confidence_threshold": state.quality_confidence_threshold,
        "display_settings": state.current_display_settings(),
        "advanced_thresholds_supported": True,
    }


@app.post("/api/settings")
async def update_settings(request: Request) -> dict[str, object]:
    """Apply independent Model 1/Model 2 thresholds and live overlays."""

    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Settings must be valid JSON.") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Settings must be a JSON object.")
    detector_setting = payload.get("detection_confidence_threshold", payload.get("confidence_threshold"))
    if detector_setting is not None:
        try:
            threshold = float(detector_setting)
            state.set_confidence_threshold(threshold)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="detection_confidence_threshold must be a number between 0 and 1.") from None
        if hasattr(model, "confidence_threshold"):
            model.confidence_threshold = threshold
        elif hasattr(model, "confidence"):
            model.confidence = threshold
    if "quality_confidence_threshold" in payload:
        try:
            threshold = float(payload["quality_confidence_threshold"])
            state.set_quality_confidence_threshold(threshold)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="quality_confidence_threshold must be a number between 0 and 1.") from None
        if quality_model is not None:
            quality_model.confidence_threshold = threshold
    if "display_settings" in payload:
        display = payload["display_settings"]
        if not isinstance(display, dict):
            raise HTTPException(status_code=422, detail="display_settings must be an object.")
        try:
            state.set_display_settings(display)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"updated": True, **_status_payload()}


@app.post("/api/export")
async def export_history(request: Request) -> Response:
    """Generate a current-session CSV or XLSX file without a fake export flow."""

    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Export settings must be valid JSON.") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Export settings must be a JSON object.")
    export_format = str(payload.get("format", "csv")).lower()
    if export_format not in {"csv", "xlsx"}:
        raise HTTPException(status_code=422, detail="format must be csv or xlsx.")
    selected_fields = payload.get("fields", list(DEFAULT_EXPORT_FIELDS))
    if not isinstance(selected_fields, list) or not selected_fields:
        raise HTTPException(status_code=422, detail="Choose at least one export field.")
    fields = tuple(str(field) for field in selected_fields)
    if payload.get("complete_analysis") is True:
        fields = DEFAULT_EXPORT_FIELDS
    if any(field not in EXPORT_FIELDS for field in fields):
        raise HTTPException(status_code=422, detail="One or more export fields are unsupported.")
    try:
        events = filter_events(
            state.tracking.archive_history(),
            range_name=str(payload.get("range", "current_session")),
            start_date=payload.get("start_date"),
            end_date=payload.get("end_date"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    snapshot = state.snapshot()
    analytics = _analytics(events, snapshot)
    quality = snapshot["quality_counters"]
    assert isinstance(quality, dict)
    summary = {
        "Total Fish": len(events),
        "Average Grade Confidence": f"{analytics['average_grade_confidence']:.1f}%" if analytics["average_grade_confidence"] is not None else "",
        "Average Detection Confidence": f"{analytics['average_detection_confidence']:.1f}%" if analytics["average_detection_confidence"] is not None else "",
        "Average Processing Time (ms)": analytics["average_processing_time_ms"] or "",
        "Average Whole-Fish Hue (deg)": analytics["average_whole_fish_hue"] if analytics["average_whole_fish_hue"] is not None else "",
    }
    part_averages = analytics["average_part_quality_scores"]
    assert isinstance(part_averages, dict)
    summary.update({f"Average {region} Grade Confidence": value if value is not None else "" for region, value in part_averages.items()})
    summary.update({f"{name} Count": value for name, value in quality.items()})
    summary.update({f"{name} Observed": detail["count"] for name, detail in _feature_counters(events).items()})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if export_format == "csv":
        body = make_csv(events, fields)
        media_type, extension = "text/csv; charset=utf-8", "csv"
    else:
        try:
            rules = _status_payload().get("grading_rules")
            body = make_xlsx(events, fields, summary, rules if isinstance(rules, dict) else None)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        media_type, extension = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="sardinella_inspection_{timestamp}.{extension}"'},
    )


@app.post("/api/analyze-image")
async def analyze_image(request: Request) -> dict[str, object]:
    """Analyze one supported still image in memory without touching live state."""

    filename = request.headers.get("x-filename", "")
    if not filename:
        raise HTTPException(status_code=400, detail="An uploaded image filename is required.")
    declared_length = request.headers.get("content-length")
    if declared_length and declared_length.isdigit() and int(declared_length) > FISH_UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"The uploaded image exceeds the {FISH_UPLOAD_MAX_BYTES} byte limit.")
    try:
        frame = decode_uploaded_image(
            await request.body(),
            filename,
            max_bytes=FISH_UPLOAD_MAX_BYTES,
            max_dimension=FISH_UPLOAD_MAX_DIMENSION,
            max_pixels=FISH_UPLOAD_MAX_PIXELS,
        )
        if image_analyzer is not None and model.model is not None:
            return image_analyzer.analyze(frame)
        if upload_test_model is not None and upload_test_model.model is not None and upload_test_analyzer is not None:
            return upload_test_analyzer.analyze(frame)
        detail = (upload_test_model.error if upload_test_model is not None else None) or model.error or "No local upload-analysis model is available."
        raise HTTPException(status_code=503, detail=detail)
    except ImageUploadError as exc:
        status = 413 if "byte limit" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
