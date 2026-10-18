"""Opt-in local research-clip access, separate from grading and inference.

No recordings or metadata are shipped. This narrowly scoped local bearer-token
gate is not a multi-user identity provider and must not be exposed via a proxy.
"""
from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import secrets
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from src.preprocessing.dataset_utils import project_root

router = APIRouter(prefix="/api/expert-library", include_in_schema=False)
REPO_ROOT = project_root().resolve()
NO_STORE = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
CATEGORIES = {"body_weighting", "missing_head", "missing_tail", "belly_defects", "grade_distinctions"}
STATUSES = {"proposed", "expert-confirmed", "experimentally-validated"}
MAX_CLIP_BYTES = 100 * 1024 * 1024


def _deny() -> None:
    raise HTTPException(status_code=404, detail="Expert library unavailable or access denied.", headers=NO_STORE)


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _authorized_paths(request: Request) -> tuple[Path, Path]:
    if os.getenv("EXPERT_LIBRARY_ENABLED") != "1" or os.getenv("EXPERT_LIBRARY_SCOPE_VERIFIED") != "1":
        _deny()
    token = os.getenv("EXPERT_LIBRARY_RESEARCH_TOKEN", "")
    if len(token) < 32:
        _deny()
    authorization = request.headers.get("authorization", "")
    if not authorization.startswith("Bearer ") or not secrets.compare_digest(authorization[7:].encode("utf-8"), token.encode("utf-8")):
        _deny()
    try:
        if not request.client or not ipaddress.ip_address(request.client.host).is_loopback:
            _deny()
        hostname = urlsplit("http://" + request.headers.get("host", "")).hostname
    except ValueError:
        _deny()
    if "@" in request.headers.get("host", "") or hostname not in {"localhost", "127.0.0.1", "::1"}:
        _deny()
    if any(name.lower() == "forwarded" or name.lower().startswith("x-forwarded-") for name in request.headers):
        _deny()
    manifest_setting = os.getenv("EXPERT_LIBRARY_MANIFEST", "")
    media_setting = os.getenv("EXPERT_LIBRARY_MEDIA_ROOT", "")
    manifest, media = Path(manifest_setting), Path(media_setting)
    if not manifest_setting or not media_setting or not manifest.is_absolute() or not media.is_absolute():
        _deny()
    manifest, media = manifest.resolve(), media.resolve()
    if _inside(manifest, REPO_ROOT) or _inside(media, REPO_ROOT) or not media.is_dir():
        _deny()
    return manifest, media


def _filename(value: object) -> bool:
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= 255 and
            PureWindowsPath(value).name == value and PurePosixPath(value).name == value and value not in {".", ".."})


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def validate_manifest(value: object) -> list[dict]:
    if not isinstance(value, dict) or value.get("schema_version") != "expert-library-v1" or not isinstance(value.get("clips"), list):
        raise ValueError("Invalid manifest structure.")
    seen, clips = set(), []
    for clip in value["clips"]:
        if not isinstance(clip, dict):
            raise ValueError("Clip must be an object.")
        clip_id = clip.get("id")
        if not isinstance(clip_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", clip_id) or clip_id in seen:
            raise ValueError("Clip ID invalid or duplicated.")
        seen.add(clip_id)
        for field in ("rule_id", "title", "research_question", "reviewer", "claim_text"):
            if not isinstance(clip.get(field), str) or not clip[field].strip() or len(clip[field]) > 10000:
                raise ValueError("Required clip field is missing or too long.")
        if clip.get("category") not in CATEGORIES or clip.get("validation_status") not in STATUSES or clip.get("claim_kind") not in {"exact_quote", "paraphrase"}:
            raise ValueError("Invalid category, status, or claim kind.")
        if not _filename(clip.get("original_filename")) or not _filename(clip.get("clipped_filename")):
            raise ValueError("Filenames must not contain paths.")
        if Path(clip["clipped_filename"]).suffix.lower() not in {".mp4", ".webm"}:
            raise ValueError("Only mp4 and webm clips are supported.")
        try:
            date.fromisoformat(clip.get("consultation_date", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError("Consultation date must be an ISO calendar date.") from exc
        start, end, duration = (_number(clip.get(field)) for field in ("original_start_seconds", "original_end_seconds", "clipped_length_seconds"))
        if start is None or end is None or duration is None or start < 0 or end <= start or duration <= 0:
            raise ValueError("Clip time range is invalid.")
        interpretation = clip.get("researcher_interpretation", "")
        if not isinstance(interpretation, str) or len(interpretation) > 10000:
            raise ValueError("Invalid researcher interpretation.")
        clips.append(dict(clip))
    return clips


def _load(request: Request) -> tuple[list[dict], Path]:
    manifest_path, media_root = _authorized_paths(request)
    try:
        if manifest_path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Manifest too large.")
        clips = validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=503, detail="Expert library manifest is unavailable or invalid.", headers=NO_STORE) from None
    return clips, media_root


def _clip_path(root: Path, name: str) -> Path | None:
    try:
        path = (root / name).resolve()
        return path if _inside(path, root) and path.is_file() and path.stat().st_size <= MAX_CLIP_BYTES else None
    except OSError:
        return None


@router.get("/clips")
def list_clips(request: Request) -> JSONResponse:
    clips, _ = _load(request)
    fields = ("id", "rule_id", "title", "category", "validation_status")
    return JSONResponse({"clips": [{key: clip[key] for key in fields} for clip in clips]}, headers=NO_STORE)


@router.get("/clips/{clip_id}")
def clip_detail(clip_id: str, request: Request) -> JSONResponse:
    clips, media_root = _load(request)
    clip = next((item for item in clips if item["id"] == clip_id), None)
    if clip is None:
        _deny()
    fields = ("id", "rule_id", "title", "category", "research_question", "consultation_date", "original_filename",
              "original_start_seconds", "original_end_seconds", "clipped_filename", "clipped_length_seconds",
              "claim_kind", "claim_text", "researcher_interpretation", "validation_status", "reviewer")
    detail = {key: clip.get(key, "") for key in fields}
    detail["media_available"] = _clip_path(media_root, clip["clipped_filename"]) is not None
    return JSONResponse(detail, headers=NO_STORE)


@router.get("/clips/{clip_id}/media")
def clip_media(clip_id: str, request: Request) -> FileResponse:
    clips, media_root = _load(request)
    clip = next((item for item in clips if item["id"] == clip_id), None)
    if clip is None:
        _deny()
    path = _clip_path(media_root, clip["clipped_filename"])
    if path is None:
        _deny()
    media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "video/webm"
    return FileResponse(path, media_type=media_type, headers=NO_STORE)
