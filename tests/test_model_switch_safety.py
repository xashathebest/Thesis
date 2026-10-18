"""Model changes must not race with or alter an active inspection."""
from __future__ import annotations

import asyncio
import importlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

app_module = importlib.import_module("src.api.app")


@pytest.mark.parametrize("status", ["starting", "running"])
@pytest.mark.parametrize("route", ["select", "settings", "upload"])
def test_all_model_change_routes_reject_active_inspections(monkeypatch, status, route):
    quality = SimpleNamespace(active_model_id="old", switch_model=MagicMock())
    monkeypatch.setattr(app_module, "quality_model", quality)
    monkeypatch.setattr(app_module.state, "snapshot", lambda: {"inspection_status": status})
    registry = SimpleNamespace(get_model=lambda _id: object(), validate_compatibility=lambda *_: (True, None))
    monkeypatch.setattr("src.inference.model_registry.get_model_registry", lambda: registry)
    client = TestClient(app_module.app)
    if route == "select":
        response = client.post("/api/models/select", json={"model_id": "new"})
    elif route == "settings":
        response = client.post("/api/settings", json={"model2_id": "new"})
    else:
        ok, image = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
        assert ok
        response = client.post("/api/analyze-image", content=image.tobytes(), headers={"x-filename": "fish.jpg", "x-model-id": "new"})
    assert response.status_code == 409
    quality.switch_model.assert_not_called()


def test_start_and_model_switch_are_serialized(monkeypatch):
    status = {"inspection_status": "stopped"}
    entered, release = threading.Event(), threading.Event()
    quality = SimpleNamespace(active_model_id="old", switch_model=MagicMock())
    monkeypatch.setattr(app_module, "quality_model", quality)
    monkeypatch.setattr(app_module.state, "snapshot", lambda: status.copy())
    monkeypatch.setattr(app_module, "_status_payload", lambda: {})
    monkeypatch.setattr(app_module.inspection_sessions, "start", lambda _metadata: None)
    monkeypatch.setattr(app_module, "_session_metadata", lambda: {})
    def start():
        entered.set()
        assert release.wait(5)
        status["inspection_status"] = "starting"
        return True
    monkeypatch.setattr(app_module.service, "start", start)
    registry = SimpleNamespace(get_model=lambda _id: object(), validate_compatibility=lambda *_: (True, None))
    monkeypatch.setattr("src.inference.model_registry.get_model_registry", lambda: registry)
    class Request:
        async def json(self):
            return {"model_id": "new"}
    result = []
    def switch():
        try:
            asyncio.run(app_module.select_model(Request()))
        except HTTPException as exc:
            result.append(exc.status_code)
    starter = threading.Thread(target=app_module.start_inspection)
    changer = threading.Thread(target=switch)
    starter.start()
    assert entered.wait(5)
    changer.start()
    release.set()
    starter.join(5)
    changer.join(5)
    assert not starter.is_alive() and not changer.is_alive()
    assert result == [409]
    quality.switch_model.assert_not_called()


def test_bad_upload_cannot_switch_models(monkeypatch):
    quality = SimpleNamespace(active_model_id="old", switch_model=MagicMock())
    monkeypatch.setattr(app_module, "quality_model", quality)
    client = TestClient(app_module.app)
    response = client.post("/api/analyze-image", content=b"invalid image", headers={"x-filename": "fish.jpg", "x-model-id": "new"})
    assert response.status_code == 400
    quality.switch_model.assert_not_called()


def test_upload_with_current_model_remains_available_during_inspection(monkeypatch):
    quality = SimpleNamespace(active_model_id="current", switch_model=MagicMock())
    monkeypatch.setattr(app_module, "quality_model", quality)
    monkeypatch.setattr(app_module.state, "snapshot", lambda: {"inspection_status": "running"})
    monkeypatch.setattr(app_module, "model", SimpleNamespace(model=object()))
    monkeypatch.setattr(app_module, "image_analyzer", SimpleNamespace(analyze=lambda _frame: {"ok": True}))
    ok, image = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    client = TestClient(app_module.app)
    response = client.post("/api/analyze-image", content=image.tobytes(), headers={"x-filename": "fish.jpg", "x-model-id": "current"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    quality.switch_model.assert_not_called()
