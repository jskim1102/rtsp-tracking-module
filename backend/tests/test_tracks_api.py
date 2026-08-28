from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.main as main_module
import app.tracks_api as tracks_api
from app.inference import InferenceResult
from app.streaming.manager import StreamManager
from app.tracking.base import Track
from app.tracking.gallery import TrackGallery


class FakeWorker:
    def __init__(self) -> None:
        self.model = "sam3"
        self.restarts = 0

    def get_status(self) -> dict:
        return {"enabled": True, "model": self.model, "device": "cpu"}

    def get_pool_duty(self) -> float:
        return 0.0

    def set_model(self, name: str) -> bool:
        if name == self.model:
            return False
        self.model = name
        self.restarts += 1
        return True

    def restart(self) -> None:
        self.restarts += 1


@pytest.fixture()
def api(tmp_path, monkeypatch):
    manager = StreamManager()
    manager._worker = FakeWorker()
    manager._gallery = TrackGallery(tmp_path)
    manager._captures = {"cam-a": object()}
    manager._gallery.observe(
        InferenceResult(
            source_id="cam-a",
            timestamp=12.5,
            tracks=[Track(7, (0, 0, 3, 3), None, 0.9, True)],
            new_track_crops={7: np.full((3, 3, 3), 255, dtype=np.uint8)},
        )
    )
    monkeypatch.setattr(tracks_api, "stream_manager", manager)
    app = FastAPI()
    app.include_router(tracks_api.tracks_router)
    with TestClient(app) as client:
        yield client, manager


def test_main_includes_tracks_and_tracker_routes():
    paths = {route.path for route in main_module.app.routes}

    assert "/api/tracks" in paths
    assert "/api/tracks/{track_id}/thumb" in paths
    assert "/api/tracks/reset" in paths
    assert "/api/tracker" in paths


def test_tracks_list_and_camera_scoped_jpeg_thumbnail(api):
    client, _ = api

    response = client.get("/api/tracks")
    thumb = client.get("/api/tracks/7/thumb", params={"camera_id": "cam-a"})

    assert response.status_code == 200
    assert response.json() == {
        "active": 1,
        "total": 1,
        "items": [
            {
                "track_id": 7,
                "camera_id": "cam-a",
                "first_seen": 12.5,
                "thumb_url": "/api/tracks/7/thumb?camera_id=cam-a",
                "active": True,
            }
        ],
    }
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"
    assert thumb.content.startswith(b"\xff\xd8")


def test_missing_or_ambiguous_thumbnail_returns_404(api):
    client, manager = api
    manager._gallery.observe(
        InferenceResult(
            source_id="cam-b",
            timestamp=13.0,
            tracks=[Track(7, (0, 0, 2, 2), None, 0.8, True)],
            new_track_crops={7: np.ones((2, 2, 3), dtype=np.uint8)},
        )
    )

    assert client.get("/api/tracks/999/thumb").status_code == 404
    assert client.get("/api/tracks/7/thumb").status_code == 404
    assert client.get(
        "/api/tracks/7/thumb", params={"camera_id": "cam-b"}
    ).status_code == 200


def test_reset_restarts_all_sessions_clears_gallery_and_emits_reset(api):
    client, manager = api

    response = client.post("/api/tracks/reset")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert manager._worker.restarts == 1
    assert manager._gallery.snapshot()["total"] == 0
    assert manager.drain_tracker_events("cam-a") == [
        {
            "type": "tracker_reset",
            "camera_id": "cam-a",
            "reason": "manual_reset",
        }
    ]


def test_reset_also_clears_gallery_items_from_deleted_cameras(api):
    client, manager = api
    manager._gallery.observe(
        InferenceResult(
            source_id="deleted-cam",
            timestamp=13.0,
            tracks=[Track(8, (0, 0, 2, 2), None, 0.8, True)],
            new_track_crops={8: np.ones((2, 2, 3), dtype=np.uint8)},
        )
    )
    deleted_thumb = manager._gallery.thumbnail(8, "deleted-cam")

    response = client.post("/api/tracks/reset")

    assert response.status_code == 200
    assert manager._gallery.snapshot() == {"active": 0, "total": 0, "items": []}
    assert deleted_thumb is not None and not deleted_thumb.exists()


def test_tracker_get_and_put_expose_registry_and_reset_on_actual_change(
    api, monkeypatch
):
    client, manager = api
    monkeypatch.setattr(tracks_api, "available_trackers", lambda: ["sam3", "motip"])
    monkeypatch.setattr(
        tracks_api,
        "tracker_capabilities",
        lambda name: {"concept_prompt": name == "sam3"},
    )

    current = client.get("/api/tracker")
    changed = client.put("/api/tracker", json={"name": "motip"})

    assert current.json() == {
        "name": "sam3",
        "available": ["sam3", "motip"],
        "capabilities": {"concept_prompt": True},
    }
    assert changed.status_code == 200
    assert changed.json()["name"] == "motip"
    assert manager._worker.restarts == 1
    assert manager._gallery.snapshot()["total"] == 0
    assert manager.drain_tracker_events("cam-a")[0]["reason"] == "tracker_switch"


def test_tracker_put_rejects_unavailable_or_extra_fields(api):
    client, _ = api

    assert client.put("/api/tracker", json={"name": "storm"}).status_code == 400
    assert client.put(
        "/api/tracker",
        json={"name": "sam3", "prompt": "car"},
    ).status_code == 422
