from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.database as database
import app.ipcam as ipcam
from app.inference import InferenceResult
from app.streaming.manager import tracks_to_json
from app.tracking.base import Track


def _result() -> InferenceResult:
    return InferenceResult(
        source_id="cam-a",
        timestamp=5.5,
        frame_w=640,
        frame_h=360,
        tracks=[
            Track(
                track_id=7,
                bbox=(10, 20, 110, 220),
                polygon=[(10, 20), (110, 20), (110, 220)],
                score=0.94,
                is_new=True,
            )
        ],
    )


def test_tracks_payload_preserves_frame_seam_and_exact_track_fields():
    payload = json.loads(tracks_to_json(_result()))

    assert payload == {
        "type": "tracks",
        "timestamp": 5.5,
        "frame": {"w": 640, "h": 360},
        "items": [
            {
                "track_id": 7,
                "bbox": [10, 20, 110, 220],
                "polygon": [[10, 20], [110, 20], [110, 220]],
                "score": 0.94,
                "is_new": True,
            }
        ],
    }


def test_tracks_payload_keeps_nullable_polygon():
    result = _result()
    result.tracks = [Track(8, (1, 2, 3, 4), None, 0.5, False)]

    assert json.loads(tracks_to_json(result))["items"][0]["polygon"] is None


def test_camera_websocket_sends_events_and_tracks_on_the_same_connection(monkeypatch):
    result = _result()

    class FakeQuery:
        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return SimpleNamespace(stream_key="demo", rtsp_url="rtsp://camera/live")

    class FakeSession:
        def query(self, *args, **kwargs):
            return FakeQuery()

        def close(self):
            return None

    class FakeManager:
        def __init__(self) -> None:
            self.event_calls = 0
            self.stopped = False

        def start_capture(self, source_id, source):
            return True

        def stop_capture(self, source_id):
            self.stopped = True

        def drain_tracker_events(self, source_id):
            self.event_calls += 1
            if self.event_calls == 1:
                return [
                    {
                        "type": "new_track",
                        "track_id": 7,
                        "camera_id": source_id,
                        "first_seen": 5.5,
                        "thumb_url": "/api/tracks/7/thumb?camera_id=cam-a",
                    },
                    {
                        "type": "tracker_reset",
                        "camera_id": source_id,
                        "reason": "worker_respawn",
                    },
                ]
            return []

        def get_source_latest_tracks(self, source_id):
            return result

    fake_manager = FakeManager()
    monkeypatch.setattr(database, "SessionLocal", FakeSession)
    monkeypatch.setattr(ipcam, "stream_manager", fake_manager)
    app = FastAPI()
    app.include_router(ipcam.router)

    with TestClient(app).websocket_connect("/api/ipcams/demo/ws") as websocket:
        first = websocket.receive_json()
        second = websocket.receive_json()
        third = websocket.receive_json()

    assert [first["type"], second["type"], third["type"]] == [
        "new_track",
        "tracker_reset",
        "tracks",
    ]
    assert fake_manager.stopped is True
