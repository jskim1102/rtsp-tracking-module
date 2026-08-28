from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.streaming.manager import StreamManager


class FakeWorker:
    def __init__(self, *, alive: bool, model: str = "sam3") -> None:
        self.alive = alive
        self.model = model
        self.starts = 0
        self.removed_sources: list[str] = []
        self.resolution_changes: list[str] = []

    def is_alive(self) -> bool:
        return self.alive

    def start(self) -> None:
        self.starts += 1

    def submit(self, request) -> bool:
        return True

    def drain_resolution_changes(self) -> list[str]:
        changes = list(self.resolution_changes)
        self.resolution_changes.clear()
        return changes

    def remove_source(self, source_id: str) -> bool:
        self.removed_sources.append(source_id)
        return True

    def get_status(self) -> dict:
        return {
            "enabled": True,
            "model": self.model,
            "device": "auto",
        }

    def set_model(self, name: str) -> None:
        self.model = name

    def configure_models(self, names: set[str]) -> None:
        return None


def test_worker_respawn_emits_one_reset_per_camera_and_keeps_ten_second_backoff(
    monkeypatch,
):
    clock = [100.0]
    monkeypatch.setattr("app.streaming.manager.time.monotonic", lambda: clock[0])
    manager = StreamManager()
    worker = FakeWorker(alive=False)
    manager._worker = worker
    manager._captures = {
        "cam-a": SimpleNamespace(is_running=True),
        "cam-b": SimpleNamespace(is_running=True),
    }

    manager._maybe_respawn_worker()

    assert worker.starts == 1
    for camera_id in ("cam-a", "cam-b"):
        assert manager.drain_tracker_events(camera_id) == [
            {
                "type": "tracker_reset",
                "camera_id": camera_id,
                "reason": "worker_respawn",
            }
        ]

    clock[0] = 105.0
    manager._maybe_respawn_worker()
    assert worker.starts == 1
    assert manager.drain_tracker_events("cam-a") == []

    clock[0] = 111.0
    manager._maybe_respawn_worker()
    assert worker.starts == 2
    assert manager.drain_tracker_events("cam-a") == [
        {
            "type": "tracker_reset",
            "camera_id": "cam-a",
            "reason": "worker_respawn",
        }
    ]


def test_new_failure_after_recovery_emits_a_new_reset(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("app.streaming.manager.time.monotonic", lambda: clock[0])
    manager = StreamManager()
    worker = FakeWorker(alive=False)
    manager._worker = worker
    manager._captures = {"cam-a": SimpleNamespace(is_running=True)}

    manager._maybe_respawn_worker()
    manager.drain_tracker_events("cam-a")
    worker.alive = True
    manager._maybe_respawn_worker()
    worker.alive = False
    clock[0] = 120.0
    manager._maybe_respawn_worker()

    assert manager.drain_tracker_events("cam-a")[0]["reason"] == "worker_respawn"


def test_tracker_switch_emits_reset_and_same_tracker_is_a_noop():
    manager = StreamManager()
    worker = FakeWorker(alive=True)
    manager._worker = worker
    manager._captures = {"cam-a": SimpleNamespace(is_running=True)}

    manager.set_inference_model("sam3")
    assert manager.drain_tracker_events("cam-a") == []

    manager.set_inference_model("future-implemented-tracker")

    assert manager.drain_tracker_events("cam-a") == [
        {
            "type": "tracker_reset",
            "camera_id": "cam-a",
            "reason": "tracker_switch",
        }
    ]


def test_resolution_change_emits_camera_scoped_reset():
    manager = StreamManager()
    worker = FakeWorker(alive=True)
    worker.resolution_changes = ["cam-a"]
    manager._worker = worker
    manager._captures = {"cam-a": SimpleNamespace(is_running=True)}

    manager._submit_frame("cam-a", np.zeros((12, 16, 3), dtype=np.uint8), 1.0)

    assert manager.drain_tracker_events("cam-a") == [
        {
            "type": "tracker_reset",
            "camera_id": "cam-a",
            "reason": "resolution_changed",
        }
    ]


def test_reset_discards_queued_events_from_the_previous_id_generation():
    manager = StreamManager()
    manager._captures = {"cam-a": SimpleNamespace(is_running=True)}
    manager._tracker_events["cam-a"] = [
        {
            "type": "new_track",
            "camera_id": "cam-a",
            "track_id": 7,
        }
    ]

    manager._emit_tracker_reset("worker_respawn")

    assert manager.drain_tracker_events("cam-a") == [
        {
            "type": "tracker_reset",
            "camera_id": "cam-a",
            "reason": "worker_respawn",
        }
    ]


def test_remove_capture_releases_the_child_tracker_session():
    manager = StreamManager()
    worker = FakeWorker(alive=True)
    manager._worker = worker
    manager._captures = {
        "cam-a": SimpleNamespace(force_stop=lambda: None, is_running=True)
    }

    manager.remove_capture("cam-a")

    assert worker.removed_sources == ["cam-a"]
