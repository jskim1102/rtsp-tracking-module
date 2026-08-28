from __future__ import annotations

import threading
import time

import numpy as np

from app.inference import FrameRequest, InferenceResult
from app.streaming.manager import StreamManager
from app.tracking.base import Track
from app.tracking.gallery import TrackGallery
from app.tracking.runner import TrackerRunner, _SessionRouter
from app.tracking.sam3_adapter import BackendTrack, Sam3Adapter


class ScriptedBackend:
    def __init__(self, frames: dict[str, list[list[BackendTrack]]]) -> None:
        self.frames = frames

    def create_session(self, camera_id: str, *, prompt: str, config: dict):
        return camera_id

    def update(self, handle, frame: np.ndarray, ts: float) -> list[BackendTrack]:
        return self.frames[handle].pop(0)

    def close(self, handle) -> None:
        return None


def _raw(track_id: int = 7) -> BackendTrack:
    return BackendTrack(track_id, (1, 1, 5, 5), None, 0.94)


def _request(camera_id: str, timestamp: float) -> FrameRequest:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[1:5, 1:5] = (0, 180, 255)
    return FrameRequest(camera_id, frame, timestamp, imgsz=8)


def test_gallery_ignores_preconfirmation_flicker_then_saves_crop_and_event(tmp_path):
    backend = ScriptedBackend(
        {"cam-a": [[_raw()], [_raw()], [], [_raw()], [_raw()], [_raw()]]}
    )
    router = _SessionRouter(Sam3Adapter(backend=backend, confirm_frames=3), {})
    gallery = TrackGallery(tmp_path)

    for timestamp in range(1, 6):
        assert gallery.observe(router.process(_request("cam-a", timestamp))) == []
        assert gallery.snapshot()["total"] == 0

    result = router.process(_request("cam-a", 6.0))
    events = gallery.observe(result)
    snapshot = gallery.snapshot()

    assert events == [
        {
            "type": "new_track",
            "track_id": 7,
            "camera_id": "cam-a",
            "first_seen": 6.0,
            "thumb_url": "/api/tracks/7/thumb?camera_id=cam-a",
        }
    ]
    assert snapshot["active"] == snapshot["total"] == 1
    thumb_path = gallery.thumbnail(7, "cam-a")
    assert thumb_path is not None and thumb_path.read_bytes().startswith(b"\xff\xd8")


def test_camera_namespaces_keep_equal_track_ids_independent(tmp_path):
    gallery = TrackGallery(tmp_path)
    crop = np.full((3, 3, 3), 255, dtype=np.uint8)

    for camera_id in ("cam-a", "cam-b"):
        gallery.observe(
            InferenceResult(
                source_id=camera_id,
                timestamp=1.0,
                tracks=[Track(7, (0, 0, 3, 3), None, 0.9, True)],
                new_track_crops={7: crop},
            )
        )

    assert gallery.snapshot()["total"] == 2
    assert gallery.thumbnail(7) is None
    assert gallery.thumbnail(7, "cam-a") != gallery.thumbnail(7, "cam-b")


def test_gallery_marks_tracks_inactive_and_reset_removes_registered_crops(tmp_path):
    gallery = TrackGallery(tmp_path)
    gallery.observe(
        InferenceResult(
            source_id="cam-a",
            timestamp=1.0,
            tracks=[Track(3, (0, 0, 2, 2), None, 0.9, True)],
            new_track_crops={3: np.ones((2, 2, 3), dtype=np.uint8)},
        )
    )
    thumb_path = gallery.thumbnail(3, "cam-a")

    gallery.observe(InferenceResult(source_id="cam-a", timestamp=2.0))
    assert gallery.snapshot()["active"] == 0
    gallery.reset()

    assert gallery.snapshot() == {"active": 0, "total": 0, "items": []}
    assert thumb_path is not None and not thumb_path.exists()


def test_gallery_cap_evicts_the_oldest_item_and_crop(tmp_path):
    gallery = TrackGallery(tmp_path, max_items=2)
    old_path = None
    for track_id in (1, 2, 3):
        gallery.observe(
            InferenceResult(
                source_id="cam-a",
                timestamp=float(track_id),
                tracks=[Track(track_id, (0, 0, 2, 2), None, 0.9, True)],
                new_track_crops={track_id: np.ones((2, 2, 3), dtype=np.uint8)},
            )
        )
        if track_id == 1:
            old_path = gallery.thumbnail(1, "cam-a")

    snapshot = gallery.snapshot()
    assert snapshot["total"] == 2
    assert [item["track_id"] for item in snapshot["items"]] == [3, 2]
    assert old_path is not None and not old_path.exists()


def test_gallery_cap_only_acknowledges_and_announces_items_retained_from_same_batch(
    tmp_path,
):
    gallery = TrackGallery(tmp_path, max_items=2)
    tracks = [
        Track(track_id, (0, 0, 2, 2), None, 0.9, True)
        for track_id in (1, 2, 3)
    ]

    observation = gallery.observe_with_ack(
        InferenceResult(
            source_id="cam-a",
            timestamp=1.0,
            tracks=tracks,
            new_track_crops={
                track.track_id: np.ones((2, 2, 3), dtype=np.uint8)
                for track in tracks
            },
        )
    )

    retained = [item["track_id"] for item in gallery.snapshot()["items"]]
    assert retained == [3, 2]
    assert observation.acknowledged_track_ids == [2, 3]
    assert observation.created_track_ids == [2, 3]
    assert [event["track_id"] for event in observation.events] == [2, 3]


def test_persistent_gallery_startup_removes_only_module_owned_jpegs(tmp_path):
    owned = tmp_path / "0123456789abcdef-7.jpg"
    owned_negative = tmp_path / "0123456789abcdef--7.jpg"
    unrelated = tmp_path / "customer-photo.jpg"
    lookalike = tmp_path / "0123456789abcdef-7.jpeg"
    for path in (owned, owned_negative, unrelated, lookalike):
        path.write_bytes(b"stale")

    TrackGallery(tmp_path)

    assert not owned.exists()
    assert not owned_negative.exists()
    assert unrelated.read_bytes() == b"stale"
    assert lookalike.read_bytes() == b"stale"


def test_gallery_write_failure_removes_partial_jpeg(tmp_path, monkeypatch):
    gallery = TrackGallery(tmp_path)
    original_write_bytes = type(tmp_path).write_bytes

    def fail_after_partial_write(path, data):
        original_write_bytes(path, data[:2])
        raise OSError("disk full")

    monkeypatch.setattr(type(tmp_path), "write_bytes", fail_after_partial_write)
    observation = gallery.observe_with_ack(
        InferenceResult(
            source_id="cam-a",
            timestamp=1.0,
            tracks=[Track(7, (0, 0, 2, 2), None, 0.9, True)],
            new_track_crops={7: np.ones((2, 2, 3), dtype=np.uint8)},
        )
    )

    assert observation.events == []
    assert observation.acknowledged_track_ids == []
    assert list(tmp_path.iterdir()) == []


def test_clear_cameras_serializes_unlink_before_same_track_can_be_recreated(
    tmp_path,
    monkeypatch,
):
    gallery = TrackGallery(tmp_path)
    initial = InferenceResult(
        source_id="cam-a",
        timestamp=1.0,
        tracks=[Track(3, (0, 0, 2, 2), None, 0.9, True)],
        new_track_crops={3: np.ones((2, 2, 3), dtype=np.uint8)},
    )
    gallery.observe(initial)
    old_path = gallery.thumbnail(3, "cam-a")
    assert old_path is not None

    unlink_entered = threading.Event()
    allow_unlink = threading.Event()
    original_unlink = type(old_path).unlink

    def blocking_unlink(path, *args, **kwargs):
        if path == old_path:
            unlink_entered.set()
            assert allow_unlink.wait(timeout=2.0)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(old_path), "unlink", blocking_unlink)
    clear_thread = threading.Thread(target=gallery.clear_cameras, args=(["cam-a"],))
    clear_thread.start()
    assert unlink_entered.wait(timeout=1.0)

    recreated = InferenceResult(
        source_id="cam-a",
        timestamp=2.0,
        tracks=[Track(3, (0, 0, 2, 2), None, 0.9, True)],
        new_track_crops={3: np.full((2, 2, 3), 255, dtype=np.uint8)},
    )
    observe_thread = threading.Thread(target=gallery.observe, args=(recreated,))
    observe_thread.start()
    time.sleep(0.05)
    assert observe_thread.is_alive()

    allow_unlink.set()
    clear_thread.join(timeout=1.0)
    observe_thread.join(timeout=1.0)
    recreated_path = gallery.thumbnail(3, "cam-a")
    assert recreated_path is not None and recreated_path.exists()


def test_manager_queues_new_track_event_after_gallery_persists_the_crop(tmp_path):
    manager = StreamManager()
    manager._gallery = TrackGallery(tmp_path)
    manager._captures = {"cam-a": object()}
    result = InferenceResult(
        source_id="cam-a",
        timestamp=4.5,
        request_id=1,
        tracks=[Track(9, (0, 0, 2, 2), None, 0.8, True)],
        new_track_crops={9: np.ones((2, 2, 3), dtype=np.uint8)},
    )

    manager._record_results([result])

    assert manager.drain_tracker_events("cam-a") == [
        {
            "type": "new_track",
            "track_id": 9,
            "camera_id": "cam-a",
            "first_seen": 4.5,
            "thumb_url": "/api/tracks/9/thumb?camera_id=cam-a",
        }
    ]
    assert manager._gallery.thumbnail(9, "cam-a") is not None


def test_manager_acknowledges_new_track_only_after_gallery_persists_it(tmp_path):
    class AckWorker:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[int], int]] = []

        def acknowledge_tracks(
            self,
            camera_id: str,
            track_ids: list[int],
            *,
            generation: int,
        ) -> bool:
            self.calls.append((camera_id, track_ids, generation))
            return True

    manager = StreamManager()
    manager._gallery = TrackGallery(tmp_path)
    manager._captures = {"cam-a": object()}
    manager._worker = AckWorker()
    result = InferenceResult(
        source_id="cam-a",
        timestamp=4.5,
        request_id=1,
        tracker_generation=3,
        tracks=[Track(9, (0, 0, 2, 2), None, 0.8, True)],
        new_track_crops={9: np.ones((2, 2, 3), dtype=np.uint8)},
    )

    manager._record_results([result])

    assert manager._gallery.thumbnail(9, "cam-a") is not None
    assert manager._worker.calls == [("cam-a", [9], 3)]


def test_manager_rejects_a_result_from_the_generation_before_source_reset(tmp_path):
    runner = TrackerRunner(adapter_factory=lambda _: None)
    assert runner.submit(_request("cam-a", 1.0)) is True
    stale = InferenceResult(
        source_id="cam-a",
        timestamp=1.0,
        request_id=1,
        tracker_generation=0,
        tracks=[Track(9, (0, 0, 2, 2), None, 0.8, True)],
        new_track_crops={9: np.ones((2, 2, 3), dtype=np.uint8)},
    )
    assert runner.remove_source("cam-a") is True

    manager = StreamManager()
    manager._worker = runner
    manager._gallery = TrackGallery(tmp_path)
    manager._captures = {"cam-a": object()}
    manager._record_results([stale])

    assert manager.get_source_latest_tracks("cam-a") is None
    assert manager._gallery.snapshot()["total"] == 0
    assert manager.drain_tracker_events("cam-a") == []


def test_manager_does_not_hold_cache_or_membership_locks_during_crop_io(tmp_path):
    observe_entered = threading.Event()
    allow_observe = threading.Event()

    class BlockingGallery(TrackGallery):
        def observe_with_ack(self, result):
            observe_entered.set()
            assert allow_observe.wait(timeout=2.0)
            return super().observe_with_ack(result)

    manager = StreamManager()
    manager._gallery = BlockingGallery(tmp_path)
    manager._captures = {"cam-a": object()}
    result = InferenceResult(
        source_id="cam-a",
        timestamp=4.5,
        request_id=1,
        tracks=[Track(9, (0, 0, 2, 2), None, 0.8, True)],
        new_track_crops={9: np.ones((2, 2, 3), dtype=np.uint8)},
    )
    record_thread = threading.Thread(target=manager._record_results, args=([result],))
    record_thread.start()
    assert observe_entered.wait(timeout=1.0)

    assert manager._results_lock.acquire(blocking=False)
    manager._results_lock.release()
    assert manager._lock.acquire(blocking=False)
    manager._lock.release()

    allow_observe.set()
    record_thread.join(timeout=1.0)
    assert not record_thread.is_alive()


def test_disabling_camera_tracking_marks_its_gallery_items_inactive(tmp_path):
    manager = StreamManager()
    manager._gallery = TrackGallery(tmp_path)
    manager._gallery.observe(
        InferenceResult(
            source_id="cam-a",
            timestamp=1.0,
            tracks=[Track(2, (0, 0, 2, 2), None, 0.8, True)],
            new_track_crops={2: np.ones((2, 2, 3), dtype=np.uint8)},
        )
    )

    manager.set_source_inference_enabled("cam-a", False)

    assert manager._gallery.snapshot()["active"] == 0
