from __future__ import annotations

import os

import numpy as np

from app.tracking.detector import Detection
from app.tracking.ultralytics_adapter import UltralyticsTrackerAdapter


def test_ultralytics_autoinstall_is_disabled_before_import():
    assert os.environ["YOLO_AUTOINSTALL"] == "false"

    from ultralytics.utils import AUTOINSTALL

    assert AUTOINSTALL is False


class FakeDetector:
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self.detections = [
            Detection(
                bbox=(10, 20, 30, 60),
                score=0.9,
                polygon=[(10, 20), (30, 20), (30, 60)],
            ),
            Detection(
                bbox=(100, 120, 160, 200),
                score=0.8,
                polygon=[(100, 120), (160, 120), (160, 200)],
            ),
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls.append(frame)
        return list(self.detections)


class FakeTracker:
    def __init__(self) -> None:
        self.calls: list[tuple[object, np.ndarray | None]] = []

    def update(self, detections, img=None):
        self.calls.append((detections, img))
        return np.asarray(
            [
                [101, 121, 161, 201, 42, 0.79, 0, 1],
                [11, 21, 31, 61, 7, 0.89, 0, 0],
            ],
            dtype=np.float32,
        )


def test_botsort_maps_tracker_ids_back_to_detection_polygons_and_xywh():
    detector = FakeDetector()
    tracker = FakeTracker()
    adapter = UltralyticsTrackerAdapter(
        "yolo-botsort",
        detector=detector,
        tracker_factory=lambda: tracker,
        confirm_frames=1,
    )
    session = adapter.create_session("cam-a", {})
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    tracks = session.update(frame, 1.0)

    tracker_input, tracker_img = tracker.calls[0]
    np.testing.assert_allclose(
        tracker_input.xywh,
        [[20, 40, 20, 40], [130, 160, 60, 80]],
    )
    np.testing.assert_allclose(tracker_input.conf, [0.9, 0.8])
    np.testing.assert_allclose(tracker_input.cls, [0, 0])
    assert tracker_img is frame
    assert [(track.track_id, track.bbox) for track in tracks] == [
        (42, (101, 121, 161, 201)),
        (7, (11, 21, 31, 61)),
    ]
    assert tracks[0].polygon == detector.detections[1].polygon
    assert tracks[1].polygon == detector.detections[0].polygon


def test_bytetrack_does_not_pass_the_frame_to_update():
    tracker = FakeTracker()
    adapter = UltralyticsTrackerAdapter(
        "yolo-bytetrack",
        detector=FakeDetector(),
        tracker_factory=lambda: tracker,
        confirm_frames=1,
    )

    adapter.create_session("cam-a", {}).update(
        np.zeros((10, 10, 3), dtype=np.uint8),
        1.0,
    )

    assert tracker.calls[0][1] is None


def test_each_camera_owns_a_tracker_and_reset_replaces_only_that_tracker():
    detector = FakeDetector()
    trackers: list[FakeTracker] = []

    def make_tracker() -> FakeTracker:
        tracker = FakeTracker()
        trackers.append(tracker)
        return tracker

    adapter = UltralyticsTrackerAdapter(
        "yolo-bytetrack",
        detector=detector,
        tracker_factory=make_tracker,
        confirm_frames=1,
    )
    first = adapter.create_session("cam-a", {})
    second = adapter.create_session("cam-b", {})
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    first.update(frame, 1.0)
    second.update(frame, 1.0)
    first.reset()
    first.update(frame, 2.0)

    assert len(trackers) == 3
    assert len(trackers[0].calls) == 1
    assert len(trackers[1].calls) == 1
    assert len(trackers[2].calls) == 1
    assert len(detector.calls) == 3


def test_close_releases_tracker_and_rejects_more_updates():
    import pytest

    adapter = UltralyticsTrackerAdapter(
        "yolo-bytetrack",
        detector=FakeDetector(),
        tracker_factory=FakeTracker,
        confirm_frames=1,
    )
    session = adapter.create_session("cam-a", {})

    session.close()
    session.close()

    with pytest.raises(RuntimeError, match="closed"):
        session.update(np.zeros((1, 1, 3), dtype=np.uint8), 1.0)
