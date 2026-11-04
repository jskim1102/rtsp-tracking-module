"""YOLO-seg with per-camera BoT-SORT or ByteTrack state."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from app import config as app_config
from app.inference.models_dir import PRESET_MODELS
from app.tracking.base import Track
from app.tracking.confirm import RawTrack, TrackConfirmer
from app.tracking.detector import Detection, YoloSegDetector

_KINDS = {
    "yolo-botsort": "botsort.yaml",
    "yolo-bytetrack": "bytetrack.yaml",
}
_TRACKER_ID_LOCK = threading.RLock()


def _default_tracker_factory(kind: str) -> Any:
    from ultralytics.trackers import BOTSORT, BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(_KINDS[kind])))
    tracker_type = BOTSORT if kind == "yolo-botsort" else BYTETracker
    return tracker_type(args=cfg, frame_rate=30)


class _NativeIdCounter:
    """Isolate Ultralytics' process-global BaseTrack counter per camera session."""

    def __init__(self) -> None:
        self.value = 0

    @contextmanager
    def use(self) -> Iterator[None]:
        from ultralytics.trackers.basetrack import BaseTrack

        with _TRACKER_ID_LOCK:
            previous = BaseTrack._count
            BaseTrack._count = self.value
            try:
                yield
            finally:
                self.value = BaseTrack._count
                BaseTrack._count = previous

    def reset(self) -> None:
        self.value = 0


class _TrackerDetections:
    """Minimal Boxes-compatible input consumed by Ultralytics trackers."""

    def __init__(self, detections: list[Detection]) -> None:
        self.xyxy = np.asarray([item.bbox for item in detections], dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray([item.score for item in detections], dtype=np.float32)
        self.cls = np.zeros(len(detections), dtype=np.float32)
        self.xywh = self.xyxy.copy()
        if len(self.xywh):
            self.xywh[:, 2:] -= self.xywh[:, :2]
            self.xywh[:, :2] += self.xywh[:, 2:] / 2

    @classmethod
    def _from_arrays(
        cls,
        xyxy: np.ndarray,
        xywh: np.ndarray,
        conf: np.ndarray,
        classes: np.ndarray,
    ) -> _TrackerDetections:
        value = object.__new__(cls)
        value.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        value.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        value.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        value.cls = np.asarray(classes, dtype=np.float32).reshape(-1)
        return value

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> _TrackerDetections:
        return self._from_arrays(
            self.xyxy[index],
            self.xywh[index],
            self.conf[index],
            self.cls[index],
        )


class _UltralyticsSession:
    def __init__(
        self,
        *,
        kind: str,
        detector: YoloSegDetector,
        tracker_factory: Callable[[], Any],
        confirm_frames: int,
        ttl_frames: int,
        isolate_native_ids: bool,
    ) -> None:
        self._kind = kind
        self._detector = detector
        self._tracker_factory = tracker_factory
        self._id_counter = _NativeIdCounter() if isolate_native_ids else None
        self._tracker = self._new_tracker()
        self._confirmer = TrackConfirmer(
            confirm_frames=confirm_frames,
            ttl_frames=ttl_frames,
            on_expire=lambda _track_id: None,
        )
        self._frame_index = 0
        self._closed = False

    def _new_tracker(self) -> Any:
        if self._id_counter is None:
            return self._tracker_factory()
        with self._id_counter.use():
            return self._tracker_factory()

    def _update_tracker(
        self,
        detections: _TrackerDetections,
        frame: np.ndarray,
    ) -> np.ndarray:
        def update() -> Any:
            if self._kind == "yolo-botsort":
                return self._tracker.update(detections, img=frame)
            return self._tracker.update(detections)

        if self._id_counter is None:
            output = update()
        else:
            with self._id_counter.use():
                output = update()
        array = np.asarray(output, dtype=np.float32)
        if array.size == 0:
            return np.empty((0, 8), dtype=np.float32)
        return array.reshape(1, -1) if array.ndim == 1 else array

    def update(self, frame: np.ndarray, ts: float) -> list[Track]:
        if self._closed:
            raise RuntimeError("tracker session is closed")
        detections = self._detector.detect(frame)
        tracker_output = self._update_tracker(_TrackerDetections(detections), frame)
        raw: list[RawTrack] = []
        for row in tracker_output:
            if len(row) < 8:
                continue
            detection_index = int(row[-1])
            if detection_index < 0 or detection_index >= len(detections):
                continue
            detection = detections[detection_index]
            raw.append(
                RawTrack(
                    track_id=int(row[4]),
                    bbox=tuple(int(round(float(value))) for value in row[:4]),
                    polygon=detection.polygon,
                    score=float(row[5]),
                )
            )
        self._frame_index += 1
        return self._confirmer.observe(self._frame_index, raw)

    def acknowledge_new_tracks(self, track_ids: list[int]) -> None:
        if not self._closed:
            self._confirmer.acknowledge_new_tracks(track_ids)

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("tracker session is closed")
        if self._id_counter is not None:
            self._id_counter.reset()
        self._tracker = self._new_tracker()
        self._confirmer.reset()
        self._frame_index = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._tracker = None
        self._confirmer.reset()


class UltralyticsTrackerAdapter:
    capabilities = {"concept_prompt": False}

    def __init__(
        self,
        kind: str,
        *,
        detector: YoloSegDetector | None = None,
        tracker_factory: Callable[[], Any] | None = None,
        confirm_frames: int | None = None,
        config: dict | None = None,
    ) -> None:
        if kind not in _KINDS:
            raise ValueError(f"unsupported ultralytics tracker: {kind}")
        self.name = kind
        self._kind = kind
        self._config = dict(config or {})
        self._confirm_frames = (
            app_config.TRACK_CONFIRM_FRAMES
            if confirm_frames is None
            else max(1, int(confirm_frames))
        )
        if detector is None:
            detector = YoloSegDetector(
                model_name=str(self._config.get("model") or PRESET_MODELS[0]),
                device=str(
                    self._config.get("device")
                    or os.getenv("TRACKER_DEVICE")
                    or app_config.TRACKER_DEVICE
                ),
                conf=float(
                    self._config.get("detector_conf")
                    or os.getenv("DETECTOR_CONF")
                    or 0.5
                ),
                imgsz=int(
                    self._config.get("input_size")
                    or app_config.TRACKER_INPUT_SIZE
                ),
            )
        self._detector = detector
        self.device = str(getattr(detector, "device", ""))
        self._tracker_factory = tracker_factory or (
            lambda: _default_tracker_factory(self._kind)
        )
        self._isolate_native_ids = tracker_factory is None

    def create_session(self, camera_id: str, config: dict) -> _UltralyticsSession:
        session_config = dict(self._config)
        session_config.update(config)
        return _UltralyticsSession(
            kind=self._kind,
            detector=self._detector,
            tracker_factory=self._tracker_factory,
            confirm_frames=self._confirm_frames,
            ttl_frames=int(session_config.get("track_state_ttl_frames", 300)),
            isolate_native_ids=self._isolate_native_ids,
        )
