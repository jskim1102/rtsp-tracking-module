"""Inference transport DTOs retained across the tracker migration.

The former multi-model YOLO subprocess pool was replaced by
``app.tracking.runner.TrackerRunner``. Detection helpers remain temporarily for the
phase4 WebSocket contract migration; no production path performs stateless batching.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from app import config
from app.tracking.base import Track


@dataclass
class Detection:
    """Legacy phase2 instance-segmentation payload (removed from WS in phase4)."""

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]
    model: str = ""
    mask: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class FrameRequest:
    source_id: str
    frame: np.ndarray
    timestamp: float
    model_names: Optional[list[str]] = None
    request_id: int = 0
    imgsz: int = config.TRACKER_INPUT_SIZE


@dataclass
class InferenceResult:
    source_id: str
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    frame_w: int = 0
    frame_h: int = 0
    infer_ms: float = 0.0
    idle_ms: float = 0.0
    request_id: int = 0
    model: str = ""
    batch_size: int = 1
    effective_imgsz: int = 0
    oom_recovered: bool = False
    device: str = ""
    tracker_generation: int = 0
    tracks: list[Track] = field(default_factory=list)
    new_track_crops: dict[int, np.ndarray] = field(default_factory=dict)


def _merge_partial_results(
    request: FrameRequest,
    partials: list[InferenceResult],
) -> InferenceResult:
    """Legacy DTO helper kept until the phase4 detections endpoint is removed."""
    frame_h, frame_w = request.frame.shape[:2]
    effective_sizes = [item.effective_imgsz for item in partials if item.effective_imgsz > 0]
    devices = {item.device for item in partials if item.device}
    return InferenceResult(
        source_id=request.source_id,
        timestamp=request.timestamp,
        detections=[item for partial in partials for item in partial.detections],
        tracks=[item for partial in partials for item in partial.tracks],
        new_track_crops={
            track_id: crop
            for partial in partials
            for track_id, crop in partial.new_track_crops.items()
        },
        frame_w=int(frame_w),
        frame_h=int(frame_h),
        infer_ms=sum(item.infer_ms for item in partials),
        idle_ms=sum(item.idle_ms for item in partials),
        request_id=request.request_id,
        batch_size=max((item.batch_size for item in partials), default=1),
        effective_imgsz=min(effective_sizes, default=int(request.imgsz)),
        oom_recovered=any(item.oom_recovered for item in partials),
        device=next(iter(devices)) if len(devices) == 1 else ",".join(sorted(devices)),
        tracker_generation=max(
            (item.tracker_generation for item in partials),
            default=0,
        ),
    )


def _rescale_result(
    result: InferenceResult,
    *,
    frame_w: int,
    frame_h: int,
) -> InferenceResult:
    if result.frame_w <= 0 or result.frame_h <= 0:
        return replace(result, frame_w=frame_w, frame_h=frame_h)
    sx = frame_w / result.frame_w
    sy = frame_h / result.frame_h
    detections = [
        replace(
            detection,
            xyxy=(
                round(detection.xyxy[0] * sx),
                round(detection.xyxy[1] * sy),
                round(detection.xyxy[2] * sx),
                round(detection.xyxy[3] * sy),
            ),
            mask=[
                (int(round(x * sx)), int(round(y * sy)))
                for x, y in detection.mask
            ],
        )
        for detection in result.detections
    ]
    tracks = [
        replace(
            track,
            bbox=(
                round(track.bbox[0] * sx),
                round(track.bbox[1] * sy),
                round(track.bbox[2] * sx),
                round(track.bbox[3] * sy),
            ),
            polygon=(
                [
                    (int(round(x * sx)), int(round(y * sy)))
                    for x, y in track.polygon
                ]
                if track.polygon is not None
                else None
            ),
        )
        for track in result.tracks
    ]
    return replace(
        result,
        detections=detections,
        tracks=tracks,
        frame_w=frame_w,
        frame_h=frame_h,
    )


def _predict_segmentation(model: Any, frames: Any, *, conf: float, imgsz: int, device: str):
    return model(
        frames,
        conf=conf,
        imgsz=int(imgsz),
        device=device,
        retina_masks=True,
        verbose=False,
    )


def _parse_results(result: Any, names: Any, model_name: str = "") -> list[Detection]:
    detections: list[Detection] = []
    if (
        result.boxes is None
        or len(result.boxes) == 0
        or getattr(result, "masks", None) is None
    ):
        return detections
    boxes = result.boxes
    xyxy_arr = boxes.xyxy.cpu().numpy().astype(int)
    conf_arr = boxes.conf.cpu().numpy()
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    polygons = getattr(result.masks, "xy", None)
    if polygons is None:
        return detections
    for index in range(min(len(boxes), len(polygons))):
        x1, y1, x2, y2 = xyxy_arr[index].tolist()
        class_id = int(cls_arr[index])
        if hasattr(names, "get"):
            class_name = str(names.get(class_id, str(class_id)))
        else:
            class_name = str(names[class_id]) if class_id < len(names) else str(class_id)
        polygon = np.asarray(polygons[index], dtype=np.float32).reshape(-1, 2)
        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(conf_arr[index]),
                xyxy=(x1, y1, x2, y2),
                model=model_name,
                mask=[
                    (int(round(float(x))), int(round(float(y))))
                    for x, y in polygon
                ],
            )
        )
    return detections


def __getattr__(name: str):
    """Backward-compatible direct import without creating a module import cycle."""
    if name == "InferenceWorker":
        from app.tracking.runner import TrackerRunner

        return TrackerRunner
    raise AttributeError(name)
