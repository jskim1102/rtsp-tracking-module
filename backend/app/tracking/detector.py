"""Shared allowlisted YOLO segmentation detector for tracking adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from app import config as app_config
from app.inference import models_dir
from app.tracking.sam3_adapter import _mask_polygon, _to_numpy


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: tuple[int, int, int, int]
    score: float
    polygon: list[tuple[int, int]] | None


def _load_yolo_model(model_path: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_path)


def _auto_device(requested: str) -> str:
    if requested:
        return requested
    if app_config.TRACKER_DEVICE:
        return app_config.TRACKER_DEVICE
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class YoloSegDetector:
    def __init__(
        self,
        model_name: str,
        device: str = "",
        conf: float | None = None,
        imgsz: int | None = None,
    ) -> None:
        model_path = models_dir.resolve_model_path(model_name)
        self.device = _auto_device(str(device or ""))
        self.conf = float(
            conf if conf is not None else (os.getenv("DETECTOR_CONF") or 0.5)
        )
        self.imgsz = int(imgsz or app_config.TRACKER_INPUT_SIZE)
        self.model = _load_yolo_model(model_path)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame_bgr,
            classes=[0],
            conf=self.conf,
            device=self.device,
            half=self.device.startswith("cuda"),
            imgsz=self.imgsz,
            retina_masks=True,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = _to_numpy(getattr(boxes, "xyxy", [])).reshape(-1, 4)
        scores = _to_numpy(getattr(boxes, "conf", [])).reshape(-1)
        classes = _to_numpy(getattr(boxes, "cls", [])).reshape(-1)
        count = min(len(xyxy), len(scores), len(classes))
        masks = getattr(result, "masks", None)
        mask_values = _to_numpy(getattr(masks, "data", [])) if masks is not None else np.empty(0)
        if mask_values.ndim == 2:
            mask_values = mask_values[None, ...]

        detections: list[Detection] = []
        for index in range(count):
            if int(classes[index]) != 0:
                continue
            polygon = None
            if mask_values.ndim == 3 and index < len(mask_values):
                mask = mask_values[index]
                if mask.shape != frame_bgr.shape[:2]:
                    import cv2

                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (frame_bgr.shape[1], frame_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                polygon = _mask_polygon(mask)
            detections.append(
                Detection(
                    bbox=tuple(
                        int(round(float(value))) for value in xyxy[index]
                    ),
                    score=float(scores[index]),
                    polygon=polygon,
                )
            )
        return detections
