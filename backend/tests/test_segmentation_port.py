"""YOLO instance-segmentation 포트의 데이터 계약 회귀 테스트."""

from __future__ import annotations

import json
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

from app.inference import models_dir
from app.inference.worker import (
    Detection,
    FrameRequest,
    InferenceResult,
    _merge_partial_results,
    _parse_results,
    _predict_segmentation,
    _rescale_result,
)
from app.streaming.manager import StreamManager, tracks_to_json
from app.tracking.base import Track


class _Tensor:
    def __init__(self, value) -> None:
        self._value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _Boxes:
    def __init__(self, count: int = 1) -> None:
        self.xyxy = _Tensor([[10.2, 20.4, 110.6, 220.8]] * count)
        self.conf = _Tensor([0.91] * count)
        self.cls = _Tensor([0] * count)
        self._count = count

    def __len__(self) -> int:
        return self._count


def _result_with_polygon(polygon: np.ndarray):
    return SimpleNamespace(
        boxes=_Boxes(),
        masks=SimpleNamespace(xy=[np.asarray(polygon, dtype=np.float32)]),
    )


def _mask() -> list[tuple[int, int]]:
    return [(10, 20), (31, 40), (50, 21)]


def test_parse_results_returns_empty_when_masks_are_missing():
    assert _parse_results(
        SimpleNamespace(boxes=_Boxes(), masks=None),
        {0: "person"},
        "yolo26n-seg.pt",
    ) == []
    assert _parse_results(
        SimpleNamespace(boxes=None, masks=None),
        {0: "person"},
        "yolo26n-seg.pt",
    ) == []


def test_parse_results_keeps_all_request_frame_polygon_points_as_ints():
    polygon = np.array(
        [[10.2, 20.6], [80.7, 20.1], [80.4, 90.8], [10.1, 90.2]],
        dtype=np.float32,
    )
    result = _result_with_polygon(polygon)

    detections = _parse_results(result, {0: "person"}, "yolo26n-seg.pt")

    assert len(detections) == 1
    assert detections[0].xyxy == (10, 20, 110, 220)
    assert detections[0].mask == [(10, 21), (81, 20), (80, 91), (10, 90)]
    assert all(isinstance(value, int) for point in detections[0].mask for value in point)


def test_parse_results_preserves_every_dense_polygon_point_as_ints():
    angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    circle = np.column_stack((64 + 40 * np.cos(angles), 64 + 40 * np.sin(angles)))

    mask = _parse_results(
        _result_with_polygon(circle), {0: "person"}, "yolo26n-seg.pt"
    )[0].mask

    assert len(mask) == len(circle)
    assert mask == [
        (int(round(float(x))), int(round(float(y))))
        for x, y in circle
    ]
    assert all(isinstance(value, int) for point in mask for value in point)


def test_prediction_requests_retina_masks_at_original_frame_resolution():
    calls = []

    def model(frames, **kwargs):
        calls.append((frames, kwargs))
        return ["result"]

    frames = [np.zeros((1080, 1920, 3), dtype=np.uint8)]
    result = _predict_segmentation(
        model,
        frames,
        conf=0.5,
        imgsz=640,
        device="cpu",
    )

    assert result == ["result"]
    assert calls[0][0] is frames
    assert calls[0][1] == {
        "conf": 0.5,
        "imgsz": 640,
        "device": "cpu",
        "retina_masks": True,
        "verbose": False,
    }


def test_oom_rescale_restores_box_and_mask_to_request_coordinates():
    smaller = InferenceResult(
        "cam",
        1.0,
        detections=[
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                xyxy=(10, 20, 100, 150),
                model="yolo26n-seg.pt",
                mask=[(10, 20), (31, 40)],
            )
        ],
        tracks=[
            Track(
                track_id=5,
                bbox=(10, 20, 100, 150),
                polygon=[(10, 20), (31, 40)],
                score=0.9,
                is_new=False,
            )
        ],
        frame_w=320,
        frame_h=180,
        effective_imgsz=320,
        oom_recovered=True,
    )

    recovered = _rescale_result(smaller, frame_w=640, frame_h=360)

    assert recovered.detections[0].xyxy == (20, 40, 200, 300)
    assert recovered.detections[0].mask == [(20, 40), (62, 80)]
    assert recovered.tracks[0].bbox == (20, 40, 200, 300)
    assert recovered.tracks[0].polygon == [(20, 40), (62, 80)]
    assert recovered.frame_w == 640
    assert recovered.frame_h == 360


def test_multimodel_merge_preserves_each_detection_mask():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    request = FrameRequest("cam", frame, 123.0, request_id=9, imgsz=640)
    partials = [
        InferenceResult(
            "cam",
            123.0,
            detections=[
                Detection(0, "person", 0.9, (1, 2, 30, 40), model=model, mask=_mask())
            ],
            frame_w=640,
            frame_h=360,
            request_id=9,
            model=model,
            tracks=[
                Track(
                    track_id=index + 1,
                    bbox=(1, 2, 30, 40),
                    polygon=_mask(),
                    score=0.9,
                    is_new=index == 0,
                )
            ],
        )
        for index, model in enumerate(("yolo26n-seg.pt", "yolo26x-seg.pt"))
    ]

    merged = _merge_partial_results(request, partials)

    assert [item.model for item in merged.detections] == [
        "yolo26n-seg.pt",
        "yolo26x-seg.pt",
    ]
    assert all(item.mask == _mask() for item in merged.detections)
    assert [track.track_id for track in merged.tracks] == [1, 2]
    assert all(track.polygon == _mask() for track in merged.tracks)


def test_ws_payload_contains_track_polygon_box_and_frame_seam():
    result = InferenceResult(
        source_id="cam",
        timestamp=1.0,
        frame_w=640,
        frame_h=360,
        tracks=[
            Track(
                track_id=7,
                bbox=(1, 2, 30, 40),
                polygon=_mask(),
                score=0.9,
                is_new=False,
            )
        ],
    )

    payload = json.loads(tracks_to_json(result))

    assert payload["frame"] == {"w": 640, "h": 360}
    assert payload["items"][0]["bbox"] == [1, 2, 30, 40]
    assert payload["items"][0]["polygon"] == [
        [10, 20],
        [31, 40],
        [50, 21],
    ]
    assert payload["items"][0]["track_id"] == 7


def test_frame_request_has_no_dead_threshold_or_class_override_fields():
    names = {field.name for field in fields(FrameRequest)}

    assert "conf_threshold" not in names
    assert "classes" not in names


def test_manager_submits_tracker_request_without_dead_filter_overrides():
    submitted: list[FrameRequest] = []

    class _Worker:
        @staticmethod
        def get_status():
            return {"enabled": True}

        @staticmethod
        def submit(request: FrameRequest):
            submitted.append(request)
            return True

    manager = StreamManager()
    manager._worker = _Worker()
    manager._submit_frame("cam", np.zeros((8, 8, 3), dtype=np.uint8), captured_at=1.0)

    assert submitted[0].source_id == "cam"
    assert not hasattr(submitted[0], "classes")
    assert not hasattr(submitted[0], "conf_threshold")


def test_seg_presets_expose_static_coco_classes_without_weight_loading():
    classes = models_dir.list_model_classes("yolo26n-seg.pt")

    assert len(classes) == 80
    assert classes[0] == {"id": 0, "name": "person"}
    assert classes[-1] == {"id": 79, "name": "toothbrush"}
