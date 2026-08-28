"""Stateful tracker sessions use one fixed input size; adaptive stages are removed."""

import numpy as np

import app.streaming.manager as manager_module
from app.config import TRACKER_INPUT_SIZE
from app.inference import FrameRequest
from app.streaming.manager import _resize_for_inference


def test_adaptive_resolution_controller_is_not_part_of_tracker_manager():
    assert not hasattr(manager_module, "_adaptive_imgsz_step")


def test_request_and_resize_use_the_configured_fixed_tracker_size():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    request = FrameRequest("cam", frame, 1.0)
    resized = _resize_for_inference(frame)

    assert request.imgsz == TRACKER_INPUT_SIZE == 560
    assert resized.shape == (315, 560, 3)
