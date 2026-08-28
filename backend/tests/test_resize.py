"""fix13: _resize_for_inference — 추론 제출 전 프레임 다운스케일(IPC pickle 비용 절감).

종횡비 보존 · long side == tracker imgsz(560) · 업스케일 금지 · dtype/채널 보존.
"""

import numpy as np
import pytest

from app.streaming.manager import _resize_for_inference


def _frame(w: int, h: int) -> np.ndarray:
    """(h, w, 3) uint8 zeros 프레임 — OpenCV/numpy 는 (rows=h, cols=w, ch)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "w,h,exp_w,exp_h",
    [
        (1920, 1080, 560, 315),
        (1280, 720, 560, 315),
        (704, 480, 560, 382),
        (1000, 1000, 560, 560),
    ],
)
def test_downscale_long_side_tracker_size_aspect_preserved(w, h, exp_w, exp_h):
    out = _resize_for_inference(_frame(w, h))
    assert out.shape[1] == exp_w
    assert out.shape[0] == exp_h


@pytest.mark.parametrize(
    "w,h,exp_w,exp_h",
    [(640, 480, 560, 420), (320, 240, 320, 240)],
)
def test_no_upscale_when_within_imgsz(w, h, exp_w, exp_h):
    """tracker size보다 큰 frame만 축소하고 작은 frame은 업스케일하지 않는다."""
    src = _frame(w, h)
    out = _resize_for_inference(src)
    assert out.shape == (exp_h, exp_w, 3)


def test_dtype_and_channels_preserved():
    out = _resize_for_inference(_frame(1920, 1080))
    assert out.dtype == np.uint8
    assert out.shape[2] == 3
