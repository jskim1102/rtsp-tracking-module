"""Inference transport contract with the stateful tracker runner facade."""

from app.inference.worker import (
    Detection,
    FrameRequest,
    InferenceResult,
)
from app.inference import models_dir


def __getattr__(name: str):
    # Lazy compatibility alias: eager import creates a cycle when multiprocessing
    # spawn imports app.tracking.runner in a fresh interpreter.
    if name == "InferenceWorker":
        from app.tracking.runner import TrackerRunner

        return TrackerRunner
    raise AttributeError(name)

__all__ = [
    "Detection",
    "FrameRequest",
    "InferenceResult",
    "InferenceWorker",
    "models_dir",
]
