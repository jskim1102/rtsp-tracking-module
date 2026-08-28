"""Legacy worker import path now resolves to the stateful tracker runner."""

from app.inference import InferenceWorker
from app.tracking.runner import TrackerRunner


def test_inference_worker_compatibility_alias_is_tracker_runner():
    assert InferenceWorker is TrackerRunner


def test_tracker_runner_preserves_manager_lifecycle_surface():
    required = {"start", "stop", "submit", "drain_results", "is_alive"}

    assert required.issubset(vars(TrackerRunner))
