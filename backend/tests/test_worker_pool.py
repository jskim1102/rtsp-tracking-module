"""Regression lock: stateless multi-lane batching is gone."""

import app.inference.worker as worker_module
from app.tracking.runner import TrackerRunner


def test_legacy_model_lane_type_is_not_exposed():
    assert not hasattr(worker_module, "_ModelLane")


def test_runner_status_declares_single_item_batches_only():
    runner = TrackerRunner()

    status = runner.get_pool_status()

    assert status["models"] == ["sam3"]
    assert status["batch_max"] == 1
    assert status["batch_timeout_ms"] == 0.0


def test_per_source_model_configuration_is_a_noop():
    runner = TrackerRunner()

    runner.configure_models({"sam3", "another-camera-model"})

    assert runner.get_pool_status()["models"] == ["sam3"]
