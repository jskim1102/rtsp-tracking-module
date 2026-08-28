from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from app.tracking.sam3_adapter import (
    BackendTrack,
    HFSam3Backend,
    Sam3Adapter,
    prune_hf_session_memory,
)


class FakeBackend:
    def __init__(self) -> None:
        self.created: list[tuple[str, str, dict]] = []
        self.closed: list[dict] = []
        self.frames: dict[str, list[list[BackendTrack]]] = {}

    def create_session(self, camera_id: str, *, prompt: str, config: dict):
        handle = {"camera_id": camera_id, "generation": len(self.created) + 1}
        self.created.append((camera_id, prompt, dict(config)))
        return handle

    def update(self, handle, frame: np.ndarray, ts: float) -> list[BackendTrack]:
        scripted = self.frames.setdefault(handle["camera_id"], [])
        return scripted.pop(0) if scripted else []

    def close(self, handle) -> None:
        self.closed.append(handle)


def _raw(track_id: int = 7) -> BackendTrack:
    return BackendTrack(
        track_id=track_id,
        bbox=(10, 20, 110, 220),
        polygon=[(10, 20), (110, 20), (110, 220)],
        score=0.94,
    )


def test_session_repeats_new_until_gallery_acknowledges_then_announces_once():
    backend = FakeBackend()
    backend.frames["cam-a"] = [[_raw()], [_raw()], [_raw()], [_raw()]]
    adapter = Sam3Adapter(
        backend=backend,
        confirm_frames=3,
        concept_prompt="person",
    )
    session = adapter.create_session("cam-a", {"memory_bank_size": 5})
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    assert session.update(frame, 1.0) == []
    assert session.update(frame, 2.0) == []
    confirmed = session.update(frame, 3.0)
    retried = session.update(frame, 4.0)
    session.acknowledge_new_tracks([7])
    backend.frames["cam-a"].append([_raw()])
    retained = session.update(frame, 5.0)

    assert confirmed[0].track_id == retried[0].track_id == retained[0].track_id == 7
    assert confirmed[0].is_new is True
    assert retried[0].is_new is True
    assert retained[0].is_new is False
    assert confirmed[0].polygon == [(10, 20), (110, 20), (110, 220)]
    assert len(backend.created) == 1
    assert backend.created[0][0:2] == ("cam-a", "person")
    assert backend.created[0][2]["memory_bank_size"] == 5


def test_unconfirmed_track_must_be_consecutive():
    backend = FakeBackend()
    backend.frames["cam-a"] = [[_raw()], [], [_raw()], [_raw()], [_raw()]]
    session = Sam3Adapter(backend=backend, confirm_frames=3).create_session("cam-a", {})
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    assert [session.update(frame, float(index)) for index in range(1, 5)] == [
        [],
        [],
        [],
        [],
    ]
    assert session.update(frame, 5.0)[0].is_new is True


def test_reset_recreates_backend_session_and_close_is_idempotent():
    backend = FakeBackend()
    adapter = Sam3Adapter(backend=backend, confirm_frames=1, concept_prompt="forklift")
    session = adapter.create_session("cam-a", {"input_size": 560})

    session.reset()
    session.close()
    session.close()

    assert [item[:2] for item in backend.created] == [
        ("cam-a", "forklift"),
        ("cam-a", "forklift"),
    ]
    assert len(backend.closed) == 2
    with pytest.raises(RuntimeError, match="closed"):
        session.update(np.zeros((1, 1, 3), dtype=np.uint8), 1.0)


def test_adapter_reads_prompt_from_environment_and_confirm_frames_from_config(monkeypatch):
    monkeypatch.setenv("TRACKER_CONCEPT_PROMPT", "yellow school bus")
    monkeypatch.setenv("TRACK_CONFIRM_FRAMES", "2")
    monkeypatch.setattr("app.tracking.sam3_adapter.app_config.TRACK_CONFIRM_FRAMES", 2)
    backend = FakeBackend()
    backend.frames["cam"] = [[_raw()], [_raw()]]
    session = Sam3Adapter(backend=backend).create_session("cam", {})
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    assert session.update(frame, 1.0) == []
    assert session.update(frame, 2.0)[0].is_new is True
    assert backend.created[0][1] == "yellow school bus"


def test_environment_confirm_frames_uses_the_central_30_frame_clamp(monkeypatch):
    monkeypatch.setenv("TRACK_CONFIRM_FRAMES", "99")
    monkeypatch.setattr("app.tracking.sam3_adapter.app_config.TRACK_CONFIRM_FRAMES", 30)
    backend = FakeBackend()
    backend.frames["cam"] = [[_raw()] for _ in range(30)]
    session = Sam3Adapter(backend=backend).create_session("cam", {})
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    for frame_index in range(29):
        assert session.update(frame, float(frame_index)) == []
    assert session.update(frame, 29.0)[0].is_new is True


def test_explicit_zero_confirm_frames_is_not_replaced_by_the_environment(monkeypatch):
    monkeypatch.setenv("TRACK_CONFIRM_FRAMES", "3")
    backend = FakeBackend()
    backend.frames["cam"] = [[_raw()]]
    session = Sam3Adapter(backend=backend, confirm_frames=0).create_session("cam", {})

    assert session.update(np.zeros((2, 2, 3), dtype=np.uint8), 1.0)[0].is_new is True


def test_absent_confirmed_track_state_expires_and_can_be_announced_again():
    backend = FakeBackend()
    backend.frames["cam"] = [[_raw()], [], [], [], [_raw()]]
    session = Sam3Adapter(backend=backend, confirm_frames=1).create_session(
        "cam",
        {"track_state_ttl_frames": 2},
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    assert session.update(frame, 1.0)[0].is_new is True
    assert session.update(frame, 2.0) == []
    assert session.update(frame, 3.0) == []
    assert session.update(frame, 4.0) == []
    assert session.update(frame, 5.0)[0].is_new is True


def test_missing_hf_token_fails_clearly_without_fallback(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN is required"):
        HFSam3Backend(
            {
                "checkpoint": "facebook/sam3.1",
                "input_size": 560,
                "memory_bank_size": 7,
                "recondition_every_nth_frame": 16,
                "max_vision_features_cache_size": 1,
            }
        )


def test_hf_session_frame_stores_are_pruned_to_the_memory_bank_size():
    session = SimpleNamespace(
        processed_frames={index: object() for index in range(10)},
        output_dict_per_obj={
            0: {
                "cond_frame_outputs": {index: object() for index in range(10)},
                "non_cond_frame_outputs": {index: object() for index in range(10)},
            }
        },
        frames_tracked_per_obj={
            0: {index: object() for index in range(10)},
        },
        obj_id_to_tracker_score_frame_wise={
            7: {index: 0.9 for index in range(10)},
        },
    )

    prune_hf_session_memory(session, 3)

    assert list(session.processed_frames) == [7, 8, 9]
    assert list(session.output_dict_per_obj[0]["cond_frame_outputs"]) == [7, 8, 9]
    assert list(session.output_dict_per_obj[0]["non_cond_frame_outputs"]) == [7, 8, 9]
    assert list(session.frames_tracked_per_obj[0]) == [7, 8, 9]
    assert list(session.obj_id_to_tracker_score_frame_wise[7]) == [7, 8, 9]


def test_hf_backend_uses_monotonic_frame_indices_after_memory_pruning():
    class FakeInferenceSession:
        def __init__(self) -> None:
            self.processed_frames: dict[int, object] = {}
            self.output_dict_per_obj = {}
            self.frames_tracked_per_obj = {}
            self.obj_id_to_tracker_score_frame_wise = {}

        def add_new_frame(self, frame, frame_idx=None):
            if frame_idx is None:
                frame_idx = len(self.processed_frames)
            self.processed_frames[frame_idx] = frame
            return frame_idx

    class FakeInputs:
        pixel_values = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        original_sizes = [(2, 2)]

        def to(self, device):
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.session_kwargs: list[dict] = []

        def init_video_session(self, **kwargs):
            self.session_kwargs.append(dict(kwargs))
            return FakeInferenceSession()

        def add_text_prompt(self, *, inference_session, text):
            return inference_session

        def __call__(self, **kwargs):
            return FakeInputs()

        def postprocess_outputs(self, *args, **kwargs):
            return {"object_ids": [], "boxes": [], "scores": [], "masks": []}

    class FakeModel:
        device = "cpu"

        def __init__(self) -> None:
            self.frame_indices: list[int] = []

        def __call__(
            self, *, inference_session, frame_idx=None, frame=None, reverse=False
        ):
            frame_idx = inference_session.add_new_frame(frame, frame_idx)
            self.frame_indices.append(frame_idx)
            return SimpleNamespace()

    backend = object.__new__(HFSam3Backend)
    backend.device = "cpu"
    backend.dtype = np.float32
    backend.memory_bank_size = 7
    backend.max_vision_features_cache_size = 1
    backend.processor = FakeProcessor()
    backend.model = FakeModel()
    handle = backend.create_session(
        "cam-a",
        prompt="person",
        config={"memory_bank_size": 3, "max_vision_features_cache_size": 4},
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    for frame_idx in range(10):
        backend.update(handle, frame, float(frame_idx))

    assert backend.model.frame_indices == list(range(10))
    inference_session = getattr(handle, "inference_session", handle)
    assert list(inference_session.processed_frames) == list(range(7, 10))
    assert backend.processor.session_kwargs[0]["max_vision_features_cache_size"] == 4


def test_hf_backend_treats_none_masks_as_no_tracks():
    class FakeInputs:
        pixel_values = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        original_sizes = [(2, 2)]

        def to(self, device):
            return self

    class FakeProcessor:
        def init_video_session(self, **kwargs):
            return SimpleNamespace(
                processed_frames={},
                output_dict_per_obj={},
                frames_tracked_per_obj={},
                obj_id_to_tracker_score_frame_wise={},
            )

        def add_text_prompt(self, *, inference_session, text):
            return inference_session

        def __call__(self, **kwargs):
            return FakeInputs()

        def postprocess_outputs(self, *args, **kwargs):
            return {
                "object_ids": np.array([1]),
                "boxes": np.array([[0, 0, 1, 1]]),
                "scores": np.array([0.9]),
                "masks": None,
            }

    class FakeModel:
        device = "cpu"

        def __call__(self, **kwargs):
            return SimpleNamespace()

    backend = object.__new__(HFSam3Backend)
    backend.device = "cpu"
    backend.dtype = np.float32
    backend.memory_bank_size = 7
    backend.max_vision_features_cache_size = 1
    backend.processor = FakeProcessor()
    backend.model = FakeModel()
    handle = backend.create_session("cam-a", prompt="person", config={})

    assert backend.update(handle, np.zeros((2, 2, 3), dtype=np.uint8), 1.0) == []
