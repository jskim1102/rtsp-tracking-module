"""SAM3 video tracker adapter backed by Hugging Face Transformers.

The Transformers import and gated checkpoint load happen only inside the tracker
subprocess. Unit tests inject a tiny fake backend and never download weights.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from app import config as app_config
from app.tracking.base import Track

logger = logging.getLogger("rtsp-tracking.tracking.sam3")


@dataclass(frozen=True, slots=True)
class BackendTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    polygon: list[tuple[int, int]] | None
    score: float


@dataclass(slots=True)
class _HFSessionHandle:
    """Per-camera HF state plus a counter independent of pruned frame stores."""

    inference_session: Any
    memory_bank_size: int
    next_frame_idx: int = 0


class Sam3Backend(Protocol):
    def create_session(self, camera_id: str, *, prompt: str, config: dict) -> Any: ...

    def update(self, handle: Any, frame: np.ndarray, ts: float) -> list[BackendTrack]: ...

    def close(self, handle: Any) -> None: ...


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name) or default))
    except ValueError:
        return default


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _output_value(outputs: Any, key: str, default: Any) -> Any:
    if isinstance(outputs, dict):
        value = outputs.get(key, default)
    else:
        value = getattr(outputs, key, default)
    return default if value is None else value


def _mask_polygon(mask: Any) -> list[tuple[int, int]] | None:
    """Binary mask → largest external contour in request-frame coordinates."""
    import cv2

    array = np.squeeze(_to_numpy(mask))
    if array.ndim != 2:
        return None
    binary = (array > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 3:
        return None
    return [(int(x), int(y)) for x, y in contour]


def _prune_mapping(mapping: Any, limit: int) -> None:
    if not isinstance(mapping, dict) or len(mapping) <= limit:
        return
    for key in sorted(mapping)[:-limit]:
        mapping.pop(key, None)


def prune_hf_session_memory(inference_session: Any, memory_bank_size: int) -> None:
    """Bound frame-indexed SAM3 session storage for long-running RTSP streams.

    Transformers' ``num_maskmem`` limits memories consumed by attention, while its
    streaming session still retains frame-indexed dictionaries. Keep only the newest
    configured window in those stores so memory does not grow with stream duration.
    """
    limit = max(1, int(memory_bank_size))
    _prune_mapping(getattr(inference_session, "processed_frames", None), limit)
    for per_object in getattr(inference_session, "output_dict_per_obj", {}).values():
        if not isinstance(per_object, dict):
            continue
        _prune_mapping(per_object.get("cond_frame_outputs"), limit)
        _prune_mapping(per_object.get("non_cond_frame_outputs"), limit)
    for per_object in getattr(inference_session, "frames_tracked_per_obj", {}).values():
        _prune_mapping(per_object, limit)
    for frame_scores in getattr(
        inference_session, "obj_id_to_tracker_score_frame_wise", {}
    ).values():
        _prune_mapping(frame_scores, limit)


class HFSam3Backend:
    """Thin lazy-loaded boundary around Transformers 5.10.x SAM3 video APIs."""

    def __init__(self, config: dict) -> None:
        token = os.getenv("HF_TOKEN") or ""
        if not token:
            raise RuntimeError(
                "HF_TOKEN is required to load gated SAM3 checkpoint "
                f"{config['checkpoint']!r}; request access on Hugging Face and set HF_TOKEN"
            )

        import torch
        from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

        requested_device = str(config.get("device") or "")
        self.device = requested_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fp16 = bool(config.get("fp16", True)) and self.device.startswith("cuda")
        self.dtype = torch.float16 if self.fp16 else torch.float32
        self.memory_bank_size = max(1, int(config["memory_bank_size"]))
        self.max_vision_features_cache_size = max(
            1, int(config["max_vision_features_cache_size"])
        )
        checkpoint = str(config["checkpoint"])
        input_size = int(config["input_size"])

        model_config = Sam3VideoConfig.from_pretrained(checkpoint, token=token)
        model_config.image_size = input_size
        if getattr(model_config, "tracker_config", None) is not None:
            model_config.tracker_config.num_maskmem = self.memory_bank_size
        model_config.recondition_every_nth_frame = int(
            config["recondition_every_nth_frame"]
        )
        self.model = Sam3VideoModel.from_pretrained(
            checkpoint,
            config=model_config,
            token=token,
            dtype=self.dtype,
        ).to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self.processor = Sam3VideoProcessor.from_pretrained(
            checkpoint,
            token=token,
            size={"height": input_size, "width": input_size},
        )

    def create_session(self, camera_id: str, *, prompt: str, config: dict) -> Any:
        memory_bank_size = max(
            1,
            int(config.get("memory_bank_size", self.memory_bank_size)),
        )
        max_vision_features_cache_size = max(
            1,
            int(
                config.get(
                    "max_vision_features_cache_size",
                    self.max_vision_features_cache_size,
                )
            ),
        )
        session = self.processor.init_video_session(
            inference_device=self.device,
            inference_state_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            max_vision_features_cache_size=max_vision_features_cache_size,
            dtype=self.dtype,
        )
        session = self.processor.add_text_prompt(inference_session=session, text=prompt)
        return _HFSessionHandle(
            inference_session=session,
            memory_bank_size=memory_bank_size,
        )

    def update(self, handle: Any, frame: np.ndarray, ts: float) -> list[BackendTrack]:
        inference_session = handle.inference_session
        # OpenCV capture is BGR; HF image processors expect RGB.
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        inputs = self.processor(
            images=rgb,
            device=self.device,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model(
            inference_session=inference_session,
            frame_idx=handle.next_frame_idx,
            frame=inputs.pixel_values[0],
            reverse=False,
        )
        # The model has committed this frame to the session. Advance before
        # post-processing so a post-processing failure cannot overwrite it.
        handle.next_frame_idx += 1
        processed = self.processor.postprocess_outputs(
            inference_session,
            outputs,
            original_sizes=inputs.original_sizes,
        )
        object_ids = _to_numpy(_output_value(processed, "object_ids", [])).reshape(-1)
        boxes = _to_numpy(_output_value(processed, "boxes", [])).reshape(-1, 4)
        scores = _to_numpy(_output_value(processed, "scores", [])).reshape(-1)
        masks = _output_value(processed, "masks", [])
        count = min(len(object_ids), len(boxes), len(scores), len(masks))
        tracks = [
            BackendTrack(
                track_id=int(object_ids[index]),
                bbox=tuple(int(round(float(value))) for value in boxes[index]),
                polygon=_mask_polygon(masks[index]),
                score=float(scores[index]),
            )
            for index in range(count)
        ]
        prune_hf_session_memory(inference_session, handle.memory_bank_size)
        return tracks

    def close(self, handle: Any) -> None:
        reset = getattr(handle.inference_session, "reset_state", None)
        if callable(reset):
            reset()


class _Sam3Session:
    def __init__(
        self,
        *,
        backend: Sam3Backend,
        camera_id: str,
        prompt: str,
        config: dict,
        confirm_frames: int,
    ) -> None:
        self._backend = backend
        self._camera_id = camera_id
        self._prompt = prompt
        self._config = dict(config)
        self._confirm_frames = max(1, int(confirm_frames))
        self._track_state_ttl_frames = max(
            1,
            int(config.get("track_state_ttl_frames", 300)),
        )
        self._handle = self._new_handle()
        self._consecutive: dict[int, int] = {}
        self._confirmed: set[int] = set()
        self._announced: set[int] = set()
        self._last_seen_frame: dict[int, int] = {}
        self._frame_index = 0
        self._closed = False

    def _new_handle(self) -> Any:
        return self._backend.create_session(
            self._camera_id,
            prompt=self._prompt,
            config=dict(self._config),
        )

    def update(self, frame: np.ndarray, ts: float) -> list[Track]:
        if self._closed:
            raise RuntimeError("tracker session is closed")
        self._frame_index += 1
        raw_tracks = self._backend.update(self._handle, frame, ts)
        current_ids = {item.track_id for item in raw_tracks}
        for track_id in current_ids:
            self._last_seen_frame[track_id] = self._frame_index
        expired_ids = {
            track_id
            for track_id, last_seen in self._last_seen_frame.items()
            if self._frame_index - last_seen > self._track_state_ttl_frames
        }
        for track_id in expired_ids:
            self._last_seen_frame.pop(track_id, None)
            self._consecutive.pop(track_id, None)
            self._confirmed.discard(track_id)
            self._announced.discard(track_id)
        for track_id in list(self._consecutive):
            if track_id not in current_ids and track_id not in self._confirmed:
                self._consecutive.pop(track_id, None)

        results: list[Track] = []
        for item in raw_tracks:
            if item.track_id not in self._confirmed:
                count = self._consecutive.get(item.track_id, 0) + 1
                self._consecutive[item.track_id] = count
                if count < self._confirm_frames:
                    continue
                self._confirmed.add(item.track_id)
            is_new = item.track_id not in self._announced
            results.append(
                Track(
                    track_id=item.track_id,
                    bbox=item.bbox,
                    polygon=item.polygon,
                    score=item.score,
                    is_new=is_new,
                )
            )
        return results

    def acknowledge_new_tracks(self, track_ids: list[int]) -> None:
        """Stop retrying ``is_new`` only after the parent persisted the crop.

        The result queue is latest-wins, so the first confirmed result can be
        dropped. Keeping announcement ownership in the gallery makes a later
        frame retry the crop/event until storage explicitly acknowledges it.
        """
        if self._closed:
            return
        self._announced.update(
            int(track_id)
            for track_id in track_ids
            if int(track_id) in self._confirmed
        )

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("tracker session is closed")
        self._backend.close(self._handle)
        self._handle = self._new_handle()
        self._consecutive.clear()
        self._confirmed.clear()
        self._announced.clear()
        self._last_seen_frame.clear()
        self._frame_index = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._backend.close(self._handle)
        self._consecutive.clear()
        self._confirmed.clear()
        self._announced.clear()
        self._last_seen_frame.clear()


class Sam3Adapter:
    name = "sam3"
    capabilities = {"concept_prompt": True}

    def __init__(
        self,
        *,
        backend: Sam3Backend | None = None,
        confirm_frames: int | None = None,
        concept_prompt: str | None = None,
        config: dict | None = None,
    ) -> None:
        defaults = {
            "checkpoint": os.getenv("TRACKER_CHECKPOINT") or app_config.TRACKER_CHECKPOINT,
            "device": os.getenv("TRACKER_DEVICE") or app_config.TRACKER_DEVICE,
            "input_size": _env_int("TRACKER_INPUT_SIZE", app_config.TRACKER_INPUT_SIZE),
            "fp16": (os.getenv("TRACKER_FP16") or str(app_config.TRACKER_FP16)).lower()
            in {"1", "true", "yes", "on"},
            "memory_bank_size": _env_int(
                "TRACKER_MEMORY_BANK_SIZE", app_config.TRACKER_MEMORY_BANK_SIZE
            ),
            "recondition_every_nth_frame": _env_int(
                "TRACKER_RECONDITION_EVERY_NTH_FRAME",
                app_config.TRACKER_RECONDITION_EVERY_NTH_FRAME,
                minimum=0,
            ),
            "max_vision_features_cache_size": _env_int(
                "TRACKER_MAX_VISION_FEATURES_CACHE_SIZE",
                app_config.TRACKER_MAX_VISION_FEATURES_CACHE_SIZE,
            ),
        }
        defaults.update(config or {})
        self._config = defaults
        self._confirm_frames = (
            app_config.TRACK_CONFIRM_FRAMES
            if confirm_frames is None
            else confirm_frames
        )
        self._concept_prompt = (
            concept_prompt
            or os.getenv("TRACKER_CONCEPT_PROMPT")
            or app_config.TRACKER_CONCEPT_PROMPT
        )
        self._backend = backend or HFSam3Backend(self._config)
        self.device = str(getattr(self._backend, "device", ""))

    def create_session(self, camera_id: str, config: dict) -> _Sam3Session:
        session_config = dict(self._config)
        session_config.update(config)
        return _Sam3Session(
            backend=self._backend,
            camera_id=camera_id,
            prompt=self._concept_prompt,
            config=session_config,
            confirm_frames=self._confirm_frames,
        )
