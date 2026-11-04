"""Real-weight SAM3 streaming eviction measurement.

This test is intentionally skipped in ordinary CI unless both a Hugging Face token
and CUDA are available.  It never prints the registered RTSP URL or credentials.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import median
from urllib.parse import quote

import cv2
import numpy as np
import pytest

from app.database import SessionLocal
from app.models import IpCam
from app.tracking.sam3_adapter import HFSam3Backend, Sam3Adapter


_CHECKPOINT = "facebook/sam3"
_INPUT_SIZE = 560
_MEMORY_BANK_SIZE = 7
_TTL_FRAMES = 20
_STRESS_FRAMES = int(os.getenv("SAM3_EVICTION_STRESS_FRAMES", "2000"))
_RESULT_PATH = Path(".pytest_cache/sam3_eviction_real.json")


def _capture_registered_frames(frames_per_camera: int = 8) -> list[np.ndarray]:
    db = SessionLocal()
    try:
        cameras = db.query(IpCam).order_by(IpCam.id).all()
    finally:
        db.close()

    viewer_user = quote(
        os.getenv("MEDIAMTX_VIEWER_USER", "viewer_rtsp_tracking"),
        safe="",
    )
    viewer_password = quote(os.getenv("MEDIAMTX_VIEWER_PASS", ""), safe="")
    rtsp_port = os.getenv("MEDIAMTX_RTSP_PORT", "8554")
    captured: list[np.ndarray] = []

    for camera in cameras:
        if viewer_password:
            source = (
                f"rtsp://{viewer_user}:{viewer_password}@127.0.0.1:"
                f"{rtsp_port}/{camera.stream_key}"
            )
        else:
            source = camera.rtsp_url

        capture = cv2.VideoCapture()
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000)
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000)
        try:
            if not capture.open(source, cv2.CAP_FFMPEG):
                continue
            for _ in range(frames_per_camera):
                ok, frame = capture.read()
                if ok and frame is not None and frame.size:
                    captured.append(frame.copy())
        finally:
            capture.release()

    if not captured:
        pytest.fail("registered RTSP cameras yielded no frame for the SAM3 real test")
    return captured


def _session_snapshot(inference_session, *, frame_index: int, phase: str) -> dict:
    import torch

    frame_scores = inference_session.obj_id_to_tracker_score_frame_wise
    unmatched = inference_session.unmatched_frame_inds
    overlap = inference_session.overlap_pair_to_frame_inds
    return {
        "frame": frame_index,
        "phase": phase,
        "obj_ids": len(inference_session.obj_ids),
        "obj_id_to_score": len(inference_session.obj_id_to_score),
        "tracker_score_frames": len(frame_scores),
        "tracker_score_entries": sum(
            len(scores) for scores in frame_scores.values()
        ),
        "suppressed_frames": len(inference_session.suppressed_obj_ids),
        "obj_id_to_last_occluded": len(
            inference_session.obj_id_to_last_occluded
        ),
        "obj_first_frame_idx": len(inference_session.obj_first_frame_idx),
        "unmatched_ids": len(unmatched),
        "unmatched_entries": sum(len(indices) for indices in unmatched.values()),
        "unmatched_list_max": max(
            (len(indices) for indices in unmatched.values()),
            default=0,
        ),
        "overlap_pairs": len(overlap),
        "overlap_entries": sum(len(indices) for indices in overlap.values()),
        "overlap_list_max": max(
            (len(indices) for indices in overlap.values()),
            default=0,
        ),
        "trk_keep_alive": len(inference_session.trk_keep_alive),
        "cuda_allocated": torch.cuda.memory_allocated(),
    }


def _max_metric(samples: list[dict], key: str) -> int:
    return max((int(sample[key]) for sample in samples), default=0)


def test_real_sam3_evicts_expired_tracks_and_stays_bounded():
    token = os.getenv("HF_TOKEN") or ""
    if not token:
        pytest.skip("HF_TOKEN is not set")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    gpu_name = torch.cuda.get_device_name(0)
    assert "RTX 3090" in gpu_name, f"measurement requires RTX 3090, got {gpu_name}"
    assert _STRESS_FRAMES >= 1

    source_frames = _capture_registered_frames()
    config = {
        "checkpoint": _CHECKPOINT,
        "device": "cuda:0",
        "input_size": _INPUT_SIZE,
        "fp16": True,
        "memory_bank_size": _MEMORY_BANK_SIZE,
        "recondition_every_nth_frame": 16,
        "max_vision_features_cache_size": 1,
    }
    backend = HFSam3Backend(config)

    person_frame: np.ndarray | None = None
    probe_handle = backend.create_session(
        "sam3-eviction-probe",
        prompt="person",
        config=config,
    )
    try:
        for frame_index, candidate in enumerate(source_frames):
            if backend.update(probe_handle, candidate, float(frame_index)):
                person_frame = candidate
                break
    finally:
        backend.close(probe_handle)
    if person_frame is None:
        pytest.fail("registered RTSP frames contained no SAM3-detectable person")

    blank_frame = np.zeros_like(person_frame)
    adapter = Sam3Adapter(
        backend=backend,
        confirm_frames=1,
        concept_prompt="person",
        config=config,
    )
    tracker = adapter.create_session(
        "sam3-eviction-real",
        {"track_state_ttl_frames": _TTL_FRAMES},
    )
    inference_session = tracker._handle.inference_session
    samples: list[dict] = []
    frame_index = 0

    def update(frame: np.ndarray, phase: str):
        nonlocal frame_index
        tracks = tracker.update(frame, float(frame_index))
        torch.cuda.synchronize()
        samples.append(
            _session_snapshot(
                inference_session,
                frame_index=frame_index,
                phase=phase,
            )
        )
        frame_index += 1
        assert samples[-1]["tracker_score_frames"] <= _MEMORY_BANK_SIZE
        assert samples[-1]["suppressed_frames"] <= _MEMORY_BANK_SIZE
        assert samples[-1]["unmatched_list_max"] <= _MEMORY_BANK_SIZE
        assert samples[-1]["overlap_list_max"] <= _MEMORY_BANK_SIZE
        return tracks

    try:
        first_tracks = []
        first_person_frames = 0
        for first_person_frames in range(1, 41):
            first_tracks = update(person_frame, "initial-person")
            if first_tracks:
                break
        assert first_tracks, "SAM3 did not reacquire the measured person frame"
        first_ids = sorted(track.track_id for track in first_tracks)
        first_max_id = max(first_ids)
        initial_object_peak = _max_metric(samples, "obj_ids")

        eviction_blank_frames = 0
        for eviction_blank_frames in range(1, _TTL_FRAMES + 81):
            update(blank_frame, "initial-empty")
            if not inference_session.obj_ids:
                break
        assert inference_session.obj_ids == []
        assert eviction_blank_frames > _TTL_FRAMES
        assert inference_session.obj_id_to_score == {}
        assert inference_session.obj_id_to_last_occluded == {}
        assert inference_session.obj_first_frame_idx == {}
        assert inference_session.unmatched_frame_inds == {}
        assert inference_session.overlap_pair_to_frame_inds == {}
        assert inference_session.trk_keep_alive == {}
        assert set(inference_session.prompts.values()) == {"person"}

        second_tracks = []
        reentry_person_frames = 0
        for reentry_person_frames in range(1, 41):
            second_tracks = update(person_frame, "reentry-person")
            if second_tracks:
                break
        assert second_tracks, "SAM3 did not detect a person after last-object reset"
        second_ids = sorted(track.track_id for track in second_tracks)
        assert min(second_ids) > first_max_id

        person_span = max(3, reentry_person_frames + 2)
        empty_span = eviction_blank_frames + 2
        cycle_span = person_span + empty_span
        zero_transition_vram: list[int] = []
        stress_start = len(samples)
        previous_count = len(inference_session.obj_ids)
        max_id_before_stress = inference_session.max_obj_id

        for stress_index in range(_STRESS_FRAMES):
            cycle_index = stress_index % cycle_span
            if cycle_index < person_span:
                frame = person_frame
                phase = "stress-person"
            else:
                frame = blank_frame
                phase = "stress-empty"
            update(frame, phase)
            current_count = len(inference_session.obj_ids)
            if previous_count and not current_count:
                zero_transition_vram.append(samples[-1]["cuda_allocated"])
            previous_count = current_count

        stress_samples = samples[stress_start:]
        assert len(stress_samples) == _STRESS_FRAMES
        expected_transitions = min(5, max(1, _STRESS_FRAMES // cycle_span))
        assert len(zero_transition_vram) >= expected_transitions
        assert inference_session.max_obj_id > max_id_before_stress
        object_bound = max(initial_object_peak + 1, len(second_ids) + 1)
        assert _max_metric(stress_samples, "obj_ids") <= object_bound
        assert _max_metric(stress_samples, "obj_id_to_score") <= object_bound
        assert _max_metric(stress_samples, "obj_first_frame_idx") <= object_bound
        assert _max_metric(stress_samples, "unmatched_ids") <= object_bound
        assert _max_metric(stress_samples, "trk_keep_alive") <= object_bound

        mib = 1024 * 1024
        vram_values = [sample["cuda_allocated"] for sample in stress_samples]
        vram_range = max(vram_values) - min(vram_values)
        assert vram_range <= 768 * mib
        plateau_delta = int(median(zero_transition_vram[-5:])) - int(
            median(zero_transition_vram[:5])
        )
        assert plateau_delta <= 64 * mib

        summary = {
            "checkpoint": _CHECKPOINT,
            "cuda_allocated_end_mib": round(vram_values[-1] / mib, 2),
            "cuda_allocated_max_mib": round(max(vram_values) / mib, 2),
            "cuda_allocated_min_mib": round(min(vram_values) / mib, 2),
            "cuda_allocated_range_mib": round(vram_range / mib, 2),
            "eviction_blank_frames": eviction_blank_frames,
            "first_ids": first_ids,
            "gpu": gpu_name,
            "input_size": _INPUT_SIZE,
            "max_metadata": {
                key: _max_metric(stress_samples, key)
                for key in (
                    "obj_id_to_score",
                    "tracker_score_frames",
                    "tracker_score_entries",
                    "suppressed_frames",
                    "obj_id_to_last_occluded",
                    "obj_first_frame_idx",
                    "unmatched_ids",
                    "unmatched_entries",
                    "unmatched_list_max",
                    "overlap_pairs",
                    "overlap_entries",
                    "overlap_list_max",
                    "trk_keep_alive",
                )
            },
            "max_obj_ids": _max_metric(stress_samples, "obj_ids"),
            "memory_bank_size": _MEMORY_BANK_SIZE,
            "new_max_obj_id": inference_session.max_obj_id,
            "plateau_delta_mib": round(plateau_delta / mib, 2),
            "reentry_ids": second_ids,
            "source": "registered_rtsp",
            "stress_frames": _STRESS_FRAMES,
            "ttl_frames": _TTL_FRAMES,
            "zero_transitions": len(zero_transition_vram),
        }
        _RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RESULT_PATH.write_text(
            json.dumps(
                {"summary": summary, "frames": samples},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("SAM3_EVICTION_RESULT=" + json.dumps(summary, sort_keys=True))
    finally:
        tracker.close()
