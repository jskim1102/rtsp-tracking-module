"""Single-process, per-camera stateful tracker runner.

Exactly one child process owns one adapter/model and a Session per camera. Requests
are consumed sequentially; frames from different cameras are never tensor-batched.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from typing import Any, Callable

from app import config
from app.inference.worker import FrameRequest, InferenceResult
from app.tracking.base import TrackerAdapter, TrackerSession
from app.tracking.registry import available_trackers, create_tracker

logger = logging.getLogger("rtsp-tracking.tracking.runner")


class SessionResolutionChanged(ValueError):
    def __init__(
        self,
        camera_id: str,
        expected: tuple[int, int],
        received: tuple[int, int],
    ) -> None:
        self.camera_id = camera_id
        self.expected = expected
        self.received = received
        super().__init__(
            f"camera {camera_id!r} resolution changed during tracker session: "
            f"expected {expected}, received {received}"
        )


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    message: str


@dataclass(frozen=True, slots=True)
class _SourceGeneration:
    source_id: str
    generation: int


@dataclass(frozen=True, slots=True)
class _TrackAcknowledgement:
    source_id: str
    generation: int
    track_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _FrameEnvelope:
    request: FrameRequest
    generation: int


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    result: InferenceResult
    generation: int


class _StaleSourceGeneration(ValueError):
    pass


_QUEUE_RECOVERY_ERRORS = (
    EOFError,
    OSError,
    ValueError,
    pickle.UnpicklingError,
    AttributeError,
    TypeError,
)


class _SessionRouter:
    """Child-process state: adapter plus isolated camera sessions."""

    def __init__(self, adapter: TrackerAdapter, session_config: dict) -> None:
        self.adapter = adapter
        self.session_config = dict(session_config)
        self.sessions: dict[str, TrackerSession] = {}
        self.resolutions: dict[str, tuple[int, int]] = {}
        self.generations: dict[str, int] = {}

    def _advance_generation(self, camera_id: str, generation: int) -> None:
        current = self.generations.get(camera_id, 0)
        if generation < current:
            raise _StaleSourceGeneration(
                f"camera {camera_id!r} stale generation {generation} < {current}"
            )
        if generation == current:
            return
        session = self.sessions.pop(camera_id, None)
        self.resolutions.pop(camera_id, None)
        self.generations[camera_id] = generation
        if session is not None:
            session.close()

    def process(self, request: FrameRequest, generation: int = 0) -> InferenceResult:
        self._advance_generation(request.source_id, generation)
        frame_h, frame_w = request.frame.shape[:2]
        received = (int(frame_h), int(frame_w))
        expected = self.resolutions.get(request.source_id)
        if expected is not None and expected != received:
            raise SessionResolutionChanged(request.source_id, expected, received)
        session = self.sessions.get(request.source_id)
        if session is None:
            session = self.adapter.create_session(
                request.source_id,
                dict(self.session_config),
            )
            self.sessions[request.source_id] = session
            self.resolutions[request.source_id] = received
            self.generations[request.source_id] = generation
        tracks = session.update(request.frame, request.timestamp)
        new_track_crops: dict[int, Any] = {}
        for track in tracks:
            if not track.is_new:
                continue
            x1, y1, x2, y2 = track.bbox
            x1 = max(0, min(int(frame_w), int(x1)))
            y1 = max(0, min(int(frame_h), int(y1)))
            x2 = max(0, min(int(frame_w), int(x2)))
            y2 = max(0, min(int(frame_h), int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            new_track_crops[track.track_id] = request.frame[y1:y2, x1:x2].copy()
        return InferenceResult(
            source_id=request.source_id,
            timestamp=request.timestamp,
            frame_w=int(frame_w),
            frame_h=int(frame_h),
            request_id=request.request_id,
            model=self.adapter.name,
            batch_size=1,
            effective_imgsz=int(request.imgsz),
            device=str(getattr(self.adapter, "device", "")),
            tracker_generation=generation,
            tracks=tracks,
            new_track_crops=new_track_crops,
        )

    def remove_source(self, camera_id: str, generation: int) -> None:
        """Close state from generations older than the parent's new fence."""
        self._advance_generation(camera_id, generation)

    def acknowledge_tracks(
        self,
        camera_id: str,
        track_ids: list[int],
        *,
        generation: int,
    ) -> bool:
        """Acknowledge only the session generation that produced the crop."""
        if self.generations.get(camera_id, 0) != generation:
            return False
        session = self.sessions.get(camera_id)
        acknowledge = getattr(session, "acknowledge_new_tracks", None)
        if session is None or not callable(acknowledge):
            return False
        acknowledge([int(track_id) for track_id in track_ids])
        return True

    def reset(self) -> None:
        for session in self.sessions.values():
            try:
                session.reset()
            except Exception:
                logger.exception("Tracker session reset failed")

    def close(self) -> None:
        for session in self.sessions.values():
            try:
                session.close()
            except Exception:
                logger.exception("Tracker session close failed")
        self.sessions.clear()
        self.resolutions.clear()
        self.generations.clear()


def _build_adapter(
    tracker_name: str,
    adapter_config: dict,
    adapter_factory: Callable[[dict], TrackerAdapter] | None,
) -> TrackerAdapter:
    if adapter_factory is not None:
        return adapter_factory(dict(adapter_config))
    return create_tracker(tracker_name, config=dict(adapter_config))


def _put_latest(out_q: Any, value: _WorkerResult | _WorkerFailure) -> None:
    try:
        out_q.put_nowait(value)
    except queue.Full:
        try:
            out_q.get_nowait()
            out_q.put_nowait(value)
        except (queue.Empty, queue.Full, OSError, ValueError):
            pass


def _close_mp_queue(ipc_queue: Any) -> None:
    if ipc_queue is None:
        return
    cancel_join = getattr(ipc_queue, "cancel_join_thread", None)
    if callable(cancel_join):
        cancel_join()
    close = getattr(ipc_queue, "close", None)
    if callable(close):
        close()


def _tracker_worker_main(
    tracker_name: str,
    adapter_config: dict,
    in_q: Any,
    out_q: Any,
    control_q: Any,
    stop_event: Any,
    adapter_factory: Callable[[dict], TrackerAdapter] | None,
) -> None:
    worker_logger = logging.getLogger("rtsp-tracking.tracking.runner.child")
    try:
        adapter = _build_adapter(tracker_name, adapter_config, adapter_factory)
    except Exception as exc:
        message = f"tracker startup failed ({tracker_name}): {exc}"
        worker_logger.exception(message)
        _put_latest(out_q, _WorkerFailure(message))
        return

    router = _SessionRouter(adapter, adapter_config)
    measured = False
    idle_started = time.perf_counter()
    try:
        while not stop_event.is_set():
            while True:
                try:
                    control = control_q.get_nowait()
                except queue.Empty:
                    break
                except _QUEUE_RECOVERY_ERRORS:
                    worker_logger.exception("Tracker control queue read failed")
                    break
                if isinstance(control, _SourceGeneration):
                    try:
                        router.remove_source(control.source_id, control.generation)
                    except Exception:
                        worker_logger.exception(
                            "Tracker source teardown failed: camera=%s",
                            control.source_id,
                        )
                elif isinstance(control, _TrackAcknowledgement):
                    try:
                        router.acknowledge_tracks(
                            control.source_id,
                            list(control.track_ids),
                            generation=control.generation,
                        )
                    except Exception:
                        worker_logger.exception(
                            "Tracker gallery acknowledgement failed: camera=%s",
                            control.source_id,
                        )
            try:
                item = in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            except _QUEUE_RECOVERY_ERRORS:
                worker_logger.exception("Tracker input queue read failed")
                continue
            if item is None:
                break
            if isinstance(item, _FrameEnvelope):
                request = item.request
                generation = item.generation
            else:
                request = item
                generation = 0
            idle_ms = (time.perf_counter() - idle_started) * 1000.0
            started = time.perf_counter()
            try:
                result = router.process(request, generation=generation)
            except _StaleSourceGeneration:
                idle_started = time.perf_counter()
                continue
            except SessionResolutionChanged:
                worker_logger.exception("Tracker session resolution fence rejected frame")
                idle_started = time.perf_counter()
                continue
            except Exception:
                worker_logger.exception(
                    "Tracker update failed: tracker=%s camera=%s",
                    tracker_name,
                    request.source_id,
                )
                idle_started = time.perf_counter()
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if measured:
                result.infer_ms = elapsed_ms
                result.idle_ms = idle_ms
            measured = True
            _put_latest(out_q, _WorkerResult(result=result, generation=generation))
            idle_started = time.perf_counter()
    finally:
        router.close()
        worker_logger.info("Tracker worker exiting: %s", tracker_name)


class TrackerRunner:
    """Parent-side facade preserving the manager worker lifecycle contract."""

    def __init__(
        self,
        tracker_name: str | None = None,
        device: str | None = None,
        *,
        adapter_config: dict | None = None,
        adapter_factory: Callable[[dict], TrackerAdapter] | None = None,
        mp_start_method: str = "spawn",
        pending_sources_max: int | None = None,
    ) -> None:
        self.model_name = tracker_name or config.TRACKER_DEFAULT
        if adapter_factory is None and self.model_name not in available_trackers():
            raise ValueError(f"unregistered tracker: {self.model_name}")
        self.device = device or config.TRACKER_DEVICE or None
        self._adapter_config = {
            "checkpoint": config.TRACKER_CHECKPOINT,
            "device": self.device or "",
            "input_size": config.TRACKER_INPUT_SIZE,
            "fp16": config.TRACKER_FP16,
            "memory_bank_size": config.TRACKER_MEMORY_BANK_SIZE,
            "recondition_every_nth_frame": config.TRACKER_RECONDITION_EVERY_NTH_FRAME,
            "max_vision_features_cache_size": config.TRACKER_MAX_VISION_FEATURES_CACHE_SIZE,
        }
        self._adapter_config.update(adapter_config or {})
        self._adapter_factory = adapter_factory
        self._ctx = mp.get_context(mp_start_method)
        self._pending_sources_max = max(
            1,
            int(pending_sources_max or config.TRACKER_PENDING_SOURCES_MAX),
        )
        self._lifecycle_lock = threading.RLock()
        self._lock = threading.RLock()
        self._pending: OrderedDict[str, _FrameEnvelope] = OrderedDict()
        self._ready_results: deque[InferenceResult] = deque()
        self._source_shapes: dict[str, tuple[int, int]] = {}
        self._source_generations: dict[str, int] = {}
        self._generation_epoch = 0
        self._resolution_changes: set[str] = set()
        self._next_request_id = 1
        self._running = False
        self._enabled = True
        self._proc = None
        self._in_q = None
        self._out_q = None
        self._control_q = None
        self._stop_event = None
        self._submit_count = 0
        self._drop_count = 0
        self._duty_ewma = 0.0
        self._last_error = ""

    def _discard_previous_output_locked(self, out_q: Any) -> None:
        if out_q is None:
            return
        while True:
            try:
                item = out_q.get_nowait()
            except queue.Empty:
                return
            except _QUEUE_RECOVERY_ERRORS as exc:
                self._last_error = f"previous worker output discard failed: {exc}"
                logger.warning("%s", self._last_error)
                return
            if isinstance(item, _WorkerFailure):
                self._last_error = item.message

    def start(self) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._proc is not None and self._proc.is_alive():
                    self._running = True
                    return
                old_proc = self._proc
                old_in_q = self._in_q
                old_out_q = self._out_q
                old_control_q = self._control_q
                if old_proc is not None:
                    old_proc.join(timeout=0)
                self._discard_previous_output_locked(old_out_q)
                _close_mp_queue(old_in_q)
                _close_mp_queue(old_out_q)
                _close_mp_queue(old_control_q)
                # A new process owns no prior session state. Pending frames and
                # completed results from the previous generation are invalid.
                self._generation_epoch += 1
                self._pending.clear()
                self._ready_results.clear()
                self._source_shapes.clear()
                self._source_generations.clear()
                self._resolution_changes.clear()
                self._in_q = self._ctx.Queue(maxsize=2)
                self._out_q = self._ctx.Queue(maxsize=16)
                self._control_q = self._ctx.Queue()
                self._stop_event = self._ctx.Event()
                self._proc = self._ctx.Process(
                    target=_tracker_worker_main,
                    args=(
                        self.model_name,
                        dict(self._adapter_config),
                        self._in_q,
                        self._out_q,
                        self._control_q,
                        self._stop_event,
                        self._adapter_factory,
                    ),
                    daemon=True,
                    name=f"tracker-{self.model_name}",
                )
                self._proc.start()
                self._running = True
                logger.info(
                    "Tracker worker started: tracker=%s pid=%s",
                    self.model_name,
                    self._proc.pid,
                )

    def stop(self) -> None:
        with self._lifecycle_lock:
            with self._lock:
                proc = self._proc
                stop_event = self._stop_event
                in_q = self._in_q
                out_q = self._out_q
                control_q = self._control_q
                self._running = False
                self._pending.clear()
                self._ready_results.clear()
                self._source_shapes.clear()
                self._source_generations.clear()
                self._resolution_changes.clear()
            if proc is not None:
                if stop_event is not None:
                    stop_event.set()
                if in_q is not None:
                    try:
                        in_q.put_nowait(None)
                    except (queue.Full, OSError, ValueError):
                        pass
                proc.join(timeout=5)
                if proc.is_alive():
                    logger.warning(
                        "Tracker worker did not exit gracefully: %s", self.model_name
                    )
                    proc.terminate()
                    proc.join(timeout=1)
            with self._lock:
                self._proc = None
                self._in_q = None
                self._out_q = None
                self._control_q = None
                self._stop_event = None
                self._duty_ewma = 0.0
            _close_mp_queue(in_q)
            _close_mp_queue(out_q)
            _close_mp_queue(control_q)
            logger.info("Tracker runner stopped")

    def is_alive(self) -> bool:
        with self._lock:
            return bool(self._running and self._proc is not None and self._proc.is_alive())

    def _queue_generation_locked(self, source_id: str, generation: int) -> bool:
        if self._control_q is None:
            return not self._running
        try:
            self._control_q.put_nowait(_SourceGeneration(source_id, generation))
            return True
        except (queue.Full, OSError, ValueError) as exc:
            self._last_error = f"tracker source control failed: {exc}"
            logger.error("%s", self._last_error)
            return False

    def _current_generation_locked(self, source_id: str) -> int:
        return self._source_generations.get(source_id, self._generation_epoch)

    def is_current_generation(self, source_id: str, generation: int) -> bool:
        """Return whether a result belongs to the live source/process generation."""
        with self._lifecycle_lock:
            with self._lock:
                return int(generation) == self._current_generation_locked(source_id)

    def submit(self, request: FrameRequest) -> bool:
        frame_h, frame_w = request.frame.shape[:2]
        received = (int(frame_h), int(frame_w))
        with self._lifecycle_lock:
            with self._lock:
                self._submit_count += 1
                if not self._enabled:
                    return False
                generation = self._current_generation_locked(request.source_id)
                expected = self._source_shapes.get(request.source_id)
                if expected is not None and expected != received:
                    generation += 1
                    if not self._queue_generation_locked(request.source_id, generation):
                        self._drop_count += 1
                        return False
                    self._source_generations[request.source_id] = generation
                    self._source_shapes[request.source_id] = received
                    self._pending.pop(request.source_id, None)
                    self._ready_results = deque(
                        result
                        for result in self._ready_results
                        if result.source_id != request.source_id
                    )
                    self._resolution_changes.add(request.source_id)
                    logger.warning(
                        "Tracker session recreated after resolution change: "
                        "camera=%s expected=%s received=%s generation=%s",
                        request.source_id,
                        expected,
                        received,
                        generation,
                    )
                else:
                    self._source_shapes.setdefault(request.source_id, received)
                    self._source_generations.setdefault(request.source_id, generation)
                request_id = self._next_request_id
                self._next_request_id += 1
                pending = _FrameEnvelope(
                    request=replace(
                        request,
                        request_id=request_id,
                        imgsz=int(self._adapter_config["input_size"]),
                        model_names=[self.model_name],
                    ),
                    generation=generation,
                )
                displaced = self._pending.get(request.source_id)
                self._pending[request.source_id] = pending
                if displaced is not None:
                    self._drop_count += 1
                while len(self._pending) > self._pending_sources_max:
                    self._pending.popitem(last=False)
                    self._drop_count += 1
        return True

    def drain_resolution_changes(self) -> list[str]:
        with self._lock:
            changed = sorted(self._resolution_changes)
            self._resolution_changes.clear()
        return changed

    def remove_source(self, source_id: str) -> bool:
        """Fence queued work and request child-side session teardown."""
        with self._lifecycle_lock:
            with self._lock:
                generation = self._current_generation_locked(source_id) + 1
                queued = self._queue_generation_locked(source_id, generation)
                if not queued:
                    return False
                self._source_generations[source_id] = generation
                self._source_shapes.pop(source_id, None)
                self._pending.pop(source_id, None)
                self._ready_results = deque(
                    result
                    for result in self._ready_results
                    if result.source_id != source_id
                )
                self._resolution_changes.discard(source_id)
                return True

    def acknowledge_tracks(
        self,
        source_id: str,
        track_ids: list[int],
        *,
        generation: int,
    ) -> bool:
        """Tell the matching child session that gallery persistence succeeded."""
        normalized = tuple(sorted({int(track_id) for track_id in track_ids}))
        if not normalized:
            return True
        with self._lifecycle_lock:
            with self._lock:
                current_generation = self._current_generation_locked(source_id)
                if generation != current_generation or self._control_q is None:
                    return False
                try:
                    self._control_q.put_nowait(
                        _TrackAcknowledgement(
                            source_id=source_id,
                            generation=generation,
                            track_ids=normalized,
                        )
                    )
                    return True
                except (queue.Full, OSError, ValueError) as exc:
                    self._last_error = f"tracker gallery acknowledgement failed: {exc}"
                    logger.error("%s", self._last_error)
                    return False

    def flush_pending(self) -> int:
        moved = 0
        with self._lifecycle_lock:
            with self._lock:
                if self._in_q is None:
                    return 0
                for source_id in list(self._pending):
                    try:
                        self._in_q.put_nowait(self._pending[source_id])
                    except queue.Full:
                        break
                    except (OSError, ValueError):
                        break
                    self._pending.pop(source_id, None)
                    moved += 1
        return moved

    def _observe_duty(self, result: InferenceResult) -> None:
        cycle = float(result.infer_ms) + float(result.idle_ms)
        if result.infer_ms <= 0.0 or cycle <= 0.0:
            return
        duty = max(0.0, min(1.0, float(result.infer_ms) / cycle))
        if self._duty_ewma == 0.0:
            self._duty_ewma = duty
        else:
            alpha = config.AUTOTUNE_EWMA_ALPHA
            self._duty_ewma += alpha * (duty - self._duty_ewma)

    def drain_results(self) -> list[InferenceResult]:
        with self._lifecycle_lock:
            self.flush_pending()
            with self._lock:
                out_q = self._out_q
                collected = list(self._ready_results)
                self._ready_results.clear()
            if out_q is None:
                return collected
            while True:
                try:
                    item = out_q.get_nowait()
                except queue.Empty:
                    break
                except _QUEUE_RECOVERY_ERRORS as exc:
                    with self._lock:
                        self._last_error = f"tracker output queue read failed: {exc}"
                    logger.warning("%s", self._last_error)
                    break
                if isinstance(item, _WorkerFailure):
                    with self._lock:
                        self._last_error = item.message
                    logger.error("%s", item.message)
                    continue
                if isinstance(item, _WorkerResult):
                    result = item.result
                    with self._lock:
                        current_generation = self._current_generation_locked(
                            result.source_id
                        )
                    if item.generation != current_generation:
                        continue
                else:
                    # Compatibility for injected tests/legacy queue contents. New
                    # worker processes always emit _WorkerResult.
                    result = item
                self._observe_duty(result)
                collected.append(result)
            collected.sort(key=lambda result: result.request_id)
            return collected

    def get_result(self, timeout: float = 0.0) -> InferenceResult | None:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            results = self.drain_results()
            if results:
                if len(results) > 1:
                    with self._lock:
                        self._ready_results.extend(results[1:])
                return results[0]
            if timeout <= 0.0 or time.monotonic() >= deadline:
                return None
            time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))

    def get_queue_stats(self) -> dict[str, int]:
        with self._lock:
            return {"submitted": self._submit_count, "dropped": self._drop_count}

    def get_pool_duty(self) -> float:
        with self._lock:
            return max(0.0, min(1.0, self._duty_ewma))

    def get_pool_status(self) -> dict:
        return {
            "models": [self.model_name],
            "workers": {self.model_name: {"alive": self.is_alive()}},
            "batch_max": 1,
            "batch_timeout_ms": 0.0,
        }

    def configure_models(self, model_names: set[str]) -> None:
        """Compatibility hook: stateful tracking always owns exactly one global model."""

    def set_model(self, model_name: str) -> bool:
        if model_name not in available_trackers():
            raise ValueError(f"unregistered tracker: {model_name}")
        with self._lifecycle_lock:
            with self._lock:
                if model_name == self.model_name:
                    return False
                was_running = self._running
            if was_running:
                self.stop()
            with self._lock:
                self.model_name = model_name
            if was_running:
                self.start()
            return True

    def restart(self) -> None:
        with self._lifecycle_lock:
            with self._lock:
                was_running = self._running
            if was_running:
                self.stop()
                self.start()

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)

    def get_status(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "model": self.model_name,
                "device": self.device or "auto",
                "last_error": self._last_error,
            }
