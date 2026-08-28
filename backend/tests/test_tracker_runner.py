from __future__ import annotations

import pickle
import queue
import threading
import time

import numpy as np

from app.inference import FrameRequest
from app.tracking.base import Track
from app.tracking.runner import (
    SessionResolutionChanged,
    TrackerRunner,
    _SessionRouter,
    _WorkerResult,
    _tracker_worker_main,
)


class FakeSession:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.timestamps: list[float] = []
        self.reset_count = 0
        self.fail_reset = False
        self.closed = False
        self.acknowledged: list[int] = []

    def update(self, frame: np.ndarray, ts: float) -> list[Track]:
        self.timestamps.append(ts)
        return [
            Track(
                track_id=100 if self.camera_id == "a" else 200,
                bbox=(1, 2, 3, 4),
                polygon=None,
                score=0.9,
                is_new=len(self.timestamps) == 1,
            )
        ]

    def reset(self) -> None:
        if self.fail_reset:
            raise RuntimeError(f"reset failed: {self.camera_id}")
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True

    def acknowledge_new_tracks(self, track_ids: list[int]) -> None:
        self.acknowledged.extend(track_ids)


class FakeAdapter:
    name = "sam3"
    capabilities = {"concept_prompt": True}

    def __init__(self) -> None:
        self.sessions: dict[str, FakeSession] = {}
        self.created: list[FakeSession] = []

    def create_session(self, camera_id: str, config: dict) -> FakeSession:
        session = FakeSession(camera_id)
        self.sessions[camera_id] = session
        self.created.append(session)
        return session


def make_fake_adapter(config: dict | None = None) -> FakeAdapter:
    return FakeAdapter()


def _request(camera_id: str, ts: float, shape=(6, 8, 3)) -> FrameRequest:
    return FrameRequest(
        source_id=camera_id,
        frame=np.zeros(shape, dtype=np.uint8),
        timestamp=ts,
        imgsz=560,
    )


def test_router_owns_one_ordered_session_per_camera_without_cross_camera_batching():
    adapter = FakeAdapter()
    router = _SessionRouter(adapter, {"input_size": 560})

    a1 = router.process(_request("a", 1.0))
    b1 = router.process(_request("b", 2.0))
    a2 = router.process(_request("a", 3.0))

    assert list(adapter.sessions) == ["a", "b"]
    assert adapter.sessions["a"].timestamps == [1.0, 3.0]
    assert adapter.sessions["b"].timestamps == [2.0]
    assert [a1.tracks[0].track_id, b1.tracks[0].track_id, a2.tracks[0].track_id] == [
        100,
        200,
        100,
    ]
    assert a1.batch_size == b1.batch_size == a2.batch_size == 1


def test_router_rejects_resolution_change_for_the_lifetime_of_a_session():
    router = _SessionRouter(FakeAdapter(), {"input_size": 560})
    router.process(_request("a", 1.0, shape=(6, 8, 3)))

    try:
        router.process(_request("a", 2.0, shape=(8, 8, 3)))
    except SessionResolutionChanged as exc:
        assert exc.camera_id == "a"
        assert exc.expected == (6, 8)
        assert exc.received == (8, 8)
    else:
        raise AssertionError("resolution change was accepted")


def test_router_closes_every_camera_session():
    adapter = FakeAdapter()
    router = _SessionRouter(adapter, {})
    router.process(_request("a", 1.0))
    router.process(_request("b", 2.0))

    router.close()

    assert all(session.closed for session in adapter.sessions.values())


def test_router_remove_source_closes_only_that_session_and_recreates_cleanly():
    adapter = FakeAdapter()
    router = _SessionRouter(adapter, {})
    router.process(_request("a", 1.0))
    router.process(_request("b", 2.0))
    removed = adapter.sessions["a"]

    router.remove_source("a", generation=1)

    assert removed.closed is True
    assert "a" not in router.sessions
    assert "b" in router.sessions
    result = router.process(_request("a", 3.0), generation=1)
    assert result.tracks[0].is_new is True
    assert adapter.created[-1] is not removed


def test_router_applies_gallery_ack_only_to_the_matching_camera_generation():
    adapter = FakeAdapter()
    router = _SessionRouter(adapter, {})
    router.process(_request("a", 1.0), generation=2)

    assert router.acknowledge_tracks("a", [100], generation=1) is False
    assert router.acknowledge_tracks("a", [100], generation=2) is True
    assert adapter.sessions["a"].acknowledged == [100]


def test_router_reset_preserves_the_session_resolution_fence():
    router = _SessionRouter(FakeAdapter(), {})
    router.process(_request("a", 1.0, shape=(6, 8, 3)))

    router.reset()

    try:
        router.process(_request("a", 2.0, shape=(8, 8, 3)))
    except SessionResolutionChanged:
        pass
    else:
        raise AssertionError("reset disabled the live session resolution fence")


def test_router_reset_isolates_a_broken_camera_session():
    adapter = FakeAdapter()
    router = _SessionRouter(adapter, {})
    router.process(_request("a", 1.0))
    router.process(_request("b", 2.0))
    adapter.sessions["a"].fail_reset = True

    router.reset()

    assert adapter.sessions["a"].reset_count == 0
    assert adapter.sessions["b"].reset_count == 1


def test_worker_recovers_from_a_corrupted_input_queue_payload():
    class CorruptedThenClosedQueue:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise pickle.UnpicklingError("truncated input payload")
            return None

    _tracker_worker_main(
        "sam3",
        {},
        CorruptedThenClosedQueue(),
        queue.Queue(),
        queue.Queue(),
        threading.Event(),
        make_fake_adapter,
    )


def test_parent_pending_is_source_bounded_and_latest_wins():
    runner = TrackerRunner(adapter_factory=make_fake_adapter, pending_sources_max=2)

    assert runner.submit(_request("a", 1.0)) is True
    assert runner.submit(_request("b", 2.0)) is True
    assert runner.submit(_request("a", 3.0)) is True
    assert runner.submit(_request("c", 4.0)) is True

    # 기존 source 위치는 유지해 한 busy source가 순서를 독점하지 않는다. 새 source가
    # cap을 넘기면 가장 오래 기다린 slot이 제거된다.
    assert list(runner._pending) == ["b", "c"]
    assert runner.get_queue_stats() == {"submitted": 4, "dropped": 2}


def test_parent_resolution_change_advances_session_generation_and_is_reported():
    runner = TrackerRunner(adapter_factory=make_fake_adapter)

    assert runner.submit(_request("a", 1.0, shape=(6, 8, 3))) is True
    assert runner.drain_resolution_changes() == []
    assert runner.submit(_request("a", 2.0, shape=(8, 8, 3))) is True

    assert runner.drain_resolution_changes() == ["a"]
    assert runner._source_shapes["a"] == (8, 8)


def test_external_lifecycle_processes_frames_in_one_tracker_process():
    runner = TrackerRunner(
        adapter_factory=make_fake_adapter,
        mp_start_method="spawn",
        pending_sources_max=4,
    )
    runner.start()
    try:
        assert runner.is_alive() is True
        assert runner.submit(_request("a", 1.0)) is True
        deadline = time.monotonic() + 3.0
        results = []
        while time.monotonic() < deadline and not results:
            results = runner.drain_results()
            time.sleep(0.01)

        assert len(results) == 1
        assert results[0].source_id == "a"
        assert results[0].tracks[0].track_id == 100
        assert results[0].batch_size == 1
    finally:
        runner.stop()

    assert runner.is_alive() is False


def test_external_remove_source_recreates_only_that_camera_session():
    runner = TrackerRunner(
        adapter_factory=make_fake_adapter,
        mp_start_method="spawn",
        pending_sources_max=4,
    )
    runner.start()
    try:
        assert runner.submit(_request("a", 1.0)) is True
        first = runner.get_result(timeout=3.0)
        assert first is not None and first.tracks[0].is_new is True

        assert runner.remove_source("a") is True
        assert runner.submit(_request("a", 2.0)) is True
        recreated = runner.get_result(timeout=3.0)

        assert recreated is not None
        assert recreated.source_id == "a"
        assert recreated.tracks[0].is_new is True
    finally:
        runner.stop()


def test_external_resolution_change_recreates_the_camera_session():
    runner = TrackerRunner(
        adapter_factory=make_fake_adapter,
        mp_start_method="spawn",
        pending_sources_max=4,
    )
    runner.start()
    try:
        assert runner.submit(_request("a", 1.0, shape=(6, 8, 3))) is True
        first = runner.get_result(timeout=3.0)
        assert first is not None and first.tracks[0].is_new is True

        assert runner.submit(_request("a", 2.0, shape=(8, 8, 3))) is True
        assert runner.drain_resolution_changes() == ["a"]
        recreated = runner.get_result(timeout=3.0)

        assert recreated is not None
        assert (recreated.frame_h, recreated.frame_w) == (8, 8)
        assert recreated.tracks[0].is_new is True
    finally:
        runner.stop()


class _DeadProcess:
    pid = 101

    def join(self, timeout=None) -> None:
        return None

    def is_alive(self) -> bool:
        return False


class _BrokenQueue:
    def get_nowait(self):
        raise pickle.UnpicklingError("truncated multiprocessing payload")

    def cancel_join_thread(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_start_discards_previous_generation_pending_and_results():
    runner = TrackerRunner(
        adapter_factory=make_fake_adapter,
        mp_start_method="spawn",
    )
    stale = _SessionRouter(FakeAdapter(), {}).process(_request("a", 1.0))
    old_out = queue.Queue()
    old_out.put(stale)
    runner._proc = _DeadProcess()
    runner._in_q = queue.Queue()
    runner._out_q = old_out
    runner._pending["a"] = _request("a", 2.0)
    runner._ready_results.append(stale)

    runner.start()
    try:
        assert list(runner._pending) == []
        assert runner.drain_results() == []
    finally:
        runner.stop()


def test_start_recovers_from_a_corrupted_previous_output_queue():
    runner = TrackerRunner(
        adapter_factory=make_fake_adapter,
        mp_start_method="spawn",
    )
    runner._proc = _DeadProcess()
    runner._in_q = queue.Queue()
    runner._out_q = _BrokenQueue()

    runner.start()
    try:
        assert runner.is_alive() is True
        assert "discard failed" in runner.get_status()["last_error"]
    finally:
        runner.stop()


def test_drain_results_recovers_from_a_corrupted_live_output_queue():
    runner = TrackerRunner(adapter_factory=make_fake_adapter)
    runner._out_q = _BrokenQueue()

    assert runner.drain_results() == []
    assert "output queue read failed" in runner.get_status()["last_error"]


def test_drain_results_drops_an_older_camera_session_generation():
    runner = TrackerRunner(adapter_factory=make_fake_adapter)
    stale = _SessionRouter(FakeAdapter(), {}).process(_request("a", 1.0))
    runner._source_generations["a"] = 2
    runner._out_q = queue.Queue()
    runner._out_q.put(_WorkerResult(result=stale, generation=1))

    assert runner.drain_results() == []


def test_stop_and_start_are_serialized_across_the_join_window():
    join_entered = threading.Event()
    release_join = threading.Event()

    class BlockingDeadProcess(_DeadProcess):
        def join(self, timeout=None) -> None:
            if timeout == 0:
                return
            join_entered.set()
            release_join.wait(timeout=1.0)

    class NewProcess:
        pid = 202

        def __init__(self) -> None:
            self.alive = False

        def start(self) -> None:
            self.alive = True

        def join(self, timeout=None) -> None:
            self.alive = False

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.alive = False

    class FakeContext:
        def __init__(self) -> None:
            self.created = 0

        def Queue(self, maxsize=0):
            return queue.Queue(maxsize=maxsize)

        def Event(self):
            return threading.Event()

        def Process(self, **kwargs):
            self.created += 1
            return NewProcess()

    runner = TrackerRunner(adapter_factory=make_fake_adapter)
    context = FakeContext()
    runner._ctx = context
    runner._proc = BlockingDeadProcess()
    runner._in_q = queue.Queue()
    runner._out_q = queue.Queue()
    runner._stop_event = threading.Event()
    runner._running = True

    stop_thread = threading.Thread(target=runner.stop)
    stop_thread.start()
    assert join_entered.wait(timeout=1.0)

    start_entered = threading.Event()

    def start_runner() -> None:
        start_entered.set()
        runner.start()

    start_thread = threading.Thread(target=start_runner)
    start_thread.start()
    assert start_entered.wait(timeout=1.0)
    time.sleep(0.05)
    serialized = context.created == 0 and start_thread.is_alive()
    release_join.set()
    stop_thread.join(timeout=1.0)
    start_thread.join(timeout=1.0)
    try:
        assert serialized is True
        assert context.created == 1
        assert runner.is_alive() is True
    finally:
        runner.stop()
