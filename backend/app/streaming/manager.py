"""모든 영상 소스 + 단일 stateful tracker process의 통합 매니저.

- 캡처 스레드는 source별 최신 frame을 tracker runner에 제출
- tracker process 하나가 camera별 Session을 순차 갱신 (cross-camera batch 금지)
- runner 결과는 dispatch 스레드가 source_id 별로 캐싱
- 캡처 스레드는 캐시에서 자기 source_id 의 최신 결과만 조회

rtsp-tracking: JPEG/get_frame 경로 제거(영상은 mediamtx WHEP). tracks_to_json
은 `frame:{w,h}`(SEAM) 동봉 — frontend TrackingOverlay 좌표 스케일용.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import cv2

from app.config import (
    AUTOTUNE_EWMA_ALPHA,
    AUTOTUNE_HEADROOM,
    AUTOTUNE_MIN_SAMPLES,
    AUTOTUNE_TARGET_FPS_MAX,
    INFERENCE_INTERVAL,
    MAX_INFER_PER_SEC,
    MAX_INFERENCE_INTERVAL,
    MIN_INFERENCE_INTERVAL,
    TRACKER_INPUT_SIZE,
)
from app.inference import FrameRequest, InferenceResult
from app.streaming.capture import SourceType, VideoCaptureThread
from app.tracking.runner import TrackerRunner
from app.tracking.gallery import TrackGallery

logger = logging.getLogger("rtsp-tracking.streaming.manager")

INFERENCE_IMGSZ = TRACKER_INPUT_SIZE
GPU_UTIL_TARGET_DEFAULT = 0.85
GPU_UTIL_TARGET_MIN = 0.1
GPU_UTIL_SCALE_MIN = 0.1
GPU_UTIL_DEADBAND = 0.03
GPU_UTIL_RECOVERY_STEP = 0.05


@dataclass
class InferenceTelemetry:
    """worker service/idle 시간의 bounded EWMA 집계."""

    alpha: float = AUTOTUNE_EWMA_ALPHA
    min_samples: int = AUTOTUNE_MIN_SAMPLES
    infer_ms_ewma: float = 0.0
    duty_ewma: float = 0.0
    samples: int = 0

    def observe(self, result: InferenceResult) -> bool:
        infer_ms = float(getattr(result, "infer_ms", 0.0))
        idle_ms = float(getattr(result, "idle_ms", 0.0))
        cycle_ms = infer_ms + idle_ms
        # worker가 0으로 표시한 모델별 첫 추론(warm-up)과 비정상 표본은 버린다.
        if infer_ms <= 0.0 or cycle_ms <= 0.0:
            return False
        duty = infer_ms / cycle_ms
        if self.samples == 0:
            self.infer_ms_ewma = infer_ms
            self.duty_ewma = duty
        else:
            self.infer_ms_ewma += self.alpha * (infer_ms - self.infer_ms_ewma)
            self.duty_ewma += self.alpha * (duty - self.duty_ewma)
        self.samples += 1
        return True

    @property
    def ready(self) -> bool:
        return self.samples >= self.min_samples and self.infer_ms_ewma > 0.0

    @property
    def capacity_fps(self) -> float:
        return 1000.0 / self.infer_ms_ewma if self.infer_ms_ewma > 0.0 else 0.0


@dataclass
class SourceTelemetry:
    """source별 bounded scalar 계측. source 제거 때 dict에서도 함께 제거한다."""

    submitted: int = 0
    received: int = 0
    last_result_ts: float = 0.0
    infer_ms_ewma: float = 0.0
    infer_samples: int = 0

    def observe(self, result: InferenceResult, *, alpha: float) -> None:
        self.received += 1
        self.last_result_ts = result.timestamp
        infer_ms = float(result.infer_ms)
        if infer_ms <= 0.0:
            return
        if self.infer_samples == 0:
            self.infer_ms_ewma = infer_ms
        else:
            self.infer_ms_ewma += alpha * (infer_ms - self.infer_ms_ewma)
        self.infer_samples += 1


def _allocate_fps(
    wants: dict[str, float],
    *,
    budget_fps: float,
    floor_fps: float,
    cost_weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """가중 총량을 budget 아래로 유지하며 want 비율과 가능한 floor를 보존한다.

    n*floor > budget이면 두 계약은 동시에 성립하지 않는다. GPU service 총량 제한을
    우선하고 budget/n을 effective floor로 쓴다. cost_weights는 global infer_ms 대비
    source service cost 비율이다.
    """
    if not wants:
        return {}
    budget = max(0.0, float(budget_fps))
    weights = {
        sid: max(1e-9, float((cost_weights or {}).get(sid, 1.0)))
        for sid in wants
    }
    weight_sum = sum(weights.values())
    effective_floor = min(
        max(0.0, float(floor_fps)),
        budget / weight_sum if weight_sum > 0.0 else 0.0,
    )
    targets = {sid: max(effective_floor, float(want)) for sid, want in wants.items()}
    if sum(targets[sid] * weights[sid] for sid in targets) <= budget:
        return targets

    remaining = max(0.0, budget - effective_floor * weight_sum)
    extra_demand = sum(
        max(0.0, targets[sid] - effective_floor) * weights[sid]
        for sid in targets
    )
    if extra_demand <= 0.0:
        return {sid: effective_floor for sid in targets}
    return {
        sid: effective_floor
        + remaining * max(0.0, target - effective_floor) / extra_demand
        for sid, target in targets.items()
    }


def _gpu_util_scale_step(
    current: float,
    *,
    target: float,
    duty: float,
) -> float:
    """GPU duty ceiling controller 한 tick.

    초과는 즉시 비율 감속하고, 여유는 5%p씩 복구하며, ±3% deadband에서
    cadence 진동을 막는다. 100% 목표는 기존 최대성능 경로와 정확히 동일하다.
    """
    target = max(GPU_UTIL_TARGET_MIN, min(1.0, float(target)))
    if target >= 1.0:
        return 1.0
    current = max(GPU_UTIL_SCALE_MIN, min(1.0, float(current)))
    duty = max(0.0, min(1.0, float(duty)))
    if duty > target + GPU_UTIL_DEADBAND:
        proportional = current * target / max(duty, 1e-9)
        return max(GPU_UTIL_SCALE_MIN, min(current, proportional))
    if duty < target - GPU_UTIL_DEADBAND:
        return min(1.0, current + GPU_UTIL_RECOVERY_STEP)
    return current


def _resize_for_inference(frame: np.ndarray, imgsz: int = INFERENCE_IMGSZ) -> np.ndarray:
    """tracker 제출 전 고정 long-side로 축소한다(종횡비 보존, 업스케일 금지).

    좌표 SEAM(frame_w/h=request frame)이 실제 축소 치수를 따라가므로 frontend가
    video.videoWidth/frame.w 비율로 원 영상에 정합할 수 있다.
    """
    h, w = frame.shape[:2]
    scale = imgsz / max(h, w)
    if scale >= 1.0:
        return frame
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class StreamManager:
    """비디오 소스(웹캠/IP CAM) + 추론 워커 통합 관리.

    싱글톤으로 사용 (`from app.streaming.manager import manager`).
    서버 lifespan 시작/종료에서 `startup()` / `shutdown()` 호출.
    """

    # dispatch 루프의 idle sleep
    _DISPATCH_IDLE_SEC = 0.01

    # H1 tombstone TTL(초): 최근 삭제된 source_id 를 이 시간 동안 기억해 resurrect 를 막는다.
    # delete_ipcam(commit+mediamtx round-trip)~WS start_capture 경합 창(~1 RTT)보다 넉넉히 크면 된다.
    # stream_key 는 재사용되지 않으므로 만료돼 잊혀도 안전(같은 sid 로 새 카메라가 생기지 않음).
    _TOMBSTONE_TTL = 30.0

    def __init__(
        self,
        inference_interval: float = INFERENCE_INTERVAL,
        *,
        headroom: float = AUTOTUNE_HEADROOM,
        target_fps_max: float = AUTOTUNE_TARGET_FPS_MAX,
    ) -> None:
        self._captures: dict[str, VideoCaptureThread] = {}
        self._lock = threading.Lock()
        # telemetry 수렴 전 안전 초기 cadence. 수렴 후에는 worker 실측 용량과 source fps로 갱신.
        self._inference_interval = inference_interval
        self._min_interval = MIN_INFERENCE_INTERVAL
        self._max_interval = MAX_INFERENCE_INTERVAL
        self._headroom = max(0.0, min(1.0, headroom))
        self._target_fps_max = max(0.0, target_fps_max)

        self._worker = TrackerRunner()
        self._gallery = TrackGallery()
        self._latest_results: dict[str, InferenceResult] = {}
        self._results_lock = threading.Lock()
        self._telemetry = InferenceTelemetry()
        self._source_telemetry: dict[str, SourceTelemetry] = {}
        self._telemetry_lock = threading.Lock()
        self._last_drop_count = 0
        self._gpu_util_target = GPU_UTIL_TARGET_DEFAULT
        self._util_scale = 1.0
        self._source_imgsz: dict[str, int] = {}
        self._imgsz_pressure_ticks: dict[str, int] = {}
        self._imgsz_relief_ticks: dict[str, int] = {}
        # MAX interval을 넘긴 source의 transition latch. 매 tick 경고 도배 없이 진입 시 1회만 기록.
        self._slow_interval_warned: set[str] = set()

        # source_id 별 추론 enabled — key 없으면 True 가 기본
        self._per_source_enabled: dict[str, bool] = {}
        self._per_source_lock = threading.Lock()

        # H1: 최근 삭제 source_id → 삭제 monotonic 시각. start_capture 의 create 분기가 참조해
        # 삭제된 카메라 캡처 resurrection(delete↔WS TOCTOU)을 막는다. self._lock 로 보호.
        self._tombstones: dict[str, float] = {}

        # fix19: worker 프로세스 사망 감지 watchdog — 마지막 respawn monotonic 시각(백오프 기준).
        self._last_worker_respawn: float = 0.0
        self._tracker_events: dict[str, list[dict[str, object]]] = {}
        self._tracker_events_lock = threading.Lock()

        # lifecycle edge는 interval을 직접 쓰지 않고 dispatch autotuner의 즉시 tick만 요청한다.
        self._tune_requested = True

        self._dispatch_thread: Optional[threading.Thread] = None
        self._dispatch_running = False

    # ── lifecycle ────────────────────────────────────────────────

    def startup(self) -> None:
        """워커 + dispatch 스레드 기동."""
        self._worker.start()
        self._dispatch_running = True
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="inference-dispatch"
        )
        self._dispatch_thread.start()
        logger.info("StreamManager started (inference worker + dispatch)")

    def shutdown(self) -> None:
        """모든 캡처 + 워커 + dispatch 정리."""
        # 1) 캡처 강제 종료
        with self._lock:
            captures = list(self._captures.values())
            self._captures.clear()
        for cap in captures:
            cap.force_stop()
            logger.info("Capture %s 강제 종료", cap.source_id)

        # 2) dispatch 종료
        self._dispatch_running = False
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=2)

        # 3) worker 종료
        self._worker.stop()
        logger.info("StreamManager shutdown 완료")

    # ── 캡처 시작/종료 (라우터에서 호출) ─────────────────────────

    def _is_tombstoned_locked(self, source_id: str) -> bool:
        """self._lock 보유 상태에서 호출 — source_id 가 TTL 내 삭제됐으면 True. 조회된 sid 가 만료됐으면
        청소한다(단, 삭제 후 재조회 안 되는 sid 는 이 lazy 경로로 안 지워짐 → 전역 청소는
        _prune_tombstones_locked 가 remove_capture 에서 담당)."""
        ts = self._tombstones.get(source_id)
        if ts is None:
            return False
        if time.monotonic() - ts >= self._TOMBSTONE_TTL:
            del self._tombstones[source_id]
            return False
        return True

    def _prune_tombstones_locked(self) -> None:
        """self._lock 하 — 만료된 tombstone 전부 청소. stream_key 는 재사용 안 되고 삭제된 sid 는
        start_capture 로 재조회되지 않아 lazy 청소가 안 먹으므로, 삭제 때마다 여기서 전역 청소해
        무한증식을 막는다(dict 는 최근 TTL 창의 삭제분만 유지)."""
        now = time.monotonic()
        for sid in [s for s, ts in self._tombstones.items() if now - ts >= self._TOMBSTONE_TTL]:
            del self._tombstones[sid]

    def start_capture(self, source_id: str, source: SourceType) -> bool:
        """source_id 캡처 시작(**없을 때만** 생성) + 뷰어 1명 증가. open 성공 시 True.

        create-only-if-absent: 엔트리가 이미 있으면 넘어온 `source` 를 **무시하고 기존 엔트리를 재사용**한다.
        URL 권위는 오직 update_ipcam→replace_source 로 단일화되어 있고, WS 핸들러가 넘기는 `source` 는
        `await accept()` 前에 읽혀 stale 일 수 있다(URL 편집 in-flight). 옛 "source 상이 시 교체" 방어망은
        stale WS 가 정확한 new_url 캡처를 옛 url 로 되돌려 모든 뷰어에게 딴 카메라를 지속 노출하는
        F1-무력화 SHOULD 였다(독립 게이트 확인) → 제거. 신규 url 반영은 replace_source 가 담당한다.

        H1: create 분기에서 tombstone(최근 삭제 sid)을 확인해 삭제된 카메라 resurrection 을 거부한다 —
        remove_capture 의 tombstone 등록과 self._lock 로 직렬화되어 delete↔WS 경합 창을 닫는다.

        증가(acquire_viewer)는 get/create 와 함께 **manager 락 안에서** 원자 수행 → replace_source 의
        adopt-read/swap 과 직렬화(증가side BLOCKER 제거, 감소side release_viewer 대칭). open(블로킹)은 락 밖.
        open 후 엔트리가 바뀌었으면(orphan: 기동 중 remove/replace) 또는 open 실패면, 증가분을 '현재
        엔트리'에서 release_viewer 로 되돌리고 방금 만든 캡처를 정리한 뒤 False.

        H2 invariant(중요): 이 메서드는 async ipcam_ws 의 **이벤트루프 스레드에서 동기 실행**된다(내부에
        await 없음). 그래서 두 WS 가 같은 sid 의 first-open 을 동시 경합할 수 없다(2nd viewer 가 open
        미확인 _running=True 를 보는 SHOULD-1/2 는 today unreachable). 이 메서드를 async/threadpool 로
        옮기면 그 경합이 재개되므로 옮기지 말 것 — 옮긴다면 acquire/open 을 재설계해야 한다.
        """
        with self._lock:
            cap = self._captures.get(source_id)
            if cap is None:
                if self._is_tombstoned_locked(source_id):
                    # H1: 최근 삭제된 카메라 — resurrect 거부(삭제된 RTSP 를 디코드하지 않는다). WS 는 닫힌다.
                    return False
                cap = VideoCaptureThread(
                    source_id=source_id,
                    source=source,
                    frame_callback=self._submit_frame,
                    # dispatch가 첫 bootstrap tick을 쓰기 전에도 다캠 30fps 폭주가 없도록
                    # 보수적으로 시작한다. 첫 frame은 deadline=0이라 즉시 제출되고, tuner가
                    # 곧 speed-up하면 setter가 다음 deadline을 앞으로 당긴다.
                    inference_interval=self._max_interval,
                )
                self._captures[source_id] = cap
            cap.acquire_viewer()   # 증가를 get/create 와 원자적으로 (락 하)
            target = cap
        ok = target._ensure_running()
        # orphan(기동 중 remove/replace) 또는 open 실패 → 증가분을 '현재 엔트리'에서 되돌린다.
        rollback_thread = None
        with self._lock:
            current = self._captures.get(source_id)
            orphaned = current is not target
            if orphaned:
                # 우리 증가는 (adopt 로) 현재 엔트리에 접혀 있으니 거기서 감소 (감소side 대칭).
                # NOTE: current 가 adopt-후손이 아닌 경우(remove_capture 로 dict 가 비워진 뒤 다른 WS 가
                # fresh 엔트리 생성)엔 남의 뷰어를 감소시킬 수 있으나, remove_capture 는 delete 전용이고
                # delete sweep(ipcam.py)이 그 fresh 엔트리도 force_stop 하므로 오늘은 무해(생존 피해자 없음).
                # 향후 non-delete remove_capture 호출자(idle-eviction 등)를 추가하면 이 가정 재검토 필요.
                if current is not None:
                    rollback_thread = current.release_viewer()
            elif not ok:
                # 현재 엔트리인데 open 실패 → 우리 증가 되돌림 (WS 는 False 에 stop_capture 를 안 부름).
                rollback_thread = target.release_viewer()
        if orphaned or not ok:
            if orphaned:
                target.force_stop()   # 밀려난 캡처 정리 (untracked 스레드 방지)
            if rollback_thread is not None:
                rollback_thread.join(timeout=5)
            self._recompute_cadence()
            return False
        self._recompute_cadence()
        return True

    def stop_capture(self, source_id: str) -> None:
        # 감소는 manager 락 안에서 '현재 엔트리'에 수행 → replace_source 의 swap 과 직렬화되어
        # 감소가 옛 캡처로 오배정되지 않는다(misdirected-decrement/ref 팽창 방지). join 은 락 밖.
        thread = None
        with self._lock:
            cap = self._captures.get(source_id)
            if cap is not None:
                thread = cap.release_viewer()
                became_idle = cap.ref_count == 0
            else:
                became_idle = False
        if cap is None:
            return
        if thread is not None:
            thread.join(timeout=5)
        if became_idle:
            with self._results_lock:
                self._latest_results.pop(source_id, None)
            gallery = getattr(self, "_gallery", None)
            if gallery is not None:
                gallery.mark_camera_inactive(source_id)
        self._recompute_cadence()

    def replace_source(self, source_id: str, new_source: SourceType) -> bool:
        """URL 편집 시 캡처를 새 source 로 원자 교체하거나, 캡처가 없으면(idle 카메라) 새 source 를
        **URL 권위로 기록**한다(idle ref-0 엔트리 신설). 변경했으면 True.

        옛 캡처가 있으면 새 캡처로 락 안에서 원자 교체 + 뷰어 수(ref_count) 이관(adopt_viewers) 후 옛
        캡처 force-stop — 교체 도중 WS 가 연결/해제돼도 decrement 유실·ref 팽창이 없다(부재 창 없음).
        옛 캡처가 없으면(old=None: 뷰어 없는 idle 카메라) no-op 하지 않고 ref-0 idle 엔트리를 신설한다 —
        이렇게 URL 권위를 기록해야, 편집 중 stale 값을 읽은 첫 연결이 create-only-if-absent 로 이
        엔트리(=새 url)를 재사용해 옛 url 로 persistent stale 엔트리를 만들지 않는다(SHOULD-4,
        safety-critical: 파생 모듈로 전파되는 정본. round-2 의 old=None no-op 은 이 self-heal 을
        없앴었다). idle 엔트리는 ref-0·미기동이라 cadence·리소스 영향 없고, 첫 연결 시 open 되거나
        재편집/삭제로 교체·제거된다.
        NOTE(accept): 옛 스레드가 force_stop 직전에 워커에 제출한 프레임이 dispatch 루프에서 pop 이후
        다시 캐시될 수 있어 새 카메라 WS가 ~1 추론간격 동안 옛 마스크를 볼 수 있다 — transient 수용.
        """
        with self._lock:
            old = self._captures.get(source_id)
            if old is not None and old.source == new_source:
                return False  # 변화 없음
            if old is None and self._is_tombstoned_locked(source_id):
                # H1 대칭: 삭제된(tombstoned) 카메라엔 idle 엔트리도 신설하지 않는다 (update↔delete
                # 경합 → 삭제 카메라의 dormant resurrection 방지). 살아있는 카메라의 실 교체(old not
                # None)는 삭제가 아니므로 tombstone 무관.
                return False
            new = VideoCaptureThread(
                source_id=source_id,
                source=new_source,
                frame_callback=self._submit_frame,
                inference_interval=self._max_interval,
            )
            if old is not None:
                new.adopt_viewers(old.ref_count)   # 뷰어 수 원자 이관 (idle 신설이면 ref-0 유지)
            self._captures[source_id] = new    # 엔트리 원자 교체/신설 (부재 창 없음)
        if old is not None:
            old.force_stop()
            with self._results_lock:
                self._latest_results.pop(source_id, None)  # 옛 카메라 검출 잔상 제거
        new._ensure_running()                  # ref>0 이면 1회 open, ref-0(idle 신설)이면 미기동
        # 기동 중 remove/재replace 로 엔트리가 바뀌었으면 방금 만든 new 는 orphan → 정리.
        orphan = None
        with self._lock:
            if self._captures.get(source_id) is not new:
                orphan = new
        if orphan is not None:
            orphan.force_stop()
        self._recompute_cadence()
        return True

    def remove_capture(self, source_id: str) -> None:
        """카메라 삭제 시 캡처를 ref_count 무시하고 완전 제거한다.

        stop_capture 는 ref-count 기반이라 viewer 2+ 면 삭제해도 스레드가 살아 삭제된 카메라를
        계속 디코드한다(F2). 삭제는 force-stop + dict 제거 + 검출/per-source 캐시 정리로 즉시 끝낸다.
        H1: pop 과 함께 tombstone 을 등록해, 아직 start_capture 를 안 부른 in-flight WS 가 삭제된
        카메라를 resurrect 하지 못하게 한다(start_capture create 분기가 self._lock 하에서 확인).
        """
        with self._lock:
            cap = self._captures.pop(source_id, None)
            self._tombstones[source_id] = time.monotonic()  # H1: resurrect 방지 (pop 과 원자)
            self._prune_tombstones_locked()  # 만료분 전역 청소 (무한증식 방지)
        if cap is not None:
            cap.force_stop()
        worker = getattr(self, "_worker", None)
        remove_tracker_source = getattr(worker, "remove_source", None)
        if callable(remove_tracker_source) and not remove_tracker_source(source_id):
            logger.error("Tracker session teardown enqueue failed: camera=%s", source_id)
        with self._results_lock:
            self._latest_results.pop(source_id, None)
        with self._per_source_lock:
            self._per_source_enabled.pop(source_id, None)
        with self._telemetry_lock:
            self._source_telemetry.pop(source_id, None)
            getattr(self, "_source_imgsz", {}).pop(source_id, None)
            getattr(self, "_imgsz_pressure_ticks", {}).pop(source_id, None)
            getattr(self, "_imgsz_relief_ticks", {}).pop(source_id, None)
            warned = getattr(self, "_slow_interval_warned", None)
            if warned is not None:
                warned.discard(source_id)
        self._ensure_tracker_event_state()
        with self._tracker_events_lock:
            self._tracker_events.pop(source_id, None)
        gallery = getattr(self, "_gallery", None)
        if gallery is not None:
            gallery.mark_camera_inactive(source_id)
        self._recompute_cadence()

    def _ensure_tracker_event_state(self) -> None:
        """이전 object.__new__ fixture/runtime state를 새 event queue로 안전 승격."""
        if not hasattr(self, "_tracker_events_lock"):
            self._tracker_events_lock = threading.Lock()
        if not hasattr(self, "_tracker_events"):
            self._tracker_events = {}

    def _recompute_cadence(self) -> None:
        """lifecycle edge에서 autotuner의 다음 tick을 즉시 요청한다.

        interval 쓰기는 _run_autotune 한 곳만 담당해 viewer 접속/해제가 직전 튜너 값을
        고정 N/MAX_INFER_PER_SEC 값으로 덮어쓰지 않는다.
        """
        self._tune_requested = True

    def _apply_bootstrap_cadence(
        self,
        active: list[tuple[str, VideoCaptureThread]],
    ) -> bool:
        """worker 표본 전에는 과거 보수적 상한으로 폐쇄형 총 제출률을 제한한다."""
        if not active:
            return False
        budget = (
            MAX_INFER_PER_SEC
            * self._headroom
            * max(GPU_UTIL_SCALE_MIN, min(1.0, getattr(self, "_util_scale", 1.0)))
        )
        if budget <= 0.0:
            interval = self._max_interval
        else:
            interval = max(self._inference_interval, len(active) / budget)
        interval = max(self._min_interval, interval)
        for _, cap in active:
            cap.set_inference_interval(interval)
        logger.info(
            "cadence bootstrap: active=%d budget=%.1f/s assigned=%.1f/s",
            len(active),
            budget,
            len(active) / interval,
        )
        return True

    def _run_autotune(self) -> bool:
        """worker 실측 용량과 source fps snapshot으로 capture cadence를 갱신."""
        with self._lock:
            active = [(sid, cap) for sid, cap in self._captures.items() if cap.is_running]
        if not active:
            return False

        with self._per_source_lock:
            enabled = dict(self._per_source_enabled)
        active = [
            (sid, cap)
            for sid, cap in active
            if enabled.get(sid, True)
        ]
        if not active:
            return False

        pool_duty_getter = getattr(self._worker, "get_pool_duty", None)
        pool_duty = (
            max(0.0, min(1.0, float(pool_duty_getter())))
            if callable(pool_duty_getter)
            else None
        )
        with self._telemetry_lock:
            if pool_duty is not None:
                # result 단위 duty가 아니라 모델 lane들의 pool-wide busy union을 제어 신호로 쓴다.
                self._telemetry.duty_ewma = pool_duty
            telemetry_ready = self._telemetry.ready
            capacity = self._telemetry.capacity_fps
            duty = self._telemetry.duty_ewma
            global_infer_ms = self._telemetry.infer_ms_ewma
            target = max(
                GPU_UTIL_TARGET_MIN,
                min(
                    1.0,
                    getattr(
                        self,
                        "_gpu_util_target",
                        GPU_UTIL_TARGET_DEFAULT,
                    ),
                ),
            )
            util_scale = _gpu_util_scale_step(
                getattr(self, "_util_scale", 1.0),
                target=target,
                duty=duty,
            )
            self._gpu_util_target = target
            self._util_scale = util_scale
            source_costs = {
                sid: (metrics.infer_ms_ewma, metrics.infer_samples)
                for sid, metrics in self._source_telemetry.items()
            }
        if not telemetry_ready:
            return self._apply_bootstrap_cadence(active)

        wants: dict[str, float] = {}
        cost_weights: dict[str, float] = {}
        unknown_sources = 0
        for sid, cap in active:
            if cap.source_fps <= 0.0:
                # isOpened=True지만 frame advance가 없는 stall도 함대 전체를 막지 않는다.
                unknown_sources += 1
                wants[sid] = 0.0
            else:
                wants[sid] = min(cap.source_fps, self._target_fps_max)

            measured_cost, cost_samples = source_costs.get(sid, (0.0, 0))
            if cost_samples > 0 and measured_cost > 0.0:
                source_infer_ms = measured_cost
            else:
                source_infer_ms = global_infer_ms
            cost_weights[sid] = source_infer_ms / global_infer_ms

        budget = capacity * self._headroom
        allocated = _allocate_fps(
            wants,
            budget_fps=budget,
            floor_fps=0.0,
            cost_weights=cost_weights,
        )
        stats_getter = getattr(self._worker, "get_queue_stats", None)
        stats = stats_getter() if callable(stats_getter) else {"dropped": 0}
        drops = int(stats.get("dropped", 0))
        weighted_demand = sum(wants[sid] * cost_weights[sid] for sid in wants)

        effective_floor = 0.0
        overruns: dict[str, float] = {}
        for sid, cap in active:
            base_fps = allocated.get(sid, 0.0)
            with self._telemetry_lock:
                self._source_imgsz[sid] = INFERENCE_IMGSZ
                self._imgsz_pressure_ticks[sid] = 0
                self._imgsz_relief_ticks[sid] = 0
            # Stateful tracker memory는 session 수명 중 해상도 변경을 허용하지 않는다.
            # 과부하는 adaptive resolution 대신 cadence만 낮춰 흡수한다(D3).
            fps = base_fps * util_scale
            if fps <= 0.0:
                interval = self._max_interval
            else:
                interval = 1.0 / fps
            # n*FLOOR가 실측 budget보다 큰 경우 두 계약은 물리적으로 양립 불가다.
            # GPU service 총량 계약을 우선해 MAX보다 느린 cadence를 허용한다.
            assigned_interval = max(self._min_interval, interval)
            cap.set_inference_interval(assigned_interval)
            if assigned_interval > self._max_interval:
                overruns[sid] = assigned_interval

        # active snapshot 뒤 삭제된 source를 경고 latch에 되살리지 않는다. remove_capture가
        # telemetry lock에서 discard하므로 두 경합 순서 모두 최종 상태가 bounded다.
        with self._lock:
            live_sources = set(self._captures)
        current_overruns = set(overruns).intersection(live_sources)
        with self._telemetry_lock:
            previous_overruns = set(getattr(self, "_slow_interval_warned", set()))
            self._slow_interval_warned = current_overruns
        for sid in sorted(current_overruns - previous_overruns):
            logger.warning(
                "autotune MAX interval 초과: source=%s interval=%.3fs budget=%.3f/s "
                "effective_floor=%.3ffps",
                sid,
                overruns[sid],
                budget,
                effective_floor,
            )

        if drops > self._last_drop_count:
            logger.warning(
                "autotune queue drop 증가: +%d (capacity=%.1f/s, budget=%.1f/s)",
                drops - self._last_drop_count,
                capacity,
                budget,
            )
        self._last_drop_count = drops
        logger.info(
            "cadence autotune: active=%d unknown=%d capacity=%.1f/s duty=%.2f "
            "target=%.2f scale=%.2f budget=%.1f/s demand=%.1f/s",
            len(active),
            unknown_sources,
            capacity,
            duty,
            target,
            util_scale,
            budget,
            weighted_demand,
        )
        return True

    # (get_frame 제거 — JPEG 버퍼 없음. 영상은 mediamtx WHEP, detection 은 좌표 WS)
    # (get_capture_stats 제거 — stats endpoint 가 mediamtx readers 기반으로 바뀌어 死코드였고,
    #  그게 읽던 _inference_ts deque 가 dispatch 루프에서 영구 append 돼 source 당 메모리 누수였다.)

    # ── 캡처 ↔ 워커 bridge ───────────────────────────────────────

    def _submit_frame(self, source_id: str, frame: np.ndarray, captured_at: Optional[float] = None) -> None:
        """캡처 스레드 callback — global AND per-source 둘 다 ON 인 경우만 워커에 제출.

        prompt와 확인 임계값은 tracker 환경설정이 소유하며 요청별 override는 없다.
        """
        status = self._worker.get_status()
        if not status.get("enabled", True):
            return
        if not self.is_source_inference_enabled(source_id):
            return
        # tracker session의 입력 해상도는 수명 동안 고정한다(D3). 과부하 제어는 cadence
        # autotune만 사용한다. WS frame.w/h는 실제 request 치수라 좌표 SEAM이 유지된다.
        imgsz = INFERENCE_IMGSZ
        frame = _resize_for_inference(frame, imgsz=imgsz)

        accepted = self._worker.submit(
            FrameRequest(
                source_id=source_id,
                frame=frame,
                timestamp=captured_at if captured_at is not None else time.time(),
                model_names=None,
                imgsz=imgsz,
            )
        )
        drain_resolution_changes = getattr(
            self._worker, "drain_resolution_changes", None
        )
        if callable(drain_resolution_changes):
            for camera_id in drain_resolution_changes():
                self._emit_tracker_reset(
                    "resolution_changed", camera_ids=[camera_id]
                )
        if accepted:
            with self._telemetry_lock:
                # remove_capture와 원자적인 membership fence. 삭제 직후 in-flight callback이
                # source telemetry를 되살리지 못한다(락 순서 telemetry→manager, 역순 중첩 없음).
                with self._lock:
                    if source_id in self._captures:
                        metrics = self._source_telemetry.setdefault(
                            source_id, SourceTelemetry()
                        )
                        metrics.submitted += 1

    def _get_latest_result(self, source_id: str) -> Optional[InferenceResult]:
        """source_id 의 최신 추론 결과 (없으면 None)."""
        with self._results_lock:
            return self._latest_results.get(source_id)

    # WS 핸들러용 public alias — frontend overlay 가 tracks JSON 으로 받는다.
    def get_source_latest_tracks(self, source_id: str) -> Optional[InferenceResult]:
        return self._get_latest_result(source_id)

    def _maybe_respawn_worker(self) -> None:
        """worker 프로세스가 죽었으면 respawn 한다 (fix19). shared state(_state)는 재사용되고 큐(in_q/out_q)는 재생성되며(fix21)
        worker_main 이 새 프로세스에서 모델을 lazy 재로드한다. respawn 폭주를 막으려 직전
        respawn 후 최소 10s 백오프를 둔다. _dispatch_loop(데몬 스레드, 이벤트루프 아님 →
        H2 무관)이 ~5s 주기로 호출한다."""
        if self._worker.is_alive():
            return
        nowm = time.monotonic()
        if nowm - self._last_worker_respawn < 10.0:
            return  # 백오프 — 직전 respawn 후 10s 미경과
        logger.error("Inference worker 프로세스 사망 감지 — respawn 시도")
        # fix22: start() 前에 갱신 — start() 가 raise(persistent OOM spawn 실패, fix21 armor 가 캐치)해도
        # 백오프가 유지돼 5s 마다 재시도(spawn-storm)하지 않는다. attempt-based(성공기반 아님).
        self._last_worker_respawn = nowm
        # Each attempt creates a distinct tracker generation even if it dies before
        # the next 5-second alive observation. Announce every attempted generation.
        self._emit_tracker_reset("worker_respawn")
        self._worker.start()

    def _emit_tracker_reset(
        self,
        reason: str,
        *,
        camera_ids: list[str] | None = None,
    ) -> None:
        """Queue an explicit ID-discontinuity event for affected cameras."""
        if reason not in {
            "worker_respawn",
            "tracker_switch",
            "resolution_changed",
            "manual_reset",
        }:
            raise ValueError(f"invalid tracker_reset reason: {reason}")
        if camera_ids is None:
            with self._lock:
                affected_camera_ids = list(self._captures)
        else:
            with self._lock:
                affected_camera_ids = [
                    camera_id
                    for camera_id in camera_ids
                    if camera_id in self._captures
                ]
        with self._results_lock:
            for camera_id in affected_camera_ids:
                self._latest_results.pop(camera_id, None)
        gallery = getattr(self, "_gallery", None)
        if gallery is not None:
            if reason in {"tracker_switch", "manual_reset"}:
                gallery.reset()
            else:
                gallery.clear_cameras(affected_camera_ids)
        self._ensure_tracker_event_state()
        with self._tracker_events_lock:
            for camera_id in affected_camera_ids:
                self._tracker_events[camera_id] = [
                    {
                        "type": "tracker_reset",
                        "camera_id": camera_id,
                        "reason": reason,
                    }
                ]

    def drain_tracker_events(self, source_id: str) -> list[dict[str, object]]:
        """phase4 WS bridge가 소비할 camera별 reset 이벤트 큐."""
        self._ensure_tracker_event_state()
        with self._tracker_events_lock:
            events = self._tracker_events.pop(source_id, [])
        return events

    def _record_results(self, results: list[InferenceResult]) -> None:
        """drain 결과를 전역 계측과 아직 추적 중인 source 캐시에 기록한다.

        worker 전역 service-time은 삭제 직전 결과도 유효하지만, source별 상태와 마스크는
        remove_capture 뒤 되살아나면 안 된다. 각 cache lock 안에서 manager membership을
        재확인해 remove와 원자적으로 직렬화한다.
        """
        if not results:
            return

        def generation_is_current(result: InferenceResult) -> bool:
            check = getattr(self._worker, "is_current_generation", None)
            return not callable(check) or bool(
                check(result.source_id, result.tracker_generation)
            )

        with self._telemetry_lock:
            for result in results:
                if not generation_is_current(result):
                    continue
                self._telemetry.observe(result)
                with self._lock:
                    if result.source_id not in self._captures:
                        continue
                    metrics = self._source_telemetry.setdefault(
                        result.source_id, SourceTelemetry()
                    )
                    metrics.observe(result, alpha=self._telemetry.alpha)

        accepted_results: list[InferenceResult] = []
        for result in results:
            if not generation_is_current(result):
                continue
            with self._results_lock:
                with self._lock:
                    if result.source_id not in self._captures:
                        continue
                    current = self._latest_results.get(result.source_id)
                    if current is not None and (
                        result.timestamp,
                        getattr(result, "request_id", 0),
                    ) < (
                        current.timestamp,
                        getattr(current, "request_id", 0),
                    ):
                        continue
                    self._latest_results[result.source_id] = result
                    accepted_results.append(result)

        # JPEG encode/write는 result/cache manager lock 밖에서 수행한다. 느린 디스크가
        # dispatch와 latest-wins drain을 막아 새-track 결과 유실을 증폭하지 않게 한다.
        gallery = getattr(self, "_gallery", None)
        if gallery is None:
            return
        for result in accepted_results:
            if not generation_is_current(result):
                with self._results_lock:
                    if self._latest_results.get(result.source_id) is result:
                        self._latest_results.pop(result.source_id, None)
                continue
            observation = gallery.observe_with_ack(result)
            with self._lock:
                if result.source_id not in self._captures:
                    gallery.discard_tracks(
                        result.source_id,
                        observation.created_track_ids,
                    )
                    continue

                acknowledge = getattr(self._worker, "acknowledge_tracks", None)
                if observation.acknowledged_track_ids and callable(acknowledge):
                    acknowledged = acknowledge(
                        result.source_id,
                        observation.acknowledged_track_ids,
                        generation=result.tracker_generation,
                    )
                    if not acknowledged:
                        logger.warning(
                            "Track gallery acknowledgement will retry: camera=%s ids=%s",
                            result.source_id,
                            observation.acknowledged_track_ids,
                        )
                if observation.events:
                    self._ensure_tracker_event_state()
                    with self._tracker_events_lock:
                        self._tracker_events.setdefault(result.source_id, []).extend(
                            observation.events
                        )

    def _dispatch_loop(self) -> None:
        """worker.out_q → _latest_results 캐시 + worker 사망 watchdog(fix19). 별도 스레드.

        fix21(G1): while-body 전체를 try/except 로 감싼다 — worker 가 out_q.put 도중 killed(OOM,
        watchdog 의 표적 시나리오)되면 drain_results 가 queue.Empty 가 아니라 EOFError/OSError/
        UnpicklingError 를 던질 수 있고, 그러면 이 스레드가 죽어 결과캐싱+watchdog 이 함께 죽는다
        (watchdog 이 감시하던 바로 그 죽음에 자신이 피살). 어떤 예외든 살아남아야 한다(watchdog 계약).
        """
        next_worker_check = 0.0
        next_tune_ts = 0.0
        while self._dispatch_running:
            try:
                # worker 사망 감지 — drain 정상경로와 독립, ~5s 주기 게이트로 저부하.
                nowm = time.monotonic()
                if nowm >= next_worker_check:
                    next_worker_check = nowm + 5.0
                    self._maybe_respawn_worker()

                results = self._worker.drain_results()
                if results:
                    self._record_results(results)

                # 새 스레드 없이 기존 dispatch loop에서 1s 주기 + lifecycle 즉시 trigger.
                nowm = time.monotonic()
                if self._tune_requested or nowm >= next_tune_ts:
                    self._tune_requested = False  # run 중 새 요청은 다시 True로 남도록 먼저 clear.
                    next_tune_ts = nowm + 1.0
                    self._run_autotune()
                if not results:
                    time.sleep(self._DISPATCH_IDLE_SEC)
            except Exception:
                logger.exception("Dispatch loop 예외 — 복구 후 계속")
                time.sleep(1.0)
                continue
        logger.info("Dispatch loop 종료")

    # ── 추론 제어 (FastAPI 라우터에서 호출) ──────────────────────

    def get_inference_config(self) -> dict:
        status = dict(self._worker.get_status())
        duty_getter = getattr(self._worker, "get_pool_duty", None)
        if callable(duty_getter):
            duty = max(0.0, min(1.0, float(duty_getter())))
        else:
            with self._telemetry_lock:
                duty = max(0.0, min(1.0, self._telemetry.duty_ewma))
        with self._telemetry_lock:
            target = max(
                GPU_UTIL_TARGET_MIN,
                min(
                    1.0,
                    getattr(
                        self,
                        "_gpu_util_target",
                        GPU_UTIL_TARGET_DEFAULT,
                    ),
                ),
            )
        status.update(
            gpu_util_target=target,
            gpu_util_duty=duty,
        )
        return status

    def set_gpu_util_target(self, target: float) -> None:
        """전역 GPU duty ceiling 설정(10~100%)."""
        clamped = max(GPU_UTIL_TARGET_MIN, min(1.0, float(target)))
        with self._telemetry_lock:
            self._gpu_util_target = clamped
            if clamped >= 1.0:
                self._util_scale = 1.0
        self._tune_requested = True
        logger.info("GPU utilization target=%.0f%%", clamped * 100.0)

    def set_inference_enabled(self, enabled: bool) -> None:
        self._worker.set_enabled(enabled)
        if not enabled:
            # OFF 시 캐시 비워서 raw 프레임으로 회귀
            with self._results_lock:
                self._latest_results.clear()
            gallery = getattr(self, "_gallery", None)
            if gallery is not None:
                with self._lock:
                    camera_ids = list(self._captures)
                for camera_id in camera_ids:
                    gallery.mark_camera_inactive(camera_id)
        self._recompute_cadence()

    def set_inference_model(self, model_name: str) -> None:
        if self._worker.get_status().get("model") == model_name:
            return
        self._worker.set_model(model_name)
        self._emit_tracker_reset("tracker_switch")
        # global 모델 비용이 달라지므로 이전 capacity로 새 모델을 과구독하지 않는다.
        # 새 모델의 warm-up 제외 표본이 모일 때까지 bootstrap cadence로 돌아간다.
        with self._telemetry_lock:
            self._telemetry = InferenceTelemetry()
            for metrics in self._source_telemetry.values():
                metrics.infer_ms_ewma = 0.0
                metrics.infer_samples = 0
            for source_id in list(getattr(self, "_source_imgsz", {})):
                self._source_imgsz[source_id] = INFERENCE_IMGSZ
                self._imgsz_pressure_ticks[source_id] = 0
                self._imgsz_relief_ticks[source_id] = 0
        self._recompute_cadence()

    def reset_tracking(self) -> None:
        self._worker.restart()
        self._emit_tracker_reset("manual_reset")

    def get_tracks(self) -> dict[str, object]:
        return self._gallery.snapshot()

    def get_track_thumbnail(
        self,
        track_id: int,
        camera_id: str | None = None,
    ):
        return self._gallery.thumbnail(track_id, camera_id)

    # ── per-source 추론 ON/OFF (각 카메라마다 독립적으로 제어) ──

    def is_source_inference_enabled(self, source_id: str) -> bool:
        """key 없으면 True (기본 ON)."""
        with self._per_source_lock:
            return self._per_source_enabled.get(source_id, True)

    def set_source_inference_enabled(self, source_id: str, enabled: bool) -> None:
        """source_id의 추론 ON/OFF. OFF 시 기존 결과 캐시를 비워 마스크를 즉시 지운다."""
        with self._per_source_lock:
            self._per_source_enabled[source_id] = enabled
        if not enabled:
            with self._results_lock:
                self._latest_results.pop(source_id, None)
            gallery = getattr(self, "_gallery", None)
            if gallery is not None:
                gallery.mark_camera_inactive(source_id)
        self._recompute_cadence()
        logger.info("Per-source inference: %s = %s", source_id, enabled)


def tracks_to_json(result: InferenceResult) -> str:
    """InferenceResult → WS 로 보낼 tracking JSON 문자열.

    box와 mask 좌표는 추론 캡처 frame 픽셀 기준. `frame:{w,h}` 동봉(SEAM) —
    frontend overlay가 video.videoWidth/frame.w 비율로 폴리곤을 변환한다.
    """
    return json.dumps({
        "type": "tracks",
        "timestamp": result.timestamp,
        "frame": {"w": result.frame_w, "h": result.frame_h},
        "items": [
            {
                "track_id": track.track_id,
                "bbox": list(track.bbox),
                "polygon": (
                    [[x, y] for x, y in track.polygon]
                    if track.polygon is not None
                    else None
                ),
                "score": track.score,
                "is_new": track.is_new,
            }
            for track in result.tracks
        ],
    }, ensure_ascii=False)


# 싱글톤 — main.py 에서 startup/shutdown 호출.
# telemetry 수렴 전 안전 초기 cadence. 이후 interval 쓰기는 _run_autotune 단일 경로.
manager = StreamManager(inference_interval=INFERENCE_INTERVAL)
