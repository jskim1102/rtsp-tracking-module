import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# backend/.env 를 로드
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# 빈 문자열(set-but-empty)도 미설정으로 취급해 안전기본 폴백.
#   os.getenv(k, default) 는 k 가 "" 면 ""(빈값)을 그대로 돌려준다 → CORS_ORIGINS="" → [""]
#   (기본 * 아님 → CORS 차단), MAX_IPCAMS="" → int("") → ValueError(import 크래시).
#   `or` 로 빈값=거짓 → 기본값으로 폴백한다.
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS") or "*"

_raw_max_ipcams = int(os.getenv("MAX_IPCAMS") or "16")
MAX_IPCAMS: int = max(1, min(64, _raw_max_ipcams))

# mediamtx API 주소 — 하드코딩 fallback 을 두지 않는다(빈 문자열 = 미설정).
# Docker compose 가 environment 블록으로 `http://mediamtx:9997` 를 주입하고,
# 로컬 실행 시에는 backend/.env 에 설정한다. 실제로 호출하는 app.mediamtx 가
# 미설정이면 명시 에러를 낸다(import 시점엔 raise 안 함 — 순수 import/테스트 허용).
MEDIAMTX_API: str = os.getenv("MEDIAMTX_API", "")

# mediamtx 인증(#100) — backend user 로 API 호출 시 Basic auth. 비번 비우면 무인증(로컬/테스트 하위호환).
MEDIAMTX_BACKEND_USER: str = os.getenv("MEDIAMTX_BACKEND_USER", "backend")
MEDIAMTX_BACKEND_PASS: str = os.getenv("MEDIAMTX_BACKEND_PASS", "")

# WebRTC 외부접속 광고 호스트(공인 IP). mediamtx.yml 의 webrtcAdditionalHosts 로
# 주입된다. 미설정이면 mediamtx 가 컨테이너 내부 주소만 광고 → 외부에서 영상 안 나옴.
MEDIAMTX_WEBRTC_HOST: str = os.getenv("MEDIAMTX_WEBRTC_HOST", "")

# ── tracker (SAM3 video, 별도 프로세스) ──


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> tuple[int, str | None]:
    """정수 env를 import-safe하게 파싱하고 운영상 안전 범위로 제한한다."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default, None
    try:
        parsed = int(raw)
    except ValueError:
        return default, f"{name}={raw!r} 가 정수가 아니라 기본값 {default} 로 폴백"
    clamped = max(minimum, min(maximum, parsed))
    warning = None
    if clamped != parsed:
        warning = f"{name}={parsed} → {clamped} 로 보정됨 (허용 범위: {minimum}~{maximum})"
    return clamped, warning


def _env_bool(name: str, default: bool) -> tuple[bool, str | None]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default, None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True, None
    if normalized in {"0", "false", "no", "off"}:
        return False, None
    return default, f"{name}={raw!r} 가 bool 이 아니라 기본값 {default} 로 폴백"


TRACKER_DEFAULT: str = os.getenv("TRACKER_DEFAULT") or "sam3"
TRACKER_CONCEPT_PROMPT: str = os.getenv("TRACKER_CONCEPT_PROMPT") or "person"
# facebook/sam3.1은 native multiplex .pt만 제공하며 Transformers 가중치가 없다.
# Sam3VideoModel/Sam3VideoProcessor 경로의 공식 저장소는 facebook/sam3이다.
TRACKER_CHECKPOINT: str = os.getenv("TRACKER_CHECKPOINT") or "facebook/sam3"
TRACKER_DEVICE: str = os.getenv("TRACKER_DEVICE") or ""
TRACK_CONFIRM_FRAMES, _track_confirm_warn = _env_int(
    "TRACK_CONFIRM_FRAMES", 3, minimum=1, maximum=30
)
# 560은 공식 Transformers custom-resolution 예시이며 3090용 480~640 보수 구간 안이다.
TRACKER_INPUT_SIZE, _tracker_size_warn = _env_int(
    "TRACKER_INPUT_SIZE", 560, minimum=480, maximum=640
)
TRACKER_FP16, _tracker_fp16_warn = _env_bool("TRACKER_FP16", True)
# HF tracker_config.num_maskmem에 적용하고 session 저장소도 같은 길이로 prune한다.
TRACKER_MEMORY_BANK_SIZE, _tracker_memory_warn = _env_int(
    "TRACKER_MEMORY_BANK_SIZE", 7, minimum=1, maximum=64
)
TRACKER_RECONDITION_EVERY_NTH_FRAME, _tracker_recondition_warn = _env_int(
    "TRACKER_RECONDITION_EVERY_NTH_FRAME", 16, minimum=0, maximum=10_000
)
TRACKER_MAX_VISION_FEATURES_CACHE_SIZE, _tracker_cache_warn = _env_int(
    "TRACKER_MAX_VISION_FEATURES_CACHE_SIZE", 1, minimum=1, maximum=16
)
TRACKER_PENDING_SOURCES_MAX, _tracker_pending_warn = _env_int(
    "TRACKER_PENDING_SOURCES_MAX", 64, minimum=1, maximum=64
)
# 갤러리는 process-local 상태이므로 기본 저장소는 임시 디렉토리다. 운영자가
# 명시한 디스크를 쓰고 싶을 때만 TRACK_GALLERY_DIR로 바꾼다.
TRACK_GALLERY_DIR: str = os.getenv("TRACK_GALLERY_DIR") or ""
TRACK_GALLERY_MAX_ITEMS, _track_gallery_max_warn = _env_int(
    "TRACK_GALLERY_MAX_ITEMS", 500, minimum=1, maximum=10_000
)


# cadence autotune 내부 정책. 더 이상 .env 수동 knob가 아니다. INFERENCE_INTERVAL은
# telemetry 수렴 전 초기값, MIN/MAX는 폭주 방지 경계다. CAPTURE_INTERVAL은 inference
# cadence와 독립인 detection WS edge polling 주기라 10ms 고정으로 aliasing을 피한다.
MIN_INFERENCE_INTERVAL: float = 0.01
MAX_INFERENCE_INTERVAL: float = 1.0
INFERENCE_INTERVAL: float = 0.033
CAPTURE_INTERVAL: float = 0.01

# 실제 budget은 worker infer_ms EWMA에서 매초 계산한다. MAX_INFER_PER_SEC=52는 더 이상
# 사용자 knob가 아니라 worker 표본이 생기기 전 몇 초만 쓰는 보수적 bootstrap 상한이다.
MAX_INFER_PER_SEC: float = 52.0
# source별 pending + model micro-batch가 burst/drop을 흡수하므로 단일 size-1 queue 시절의
# 15% 여유 대신 5%만 남겨 GPU를 더 채운다.
AUTOTUNE_HEADROOM: float = 0.95
AUTOTUNE_EWMA_ALPHA: float = 0.2
AUTOTUNE_MIN_SAMPLES: int = 5
AUTOTUNE_TARGET_FPS_MAX: float = MAX_INFER_PER_SEC

# custom .pt 모델 디렉토리 — 미설정 시 backend/models(네이티브 dev) = /app/models(컨테이너)
# 로 자동 결정(절대경로, cwd 무관). models_dir 가 import 시 CUSTOM_MODELS_DIR 를 읽으므로
# 반드시 그 전에(여기, config import 시점) setdefault. compose 가 명시하면 그 값이 우선.
os.environ.setdefault(
    "CUSTOM_MODELS_DIR", str(Path(__file__).resolve().parent.parent / "models")
)

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging() -> logging.Logger:
    """애플리케이션 로거 설정"""
    logger = logging.getLogger("rtsp-tracking")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s  %(name)s — %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)

    return logger


logger = setup_logging()

if _raw_max_ipcams != MAX_IPCAMS:
    logger.warning("MAX_IPCAMS=%d → %d 로 보정됨 (허용 범위: 1~64)", _raw_max_ipcams, MAX_IPCAMS)
for _w in (
    _track_confirm_warn,
    _tracker_size_warn,
    _tracker_fp16_warn,
    _tracker_memory_warn,
    _tracker_recondition_warn,
    _tracker_cache_warn,
    _tracker_pending_warn,
    _track_gallery_max_warn,
):
    if _w:
        logger.warning("%s", _w)
