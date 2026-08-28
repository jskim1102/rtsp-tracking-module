"""YOLO preset 가중치 관리 (preset-only).

custom `.pt` 업로드는 제거됐다 — `.pt`=pickle=코드실행 신뢰경계라, 인증 없는 업로드 +
메인 프로세스 로드는 RCE 였다(codex #1). 이제 ultralytics 가 자동 다운로드하는 preset 5종만
허용(allowlist). 워커·class 메타 조회 모두 이 allowlist 를 통과한 이름만 로드한다.
"""

from __future__ import annotations

# Preset 모델 — UI 토글 + worker 자동 다운로드 기본값. 신뢰경계: ultralytics 공식 가중치만.
PRESET_MODELS: tuple[str, ...] = (
    "yolo26n-seg.pt",
    "yolo26s-seg.pt",
    "yolo26m-seg.pt",
    "yolo26l-seg.pt",
    "yolo26x-seg.pt",
)

# 허용된 YOLO26 seg preset은 모두 COCO 80-class 모델이다. 클래스 설정 UI가
# 가중치를 다운로드하지 않도록 정적 메타데이터를 별도로 보관한다.
COCO_CLASS_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def is_preset(name: str) -> bool:
    """name 이 허용된 preset 인지 (단일 allowlist 게이트)."""
    return name in PRESET_MODELS


def list_all_models() -> list[dict]:
    """preset 목록 (UI 드롭다운용). custom 없음."""
    return [{"name": n, "type": "preset", "size_mb": None} for n in PRESET_MODELS]


def list_model_classes(name: str) -> list[dict]:
    """preset 모델의 COCO 클래스 메타데이터를 가중치 로드 없이 반환한다."""
    if not is_preset(name):
        raise ValueError(f"허용되지 않은 모델: {name!r} (preset 만 가능)")
    return [
        {"id": class_id, "name": class_name}
        for class_id, class_name in enumerate(COCO_CLASS_NAMES)
    ]


def resolve_model_path(name: str) -> str:
    """모델 이름 → ultralytics 로드용 이름. preset 만 허용, 그 외는 거부.

    경로가 아니라 preset 이름을 그대로 돌려준다(ultralytics 가 캐시/다운로드). 임의 파일
    경로 로드를 원천 차단해 pickle RCE 경로를 없앤다.
    """
    if not is_preset(name):
        raise ValueError(f"허용되지 않은 모델: {name!r} (preset 만 가능)")
    return name
