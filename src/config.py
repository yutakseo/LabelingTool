"""User-editable paths, class definitions, and application defaults."""

from __future__ import annotations

from pathlib import Path
import time


# 사용자가 설정할 세 가지 경로
# 1) 원본 이미지 파일, 이미지 폴더 또는 비디오
ORIGINAL_IMAGE_INPUT_PATH = Path(
    r"D:\workspace\LabelingTool\output\시범 라벨링_v2\images\c1_mono_cropped.png"
)

# 2) 최초 라벨로 사용할 수도 마스크 파일 또는 마스크 폴더(None 가능)
PSEUDO_MASK_INPUT_PATH: Path | None = Path(
    r"D:\workspace\LabelingTool\output\시범 라벨링_v2\masks\c1_mono_cropped.png"
)

# 3) 학습용 데이터셋을 생성할 상위 폴더
DATASET_OUTPUT_PARENT_PATH = Path(r"D:\workspace\LabelingTool\output")
DATASET_SESSION_NAME = time.strftime("%Y%m%d_%H%M")

# Add, remove, or rename classes here.  The numeric key is the value saved in
# the multi-class mask; names and colours are reflected in the labeling UI.
CLASS_DEFINITIONS: dict[int, dict[str, object]] = {
    1: {"name": "class_1", "color": (0.0, 1.0, 0.0, 1.0)},
}

RGB_EXTS   = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MASK_EXTS  = {".png", ".tif", ".tiff", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}

# 비디오 프레임 저장 포맷
FRAME_EXT = ".png"
FRAME_NAME_FMT = "{idx:06d}"
FPS_DIR_PREFIX = "fps_"

# Brush
BRUSH_MIN = 1
BRUSH_MAX = 100
BRUSH_STEP = 1
BRUSH_WHEEL_STEP = 3

# Zoom
ZOOM_MIN = 0.1
ZOOM_MAX = 20.0
ZOOM_STEP = 0.1
ZOOM_WHEEL_RATIO = 1.1

# Undo history (per layer)
MAX_HISTORY = 60
FRAME_CACHE_LIMIT = 4
AUTOSAVE_DELAY_MS = 700
CHANGE_SCAN_MS = 500

