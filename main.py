# main.py
# pip install "napari[all]" imageio imageio-ffmpeg numpy qtpy tqdm

from __future__ import annotations
import argparse
import json
import logging
import math
import shutil
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import TypeAlias, cast

import imageio_ffmpeg
import numpy as np
import imageio.v3 as iio
import napari
from napari.utils.colormaps import DirectLabelColormap
from qtpy import QtWidgets, QtCore, QtGui
from tqdm import tqdm
import time

LOGGER = logging.getLogger(__name__)

# 사용자가 설정할 세 가지 경로
# 1) 원본 이미지 파일, 이미지 폴더 또는 비디오
ORIGINAL_IMAGE_INPUT_PATH = Path(
    r"D:\workspace\LabelingTool\output\시범 라벨링\images\c1_mono_cropped.png"
)

# 2) 최초 라벨로 사용할 수도 마스크 파일 또는 마스크 폴더(None 가능)
PSEUDO_MASK_INPUT_PATH: Path | None = Path(
    r"D:\workspace\LabelingTool\output\시범 라벨링\masks\c1_mono_cropped.png"
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
FILL_STRUCTURE = np.array(
    [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ],
    dtype=bool,
)

NEXT_KEY_CODES: frozenset[int] = frozenset(
    {
        int(QtCore.Qt.Key.Key_Return),
        int(QtCore.Qt.Key.Key_Enter),
        int(QtCore.Qt.Key.Key_Right),
        int(QtCore.Qt.Key.Key_Down),
    }
)
PREV_KEY_CODES: frozenset[int] = frozenset(
    {
        int(QtCore.Qt.Key.Key_Left),
        int(QtCore.Qt.Key.Key_Up),
    }
)
RIGHT_BUTTON = QtCore.Qt.MouseButton.RightButton

NavAction: TypeAlias = Callable[[], None]
RangeAction: TypeAlias = Callable[[int], bool]
PanAction: TypeAlias = Callable[[int, int], bool]
ClassMasks: TypeAlias = dict[int, np.ndarray]
FrameData: TypeAlias = tuple[np.ndarray, ClassMasks]
FrameCache: TypeAlias = dict[str, FrameData]
FrameFuture: TypeAlias = dict[str, Future[FrameData]]


def text_input_has_focus() -> bool:
    """Return True while the user is typing or editing a numeric text field."""
    focus = QtWidgets.QApplication.focusWidget()
    return isinstance(
        focus,
        (
            QtWidgets.QLineEdit,
            QtWidgets.QTextEdit,
            QtWidgets.QPlainTextEdit,
            QtWidgets.QAbstractSpinBox,
        ),
    )


def validate_class_definitions() -> None:
    if not CLASS_DEFINITIONS:
        raise ValueError("CLASS_DEFINITIONS must contain at least one class")
    for class_id, definition in CLASS_DEFINITIONS.items():
        if not isinstance(class_id, int) or not 1 <= class_id <= 255:
            raise ValueError("Each class ID must be an integer between 1 and 255")
        if not str(definition.get("name", "")).strip():
            raise ValueError(f"Class {class_id} must have a non-empty name")
        color = definition.get("color")
        if not isinstance(color, tuple) or len(color) != 4:
            raise ValueError(f"Class {class_id} color must be an RGBA tuple")


def default_class_color(class_id: int) -> tuple[float, float, float, float]:
    palette = (
        (0.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (0.0, 0.7, 1.0, 1.0),
        (1.0, 0.4, 0.0, 1.0),
        (0.6, 0.3, 1.0, 1.0),
    )
    return palette[(class_id - 1) % len(palette)]


def register_mask_class_ids(values: np.ndarray) -> None:
    """Register IDs found in an existing mask without merging or discarding them."""
    for value in np.unique(values):
        class_id = int(value)
        if class_id == 0 or class_id in CLASS_DEFINITIONS:
            continue
        if not 1 <= class_id <= 255:
            raise ValueError(f"Mask contains unsupported class ID: {class_id}")
        CLASS_DEFINITIONS[class_id] = {
            "name": f"class_{class_id}",
            "color": default_class_color(class_id),
        }


def load_dataset_class_definitions(dataset_root: Path) -> None:
    """Restore classes already recorded in a dataset before opening its masks."""
    metadata_path = dataset_root / "classes.json"
    if not metadata_path.is_file():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for item in metadata.get("classes", []):
        try:
            class_id = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= class_id <= 255:
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        color = item.get("color_rgba", default_class_color(class_id))
        if not isinstance(color, (list, tuple)) or len(color) != 4:
            color = default_class_color(class_id)
        existing = CLASS_DEFINITIONS.get(class_id)
        if existing is not None:
            existing_name = str(existing.get("name") or "").strip()
            # Preserve a meaningful name already loaded from the destination,
            # but replace empty/automatic placeholders with dataset metadata.
            if existing_name and existing_name != f"class_{class_id}":
                continue
        CLASS_DEFINITIONS[class_id] = {
            "name": name,
            "color": tuple(float(channel) for channel in color),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an image file, image folder, or video file in the napari labeling tool."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Original image file/folder or video. Defaults to ORIGINAL_IMAGE_INPUT_PATH.",
    )
    parser.add_argument(
        "--pseudo-mask", "--mask-dir",
        dest="pseudo_mask",
        help="Optional pseudo-mask file or directory used to initialize new masks.",
    )
    parser.add_argument(
        "--dataset-output", "--output-root",
        dest="dataset_output",
        help="Parent directory where a YYYYMMDD_HHMM dataset folder is created.",
    )
    parser.add_argument(
        "--fps",
        dest="fps",
        type=float,
        help="Target FPS for video frame extraction. If omitted, the script asks in the terminal.",
    )
    return parser.parse_args(argv)


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser()


def resolve_runtime_paths(argv: list[str] | None = None) -> tuple[Path, Path | None, Path | None, float | None]:
    args = parse_args(argv)
    input_path = _optional_path(args.input_path) or ORIGINAL_IMAGE_INPUT_PATH
    pseudo_mask = (
        _optional_path(args.pseudo_mask)
        if args.pseudo_mask is not None
        else PSEUDO_MASK_INPUT_PATH
    )
    dataset_parent = (
        _optional_path(args.dataset_output)
        if args.dataset_output is not None
        else DATASET_OUTPUT_PARENT_PATH
    )
    dataset_output = Path(dataset_parent).expanduser() / DATASET_SESSION_NAME
    return Path(input_path).expanduser(), pseudo_mask, dataset_output, args.fps


def list_images(folder: Path, exts: set[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    return out


def load_rgb(path: Path) -> np.ndarray:
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def load_multiclass_mask(mask_path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    """Load a configured multi-class mask; treat legacy nonzero masks as class 1."""
    m = iio.imread(mask_path)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != shape_hw:
        raise ValueError(f"Shape mismatch: RGB {shape_hw} vs MASK {m.shape[:2]} ({mask_path.name})")
    mask = m.astype(np.uint8)
    values = set(int(value) for value in np.unique(mask))
    # Only the old binary 0/255 convention is converted to class 1. Other
    # numeric IDs are preserved even when they are not configured yet.
    if values.issubset({0, 255}) and 255 not in CLASS_DEFINITIONS:
        return (mask > 0).astype(np.uint8)
    return mask


def compose_multiclass_mask(class_masks: Mapping[int, np.ndarray]) -> np.ndarray:
    """Combine class layers into one mask; later class IDs win on overlap."""
    if set(class_masks) != set(CLASS_DEFINITIONS):
        raise ValueError("Class layer IDs do not match CLASS_DEFINITIONS")
    shapes = {np.asarray(mask).shape for mask in class_masks.values()}
    if len(shapes) != 1:
        raise ValueError(f"Class-layer shape mismatch: {sorted(shapes)}")
    mask = np.zeros(next(iter(class_masks.values())).shape, dtype=np.uint8)
    for class_id in sorted(CLASS_DEFINITIONS):
        mask[np.asarray(class_masks[class_id]) > 0] = class_id
    return mask


def colorize_multiclass_mask(mask: np.ndarray) -> np.ndarray:
    """Create an RGB preview while preserving the indexed mask for training."""
    preview = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for class_id, definition in CLASS_DEFINITIONS.items():
        rgba = np.asarray(definition["color"], dtype=float)
        preview[np.asarray(mask) == class_id] = np.clip(
            np.rint(rgba[:3] * 255), 0, 255
        ).astype(np.uint8)
    return preview


def write_mask_preview(mask_path: Path, mask: np.ndarray) -> Path:
    """Save a human-readable colour preview outside the training masks folder."""
    preview_dir = mask_path.parent.parent / "mask_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / mask_path.name
    iio.imwrite(preview_path, colorize_multiclass_mask(mask))
    return preview_path


def loadFrame(
    frame: str,
    rgb_map: dict[str, Path],
    mask_map: dict[str, Path],
) -> FrameData:
    try:
        image = load_rgb(rgb_map[frame])
        mask = load_multiclass_mask(mask_map[frame], image.shape[:2])
    except Exception as exc:
        raise RuntimeError(f"Frame load failed: {frame}") from exc
    class_masks = {
        class_id: (mask == class_id).astype(np.uint8)
        for class_id in CLASS_DEFINITIONS
    }
    return image, class_masks


def fillRegion(
    mask: np.ndarray,
    seed_y: int,
    seed_x: int,
    target_label: int,
) -> np.ndarray | None:
    try:
        from scipy import ndimage
    except Exception:
        return None

    arr = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = arr.shape[:2]
    row = int(np.clip(seed_y, 0, height - 1))
    col = int(np.clip(seed_x, 0, width - 1))

    if target_label == 1:
        if arr[row, col] == 1:
            return arr.copy()
        region = arr == 0
        fill_label = 1
    else:
        if arr[row, col] == 0:
            return arr.copy()
        region = arr == 1
        fill_label = 0

    try:
        labels, _ = ndimage.label(region, structure=FILL_STRUCTURE)
    except Exception:
        return None

    label = int(labels[row, col])
    out = arr.copy()
    if label == 0:
        return out
    out[labels == label] = fill_label
    return out


def flood_fill_region(mask: np.ndarray, seed_y: int, seed_x: int, target_label: int) -> np.ndarray:
    """
    L1 전용 페인트통.
    - target_label == 1: 현재 L1의 값 1을 경계로 보고, seed가 포함된 0 영역을 1로 채움
    - target_label == 0: seed가 포함된 연결된 1 영역만 0으로 지움
    """
    filled = fillRegion(mask, seed_y, seed_x, target_label)
    if filled is not None:
        return filled

    arr = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = arr.shape[:2]
    sy = int(np.clip(seed_y, 0, h - 1))
    sx = int(np.clip(seed_x, 0, w - 1))
    out = arr.copy()

    if target_label == 1:
        if arr[sy, sx] == 1:
            return out
        visited = np.zeros((h, w), dtype=bool)
        q = deque([(sy, sx)])
        visited[sy, sx] = True
        while q:
            y, x = q.popleft()
            out[y, x] = 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and arr[ny, nx] == 0:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        return out

    if arr[sy, sx] == 0:
        return out

    visited = np.zeros((h, w), dtype=bool)
    q = deque([(sy, sx)])
    visited[sy, sx] = True
    while q:
        y, x = q.popleft()
        out[y, x] = 0
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and arr[ny, nx] == 1:
                visited[ny, nx] = True
                q.append((ny, nx))
    return out


def is_video_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def default_video_project_root(video_path: Path, output_root: Path | None) -> Path:
    if output_root is None:
        return video_path.parent / video_path.stem
    return Path(output_root) / video_path.stem


def build_missing_input_message(input_path: Path) -> str:
    lines = [
        "INPUT_PATH가 유효한 이미지 폴더도 아니고 지원되는 비디오 파일도 아니에요:",
        f"  {input_path}",
    ]

    parent = input_path.parent
    if parent.exists():
        if input_path.suffix.lower() in VIDEO_EXTS:
            candidate_exts = VIDEO_EXTS
        else:
            candidate_exts = RGB_EXTS | VIDEO_EXTS
        candidates = sorted(
            p.name for p in parent.iterdir()
            if p.is_file() and p.suffix.lower() in candidate_exts
        )
        if candidates:
            preview = ", ".join(candidates[:8])
            if len(candidates) > 8:
                preview += ", ..."
            lines.append(f"같은 폴더의 후보 파일: {preview}")

    lines.append(r"실행 예: python main.py Z:\CODE\video\kia-1-1.mp4")
    return "\n".join(lines)


def format_fps_value(fps: float) -> str:
    if math.isclose(fps, round(fps), abs_tol=1e-6):
        return str(int(round(fps)))
    return f"{fps:.3f}".rstrip("0").rstrip(".")


def format_fps_dirname(fps: float) -> str:
    return f"{FPS_DIR_PREFIX}{format_fps_value(fps).replace('.', '_')}"


def load_video_source_info(video_path: Path) -> dict[str, float | int]:
    meta = iio.immeta(video_path)
    source_fps = float(meta.get("fps") or 0.0)
    if source_fps <= 0:
        raise RuntimeError(f"비디오 fps 정보를 읽지 못했어요: {video_path}")

    total_frames = 0
    duration = float(meta.get("duration") or 0.0)
    try:
        total_frames, counted_duration = imageio_ffmpeg.count_frames_and_secs(str(video_path))
        total_frames = int(total_frames)
        duration = float(counted_duration)
    except Exception:
        if duration > 0:
            total_frames = max(1, int(round(duration * source_fps)))

    if total_frames <= 0:
        raise RuntimeError(f"비디오 총 프레임 수를 계산하지 못했어요: {video_path}")

    if duration <= 0:
        duration = float(total_frames / source_fps)

    return {
        "source_fps": source_fps,
        "duration": duration,
        "total_frames": total_frames,
    }


def estimate_saved_frame_count(total_frames: int, source_fps: float, target_fps: float) -> int:
    if total_frames <= 0:
        return 0
    last_src_idx = total_frames - 1
    return int(math.floor((last_src_idx * target_fps) / source_fps + 1e-9)) + 1


def build_preview_fps_values(source_fps: float) -> list[float]:
    preview_cap = min(int(math.floor(source_fps)), 60)
    values = [float(v) for v in range(1, max(preview_cap, 1) + 1)]
    if not any(math.isclose(v, source_fps, abs_tol=1e-6) for v in values):
        values.append(float(source_fps))
    return values


def print_video_fps_table(video_path: Path, video_info: dict[str, float | int]):
    source_fps = float(video_info["source_fps"])
    total_frames = int(video_info["total_frames"])
    duration = float(video_info["duration"])
    preview_values = build_preview_fps_values(source_fps)

    print()
    print("=" * 58)
    print(f"Video           : {video_path}")
    print(f"Source FPS      : {source_fps:.3f}")
    print(f"Source Frames   : {total_frames}")
    print(f"Duration (sec)  : {duration:.2f}")
    print("-" * 58)
    print(f"{'Target FPS':>12} | {'Saved Frames':>12}")
    print("-" * 58)
    for fps in preview_values:
        expected_frames = estimate_saved_frame_count(total_frames, source_fps, fps)
        suffix = "  <- original" if math.isclose(fps, source_fps, abs_tol=1e-6) else ""
        print(f"{format_fps_value(fps):>12} | {expected_frames:>12}{suffix}")
    if source_fps > 60:
        print("-" * 58)
        print("표에는 1~60fps와 원본 fps만 표시했어요. 입력은 원본 fps 이하 실수값도 가능합니다.")
    print("=" * 58)


def validate_target_fps(target_fps: float, source_fps: float) -> float:
    target_fps = float(target_fps)
    if not math.isfinite(target_fps):
        raise ValueError("fps는 유한한 숫자여야 해요.")
    if target_fps <= 0:
        raise ValueError("fps는 0보다 커야 해요.")
    if target_fps > source_fps + 1e-6:
        raise ValueError(f"fps는 원본 fps({source_fps:.3f})보다 클 수 없어요.")
    return target_fps


def choose_video_target_fps(video_path: Path, preset_fps: float | None = None) -> tuple[float, dict[str, float | int]]:
    video_info = load_video_source_info(video_path)
    source_fps = float(video_info["source_fps"])
    print_video_fps_table(video_path, video_info)

    if preset_fps is not None:
        target_fps = validate_target_fps(preset_fps, source_fps)
        print(f"선택된 fps: {format_fps_value(target_fps)}")
        return target_fps, video_info

    prompt = (
        f"저장할 fps를 입력하세요 "
        f"(0 < fps <= {source_fps:.3f}, Enter={format_fps_value(source_fps)}): "
    )
    while True:
        raw = input(prompt).strip()
        if raw == "":
            target_fps = source_fps
            break
        try:
            target_fps = validate_target_fps(float(raw), source_fps)
            break
        except ValueError as e:
            print(e)

    print(f"선택된 fps: {format_fps_value(target_fps)}")
    return target_fps, video_info


def clear_existing_rgb_frames(image_dir: Path):
    if not image_dir.exists():
        return
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in RGB_EXTS:
            p.unlink()


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def find_existing_video_project(
    project_root: Path,
) -> tuple[Path, Path, dict[str, Path]] | None:
    candidates: list[tuple[float, Path, Path, Path, dict[str, Path]]] = []

    def add_candidate(root: Path):
        rgb_dir = root / "images"
        if not rgb_dir.is_dir():
            return
        rgb_map = dict(sorted(list_images(rgb_dir, RGB_EXTS).items()))
        if not rgb_map:
            return
        mask_dir = root / "masks"
        mtime = max(_path_mtime(root), _path_mtime(rgb_dir), _path_mtime(mask_dir))
        candidates.append((mtime, root, rgb_dir, mask_dir, rgb_map))

    add_candidate(project_root)
    if project_root.is_dir():
        for child in project_root.iterdir():
            if child.is_dir() and child.name.startswith(FPS_DIR_PREFIX):
                add_candidate(child)

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, chosen_root, rgb_dir, mask_dir, rgb_map = candidates[0]
    print(f"Opening existing project without FPS prompt: {chosen_root}")
    return rgb_dir, mask_dir, rgb_map


def source_frame_index_for_output(
    output_idx: int,
    source_fps: float,
    target_fps: float,
    total_frames: int,
) -> int:
    src_idx = int(round((output_idx * source_fps) / target_fps))
    return min(src_idx, total_frames - 1)


def ensure_mask_files_for_rgb(
    rgb_dir: Path,
    mask_dir: Path,
    rgb_map: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """RGB 기준으로 비어있는 마스크 파일을 자동 생성해서 stem 매칭을 보장"""
    if rgb_map is None:
        rgb_map = list_images(rgb_dir, RGB_EXTS)
    mask_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path] = {}
    for stem, rgb_path in rgb_map.items():
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            rgb = load_rgb(rgb_path)
            empty_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            iio.imwrite(mask_path, empty_mask)
        out[stem] = mask_path
    return out


def extract_video_frames(
    video_path: Path,
    image_dir: Path,
    source_fps: float,
    target_fps: float,
    total_frames: int,
) -> dict[str, Path]:
    """비디오를 프레임 이미지로 저장. 이미 추출된 프레임이 있으면 재사용"""
    image_dir.mkdir(parents=True, exist_ok=True)

    expected_frames = estimate_saved_frame_count(total_frames, source_fps, target_fps)
    existing = list_images(image_dir, RGB_EXTS)
    if len(existing) == expected_frames and expected_frames > 0:
        print(f"기존 추출 프레임 재사용: {image_dir} ({len(existing)} frames)")
        return dict(sorted(existing.items()))

    if existing:
        print(
            f"기존 추출 프레임 수({len(existing)})와 예상 수({expected_frames})가 달라서 다시 추출합니다: {image_dir}"
        )
        clear_existing_rgb_frames(image_dir)

    next_output_idx = 0
    next_source_idx = source_frame_index_for_output(
        next_output_idx, source_fps, target_fps, total_frames
    )

    with tqdm(
        total=expected_frames,
        desc=f"Extract {format_fps_value(target_fps)} fps",
        unit="frame",
    ) as pbar:
        for src_idx, frame in enumerate(iio.imiter(video_path)):
            if src_idx != next_source_idx:
                continue
            frame = np.asarray(frame)
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            out_path = image_dir / f"{FRAME_NAME_FMT.format(idx=next_output_idx)}{FRAME_EXT}"
            iio.imwrite(out_path, frame)
            next_output_idx += 1
            pbar.update(1)

            if next_output_idx >= expected_frames:
                break

            next_source_idx = source_frame_index_for_output(
                next_output_idx, source_fps, target_fps, total_frames
            )

    created = list_images(image_dir, RGB_EXTS)
    if len(created) != expected_frames:
        raise RuntimeError(f"비디오에서 프레임을 추출하지 못했어요: {video_path}")
    return dict(sorted(created.items()))


def prepare_input_output(
    input_path: Path,
    mask_dir_override: Path | None = None,
    output_root: Path | None = None,
    video_fps: float | None = None,
) -> tuple[Path, Path, dict[str, Path], dict[str, Path], list[str]]:
    """
    입력이 이미지 폴더면:
        rgb_dir = input_path
        mask_dir = MASK_DIR 또는 자동 생성 폴더
    입력이 비디오 파일이면:
        rgb_dir = AUTO_OUTPUT_ROOT/<video_stem>/images
        mask_dir = AUTO_OUTPUT_ROOT/<video_stem>/masks
        이미 images / masks가 있으면 그것을 그대로 재사용
        없으면 비디오를 프레임별 이미지로 자동 저장
    """
    input_path = Path(input_path).expanduser()
    explicit_mask_path: Path | None = None

    if is_video_path(input_path):
        base_project_root = default_video_project_root(input_path, output_root)
        existing_project = None if video_fps is not None else find_existing_video_project(base_project_root)
        if existing_project is not None:
            rgb_dir, mask_dir, rgb_map = existing_project
        else:
            target_fps, video_info = choose_video_target_fps(input_path, preset_fps=video_fps)

        # 비디오에 대한 기존 작업 폴더(images + masks)가 이미 있으면 그대로 사용
        # mask가 일부 비어 있어도 아래 ensure_mask_files_for_rgb에서 자동 보완됨
            project_root = base_project_root / format_fps_dirname(target_fps)
            rgb_dir = project_root / "images"
            mask_dir = project_root / "masks"
            rgb_map = extract_video_frames(
            input_path,
            rgb_dir,
            source_fps=float(video_info["source_fps"]),
            target_fps=target_fps,
            total_frames=int(video_info["total_frames"]),
        )
    elif input_path.is_dir():
        rgb_dir = input_path
        if mask_dir_override is not None:
            mask_dir = Path(mask_dir_override)
            if mask_dir.is_file() or (
                not mask_dir.exists() and mask_dir.suffix.lower() in MASK_EXTS
            ):
                raise ValueError(
                    "MASK_DIR must be a directory when INPUT_PATH is an image directory"
                )
        else:
            mask_dir = input_path.parent / f"{input_path.name}_masks"
        rgb_map = dict(sorted(list_images(rgb_dir, RGB_EXTS).items()))
        if not rgb_map:
            raise RuntimeError(f"이미지 폴더에 읽을 수 있는 이미지가 없어요: {rgb_dir}")
    elif input_path.is_file() and input_path.suffix.lower() in RGB_EXTS:
        # A single image is handled as a one-frame labeling project.
        rgb_dir = input_path.parent
        if mask_dir_override is not None:
            mask_location = Path(mask_dir_override)
            if mask_location.is_file() or (
                not mask_location.exists() and mask_location.suffix.lower() in MASK_EXTS
            ):
                explicit_mask_path = mask_location
                mask_dir = mask_location.parent
            else:
                mask_dir = mask_location
        else:
            mask_dir = input_path.parent / f"{input_path.stem}_masks"
        rgb_map = {input_path.stem: input_path}
    else:
        raise FileNotFoundError(
            "INPUT_PATH가 유효한 이미지 폴더도 아니고 지원되는 비디오 파일도 아니에요: "
            f"{input_path}"
        )

    if explicit_mask_path is not None:
        explicit_mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not explicit_mask_path.exists():
            rgb = load_rgb(input_path)
            iio.imwrite(explicit_mask_path, np.zeros(rgb.shape[:2], dtype=np.uint8))
        mask_map = {input_path.stem: explicit_mask_path}
    else:
        mask_map = dict(sorted(ensure_mask_files_for_rgb(rgb_dir, mask_dir, rgb_map).items()))
    keys = sorted(set(rgb_map.keys()) & set(mask_map.keys()))
    if not keys:
        raise RuntimeError(
            "매칭되는 파일이 없어요. RGB/MASK 파일명이 stem(확장자 제외) 기준으로 같은지 확인해줘.\n"
            f"RGB sample: {list(rgb_map.keys())[:10]}\n"
            f"MASK sample: {list(mask_map.keys())[:10]}"
        )

    return rgb_dir, mask_dir, rgb_map, mask_map, keys


def write_dataset_metadata(dataset_root: Path) -> Path:
    """Write class metadata beside the standard images/ and masks/ folders."""
    metadata = {
        "format": "semantic-segmentation-indexed-png",
        "images": "images",
        "masks": "masks",
        "mask_previews": "mask_previews (visualization only; do not use for training)",
        "mask_dtype": "uint8",
        "background": {"id": 0, "name": "background"},
        "classes": [
            {
                "id": class_id,
                "name": str(definition["name"]),
                "color_rgba": list(definition["color"]),
            }
            for class_id, definition in sorted(CLASS_DEFINITIONS.items())
        ],
    }
    dataset_root.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_root / "classes.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def _copy_source_image(source: Path, images_dir: Path) -> Path:
    destination = images_dir / source.name
    images_dir.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _pseudo_mask_map(
    pseudo_mask_input: Path | None,
    source_images: Mapping[str, Path],
) -> dict[str, Path]:
    if pseudo_mask_input is None:
        return {}
    pseudo_mask_input = Path(pseudo_mask_input).expanduser()
    if pseudo_mask_input.is_file():
        if len(source_images) != 1:
            raise ValueError(
                "A pseudo-mask file can only be used with one original image; "
                "use a pseudo-mask directory for an image folder"
            )
        return {next(iter(source_images)): pseudo_mask_input}
    if pseudo_mask_input.is_dir():
        return dict(sorted(list_images(pseudo_mask_input, MASK_EXTS).items()))
    raise FileNotFoundError(f"Cannot find pseudo-mask input: {pseudo_mask_input}")


def prepare_segmentation_dataset(
    input_path: Path,
    pseudo_mask_input: Path | None,
    dataset_output_root: Path | None,
    video_fps: float | None = None,
) -> tuple[Path, Path, dict[str, Path], dict[str, Path], list[str]]:
    """Stage data as dataset/images + dataset/masks with matching file stems."""
    input_path = Path(input_path).expanduser()

    if is_video_path(input_path):
        rgb_dir, mask_dir, rgb_map, mask_map, keys = prepare_input_output(
            input_path,
            mask_dir_override=None,
            output_root=dataset_output_root,
            video_fps=video_fps,
        )
        write_dataset_metadata(mask_dir.parent)
        return rgb_dir, mask_dir, rgb_map, mask_map, keys

    if input_path.is_file() and input_path.suffix.lower() in RGB_EXTS:
        source_images = {input_path.stem: input_path}
        default_root = input_path.parent / f"{input_path.stem}_dataset"
    elif input_path.is_dir():
        source_images = dict(sorted(list_images(input_path, RGB_EXTS).items()))
        if not source_images:
            raise RuntimeError(f"No readable images in input directory: {input_path}")
        default_root = input_path.parent / f"{input_path.name}_dataset"
    else:
        raise FileNotFoundError(
            f"Original input is not a supported image, image directory, or video: {input_path}"
        )

    dataset_root = Path(dataset_output_root or default_root).expanduser()
    load_dataset_class_definitions(dataset_root)
    # When the inputs come from another generated dataset, inherit its class
    # names and colours into the new timestamped output folder.
    if input_path.is_file() and input_path.parent.name == "images":
        load_dataset_class_definitions(input_path.parent.parent)
    if pseudo_mask_input is not None:
        pseudo_path = Path(pseudo_mask_input).expanduser()
        if pseudo_path.is_file() and pseudo_path.parent.name == "masks":
            load_dataset_class_definitions(pseudo_path.parent.parent)
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rgb_map = {
        stem: _copy_source_image(source, images_dir)
        for stem, source in source_images.items()
    }
    pseudo_masks = _pseudo_mask_map(pseudo_mask_input, source_images)
    mask_map: dict[str, Path] = {}

    for stem, image_path in rgb_map.items():
        mask_path = masks_dir / f"{stem}.png"
        image = load_rgb(image_path)
        if not mask_path.exists():
            pseudo_path = pseudo_masks.get(stem)
            if pseudo_path is None:
                initial_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                initial_mask = load_multiclass_mask(pseudo_path, image.shape[:2])
            iio.imwrite(mask_path, initial_mask.astype(np.uint8))
        indexed_mask = load_multiclass_mask(mask_path, image.shape[:2])
        register_mask_class_ids(indexed_mask)
        write_mask_preview(mask_path, indexed_mask)
        mask_map[stem] = mask_path

    keys = sorted(rgb_map)
    write_dataset_metadata(dataset_root)
    return images_dir, masks_dir, rgb_map, mask_map, keys


def prepare_input_output_checked(
    input_path: Path,
    pseudo_mask_input: Path | None = None,
    dataset_output_root: Path | None = None,
    video_fps: float | None = None,
) -> tuple[Path, Path, dict[str, Path], dict[str, Path], list[str]]:
    input_path = Path(input_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(build_missing_input_message(input_path))
    return prepare_segmentation_dataset(
        input_path,
        pseudo_mask_input=pseudo_mask_input,
        dataset_output_root=dataset_output_root,
        video_fps=video_fps,
    )


class History:
    """레이어별 undo/redo 스택 (현재는 남겨두지만 버튼/키는 Ctrl+Z 방식으로 동작)"""
    def __init__(self, maxlen: int = 60):
        self.maxlen = maxlen
        self.undo_stack: list[np.ndarray] = []
        self.redo_stack: list[np.ndarray] = []
        self.suspend = False

    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()

    def push(self, prev_state: np.ndarray):
        if self.suspend:
            return
        self.undo_stack.append(prev_state.copy())
        if len(self.undo_stack) > self.maxlen:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self, current: np.ndarray) -> np.ndarray | None:
        if not self.undo_stack:
            return None
        prev = self.undo_stack.pop()
        self.redo_stack.append(current.copy())
        return prev

    def redo(self, current: np.ndarray) -> np.ndarray | None:
        if not self.redo_stack:
            return None
        nxt = self.redo_stack.pop()
        self.undo_stack.append(current.copy())
        return nxt


class InputFilter(QtCore.QObject):
    def __init__(
        self,
        window: QtWidgets.QWidget,
        canvas: QtWidgets.QWidget | None,
        next_action: NavAction,
        prev_action: NavAction,
        range_action: RangeAction,
        zoom_action: RangeAction,
        pan_action: PanAction,
    ) -> None:
        super().__init__(window)
        self.window: QtWidgets.QWidget = window
        self.canvas: QtWidgets.QWidget | None = canvas
        self.next_action: NavAction = next_action
        self.prev_action: NavAction = prev_action
        self.range_action: RangeAction = range_action
        self.zoom_action: RangeAction = zoom_action
        self.pan_action: PanAction = pan_action
        self.pan_point: QtCore.QPoint | None = None

    def eventFilter(
        self,
        source: QtCore.QObject,
        event: QtCore.QEvent,
    ) -> bool:
        if not self.checkWindow():
            return False

        if event.type() == QtCore.QEvent.Type.KeyPress and isinstance(event, QtGui.QKeyEvent):
            if text_input_has_focus():
                return False
            return self.filterKey(event)
        if event.type() == QtCore.QEvent.Type.Wheel and isinstance(event, QtGui.QWheelEvent):
            return self.filterWheel(source, event)
        if isinstance(event, QtGui.QMouseEvent) and event.type() in {
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseMove,
            QtCore.QEvent.Type.MouseButtonRelease,
        }:
            return self.filterMouse(source, event)
        return False

    def filterKey(self, event: QtCore.QEvent) -> bool:
        if not isinstance(event, QtGui.QKeyEvent):
            return False
        key_event = cast(QtGui.QKeyEvent, event)
        key_code = self.readKeyCode(key_event)
        if key_code in NEXT_KEY_CODES:
            self.next_action()
            key_event.accept()
            return True
        if key_code in PREV_KEY_CODES:
            self.prev_action()
            key_event.accept()
            return True
        return False

    def filterWheel(
        self,
        source: QtCore.QObject,
        event: QtCore.QEvent,
    ) -> bool:
        # Some Qt/napari event dispatch paths can report Wheel for a QKeyEvent.
        # Do not cast blindly: only QWheelEvent provides angleDelta().
        if not isinstance(event, QtGui.QWheelEvent):
            return False
        if not self.checkCanvas(source):
            return False

        wheel_event = cast(QtGui.QWheelEvent, event)
        wheel_step = self.readWheelStep(wheel_event)
        if wheel_step == 0:
            return False

        if self.checkControl(wheel_event):
            wheel_action = self.zoom_action
        else:
            wheel_action = self.range_action
        if wheel_action(wheel_step):
            wheel_event.accept()
            return True
        return False

    def filterMouse(
        self,
        source: QtCore.QObject,
        event: QtCore.QEvent,
    ) -> bool:
        if not isinstance(event, QtGui.QMouseEvent):
            return False
        mouse_event = cast(QtGui.QMouseEvent, event)
        event_type = event.type()

        if event_type == QtCore.QEvent.Type.MouseButtonRelease:
            if self.pan_point is not None and self.checkRightButton(mouse_event):
                return self.stopPan(mouse_event)
            return False

        if not self.checkCanvas(source):
            return False
        if event_type == QtCore.QEvent.Type.MouseButtonPress:
            if self.checkRightButton(mouse_event):
                return self.startPan(mouse_event)
            return False
        if event_type == QtCore.QEvent.Type.MouseMove:
            return self.movePan(mouse_event)
        return False

    def startPan(self, event: QtGui.QMouseEvent) -> bool:
        self.pan_point = self.readMousePoint(event)
        event.accept()
        return True

    def movePan(self, event: QtGui.QMouseEvent) -> bool:
        if self.pan_point is None:
            return False
        if not self.checkRightButtons(event):
            return False

        mouse_point = self.readMousePoint(event)
        point_delta = mouse_point - self.pan_point
        self.pan_point = mouse_point
        if point_delta.isNull():
            event.accept()
            return True

        if self.pan_action(point_delta.x(), point_delta.y()):
            event.accept()
            return True
        return False

    def stopPan(self, event: QtGui.QMouseEvent) -> bool:
        self.pan_point = None
        event.accept()
        return True

    def checkWindow(self) -> bool:
        modal_widget = QtWidgets.QApplication.activeModalWidget()
        if modal_widget is not None:
            return False

        focus_widget = QtWidgets.QApplication.focusWidget()
        try:
            if focus_widget is None:
                return self.window.isActiveWindow()
            return focus_widget is self.window or self.window.isAncestorOf(focus_widget)
        except RuntimeError:
            # Qt may deliver a final event while the window is being destroyed.
            return False

    def checkCanvas(self, source: QtCore.QObject) -> bool:
        if self.canvas is None:
            return True
        if isinstance(source, QtWidgets.QWidget):
            return source is self.canvas or self.canvas.isAncestorOf(source)

        widget = QtWidgets.QApplication.widgetAt(QtGui.QCursor.pos())
        if widget is None:
            return False
        return widget is self.canvas or self.canvas.isAncestorOf(widget)

    def readKeyCode(self, event: QtGui.QKeyEvent) -> int:
        key_value = event.key()
        return int(getattr(key_value, "value", key_value))

    def checkControl(self, event: QtGui.QWheelEvent) -> bool:
        control = QtCore.Qt.KeyboardModifier.ControlModifier
        return bool(event.modifiers() & control)

    def checkRightButton(self, event: QtGui.QMouseEvent) -> bool:
        return event.button() == RIGHT_BUTTON

    def checkRightButtons(self, event: QtGui.QMouseEvent) -> bool:
        return bool(event.buttons() & RIGHT_BUTTON)

    def readWheelStep(self, event: QtGui.QWheelEvent) -> int:
        wheel_delta = event.angleDelta().y()
        if wheel_delta == 0:
            wheel_delta = event.pixelDelta().y()
        if wheel_delta > 0:
            return 1
        if wheel_delta < 0:
            return -1
        return 0

    def readMousePoint(self, event: QtGui.QMouseEvent) -> QtCore.QPoint:
        if hasattr(event, "globalPosition"):
            return event.globalPosition().toPoint()
        return event.globalPos()


def main(argv: list[str] | None = None) -> None:
    validate_class_definitions()
    input_path, pseudo_mask_input, dataset_output_root, video_fps = resolve_runtime_paths(argv)
    rgb_dir, mask_dir, rgb_map, mask_map, keys = prepare_input_output_checked(
        input_path,
        pseudo_mask_input=pseudo_mask_input,
        dataset_output_root=dataset_output_root,
        video_fps=video_fps,
    )

    state = {
        "idx": 0,
        "keys": keys,
        "rgb_map": rgb_map,
        "mask_map": mask_map,

        "dirty": False,
        "suspend": False,

        "active": next(iter(CLASS_DEFINITIONS)),
        "label": 1,
        "brush_b0": 100,
        "brush_b1": 60,
        "zoom": 1.0,
        "tool_mode": "paint",

        # ✅ 실행(세션) 전체에서 누적되는 저장 카운터
        "save_idx": 0,
    }
    for class_id in CLASS_DEFINITIONS:
        state[f"class_{class_id}_opacity"] = 0.85
        state[f"class_{class_id}_visible"] = True

    frame_cache: FrameCache = {}
    frame_futures: FrameFuture = {}
    frame_lock = Lock()
    frame_executor = ThreadPoolExecutor(max_workers=2)

    viewer = napari.Viewer(title="Multi-class Mask Editor")
    autosave_timer = QtCore.QTimer(viewer.window._qt_window)
    autosave_timer.setSingleShot(True)
    autosave_timer.setInterval(AUTOSAVE_DELAY_MS)
    change_timer = QtCore.QTimer(viewer.window._qt_window)
    change_timer.setInterval(CHANGE_SCAN_MS)
    mask_snapshot: np.ndarray | None = None

    img_layer = None
    class_layers: dict[int, object] = {}
    histories = {class_id: History(MAX_HISTORY) for class_id in CLASS_DEFINITIONS}

    # UI refs
    page_title = None
    page_slider = None
    page_spin = None
    class_buttons: dict[int, QtWidgets.QPushButton] = {}
    btnBG = btnFG = None
    btnPaint = btnFill = btnPan = None
    visibility_buttons: dict[int, QtWidgets.QPushButton] = {}
    brush_controls: dict[int, tuple[QtWidgets.QSlider, QtWidgets.QSpinBox]] = {}
    class_summary_label: QtWidgets.QLabel | None = None

    zoom_slider = None
    zoom_spin = None

    # ✅ log UI refs
    # ✅ log UI refs
    log_box: QtWidgets.QTextEdit | None = None
    LOG_MAX_LINES = 8  # ✅ 6줄 넘으면 자동 clear
    def log_append(msg: str):
        """로그 한 줄 추가 + 6줄 초과 시 자동 clear + 자동 스크롤"""
        nonlocal log_box
        if log_box is None:
            return

        # 현재 줄 수 체크 (QTextEdit은 document().blockCount()가 줄 수)
        if log_box.document().blockCount() >= LOG_MAX_LINES:
            log_box.clear()

        log_box.append(msg)

        # 맨 아래로 스크롤
        sb = log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def trimCache() -> None:
        while len(frame_cache) > FRAME_CACHE_LIMIT:
            cache_key = next(iter(frame_cache))
            frame_cache.pop(cache_key, None)

    def readFrame(frame: str) -> FrameData:
        with frame_lock:
            frame_data = frame_cache.pop(frame, None)
            frame_future = frame_futures.pop(frame, None)

        if frame_data is not None:
            return frame_data

        if frame_future is not None:
            try:
                return frame_future.result()
            except Exception:
                pass

        return loadFrame(frame, rgb_map, mask_map)

    def dropFrame(frame: str) -> None:
        with frame_lock:
            frame_cache.pop(frame, None)
            frame_future = frame_futures.pop(frame, None)
        if frame_future is not None:
            frame_future.cancel()

    def cacheFrame(frame: str) -> None:
        with frame_lock:
            if frame in frame_cache or frame in frame_futures:
                return
            try:
                frame_future = frame_executor.submit(loadFrame, frame, rgb_map, mask_map)
            except RuntimeError:
                return
            frame_futures[frame] = frame_future

        def storeFrame(frame_future: Future[FrameData], frame: str = frame) -> None:
            try:
                frame_data = frame_future.result()
            except Exception:
                with frame_lock:
                    if frame_futures.get(frame) is frame_future:
                        frame_futures.pop(frame, None)
                return

            with frame_lock:
                if frame_futures.get(frame) is not frame_future:
                    return
                frame_futures.pop(frame, None)
                frame_cache[frame] = frame_data
                trimCache()

        frame_future.add_done_callback(storeFrame)

    def cacheFrames() -> None:
        index = int(state["idx"])
        for offset in (-1, 1):
            item = index + offset
            if 0 <= item < len(keys):
                cacheFrame(keys[item])

    def active_layer_obj():
        return class_layers.get(int(state["active"]))

    def set_window_title(extra: str = ""):
        try:
            k = keys[state["idx"]]
            star = " *" if state["dirty"] else ""
            class_state = " ".join(
                f"{CLASS_DEFINITIONS[class_id]['name']}"
                f"(vis={int(state[f'class_{class_id}_visible'])})"
                for class_id in CLASS_DEFINITIONS
            )
            base = (
                f"[ACTIVE={state['active']} label={state['label']} tool={state['tool_mode']}] "
                f"{class_state} "
                f"brush(erase={state['brush_b0']},paint={state['brush_b1']}) "
                f"zoom={state['zoom']:.2f} "
                f"- [{state['idx']+1}/{len(keys)}] {k}{star}"
            )
            if extra:
                base += f"   {extra}"

            qt_window = getattr(viewer.window, "_qt_window", None)
            if qt_window is not None:
                QtCore.QTimer.singleShot(0, lambda: qt_window.setWindowTitle(base))
        except Exception as e:
            print("[TITLE] set_window_title error:", type(e).__name__, e)

    def refresh_page_ui():
        if page_title is None:
            return
        v = state["idx"] + 1
        page_title.setText(f"Page  {v}/{len(keys)}" + ("  *unsaved" if state["dirty"] else ""))
        page_slider.blockSignals(True)
        page_spin.blockSignals(True)
        page_slider.setValue(v)
        page_spin.setValue(v)
        page_slider.blockSignals(False)
        page_spin.blockSignals(False)

    def apply_zoom(z: float):
        z = float(np.clip(z, ZOOM_MIN, ZOOM_MAX))
        state["zoom"] = z
        try:
            viewer.camera.zoom = z
        except Exception:
            pass
        set_window_title()

    def apply_colormaps():
        for class_id, layer in class_layers.items():
            color = CLASS_DEFINITIONS[class_id]["color"]
            cmap = DirectLabelColormap(
                color_dict={None: (0, 0, 0, 0), 0: (0, 0, 0, 0), 1: color}
            )
            layer.colormap = cmap
            layer.opacity = float(np.clip(state[f"class_{class_id}_opacity"], 0.0, 1.0))
            try:
                layer.blending = "translucent"
            except Exception:
                pass
            layer.visible = bool(state[f"class_{class_id}_visible"])
            try:
                layers = viewer.layers
                i = layers.index(layer)
                layers.move(i, len(layers) - 1)
            except Exception:
                pass

    def _class_opacity_changed(class_id: int, _event=None):
        layer = class_layers.get(class_id)
        if state["suspend"] or layer is None:
            return
        state[f"class_{class_id}_opacity"] = float(np.clip(layer.opacity, 0.0, 1.0))
        set_window_title()

    def _class_visible_changed(class_id: int, _event=None):
        layer = class_layers.get(class_id)
        if state["suspend"] or layer is None:
            return
        state[f"class_{class_id}_visible"] = bool(layer.visible)
        sync_buttons()
        set_window_title()

    def refresh_class_name_ui(class_id: int) -> None:
        name = str(CLASS_DEFINITIONS[class_id]["name"])
        button = class_buttons.get(class_id)
        if button is not None:
            button.setText(f"Active: {name} ({class_id})")
        visibility_button = visibility_buttons.get(class_id)
        if visibility_button is not None:
            visibility_button.setText(f"Show {name}")
        if class_summary_label is not None:
            summary = ", ".join(
                f"{cid}={definition['name']}"
                for cid, definition in CLASS_DEFINITIONS.items()
            )
            class_summary_label.setText(f"Saved mask values: background=0, {summary}.")

    def _class_name_changed(class_id: int, _event=None) -> None:
        """Keep user-entered Napari layer names in the class UI for this session."""
        layer = class_layers.get(class_id)
        if layer is None:
            return
        name = str(layer.name).strip()
        default_suffix = f" (class {class_id})"
        if name.endswith(default_suffix):
            name = name[: -len(default_suffix)].strip()
        if not name:
            return
        CLASS_DEFINITIONS[class_id]["name"] = name
        refresh_class_name_ui(class_id)
        write_dataset_metadata(mask_dir.parent)
        set_window_title()

    def apply_tool_mode():
        layer = active_layer_obj()
        if layer is None:
            return
        try:
            viewer.layers.selection.active = layer
        except Exception:
            pass
        try:
            if state["tool_mode"] in ("paint", "pan_zoom"):
                layer.mode = str(state["tool_mode"])
            else:
                layer.mode = "pan_zoom"
        except Exception:
            pass

    def brush_value_for(lbl: int) -> int:
        return int(state[f"brush_b{lbl}"])

    def remember_current_brush_size():
        if state.get("suspend"):
            return
        layer = active_layer_obj()
        if layer is None:
            return
        try:
            v = int(round(float(layer.brush_size)))
        except Exception:
            return
        v = int(np.clip(v, BRUSH_MIN, BRUSH_MAX))
        state[f"brush_b{state['label']}"] = v
        sync_brush_ui()

    def set_brush_value_for(lbl: int, v: int):
        v = int(np.clip(int(v), BRUSH_MIN, BRUSH_MAX))
        state[f"brush_b{lbl}"] = v
        if state["label"] == lbl:
            layer = active_layer_obj()
            if layer is not None:
                try:
                    layer.brush_size = v
                except Exception:
                    pass
        set_window_title()

    def adjustBrushRange(direction: int) -> bool:
        if state.get("tool_mode") == "pan_zoom":
            return False

        label = int(state["label"])
        delta = BRUSH_WHEEL_STEP if direction > 0 else -BRUSH_WHEEL_STEP
        size = brush_value_for(label) + delta
        set_brush_value_for(label, size)
        sync_brush_ui()
        return True

    def adjustImageZoom(direction: int) -> bool:
        zoom = float(state["zoom"])
        if direction > 0:
            zoom *= ZOOM_WHEEL_RATIO
        else:
            zoom /= ZOOM_WHEEL_RATIO
        apply_zoom(zoom)
        return True

    def adjustImagePan(delta_x: int, delta_y: int) -> bool:
        try:
            zoom = float(viewer.camera.zoom)
            center = [float(value) for value in viewer.camera.center]
        except Exception:
            return False

        if zoom <= 0 or len(center) < 2:
            return False

        center[-2] -= float(delta_y) / zoom
        center[-1] -= float(delta_x) / zoom
        try:
            viewer.camera.center = tuple(center)
        except Exception:
            return False
        set_window_title()
        return True

    def sync_buttons():
        for class_id, button in class_buttons.items():
            button.setChecked(state["active"] == class_id)
        if btnBG is not None:
            btnBG.setChecked(state["label"] == 0)
            btnFG.setChecked(state["label"] == 1)
        if btnPaint is not None:
            btnPaint.setChecked(state["tool_mode"] == "paint")
        if btnFill is not None:
            btnFill.setChecked(state["tool_mode"] == "fill")
        for class_id, button in visibility_buttons.items():
            button.setChecked(state[f"class_{class_id}_visible"])

    def sync_brush_ui():
        for label, (slider, spinbox) in brush_controls.items():
            v = brush_value_for(label)
            slider.blockSignals(True); spinbox.blockSignals(True)
            slider.setValue(v); spinbox.setValue(v)
            slider.blockSignals(False); spinbox.blockSignals(False)

    def apply_active_state_to_layer():
        layer = active_layer_obj()
        if layer is None:
            return
        apply_tool_mode()
        try:
            layer.selected_label = int(state["label"])
        except Exception:
            pass
        try:
            layer.brush_size = brush_value_for(int(state["label"]))
        except Exception:
            pass
        sync_buttons()
        set_window_title()

    def restore_editor_state():
        sync_brush_ui()
        apply_active_state_to_layer()

    def _label_layer_brush_size_changed(class_id: int):
        if state.get("suspend"):
            return
        layer = class_layers.get(class_id)
        if layer is None:
            return
        if state.get("active") != class_id:
            return
        try:
            v = int(round(float(layer.brush_size)))
        except Exception:
            return
        v = int(np.clip(v, BRUSH_MIN, BRUSH_MAX))
        state[f"brush_b{state['label']}"] = v
        sync_brush_ui()
        set_window_title()

    def fill_active_class_at_current_point(position) -> bool:
        class_id = int(state["active"])
        layer = class_layers.get(class_id)
        if layer is None:
            return False
        if position is None or len(position) < 2:
            return False

        data = np.asarray(layer.data)
        y = int(round(float(position[0])))
        x = int(round(float(position[1])))
        if not (0 <= y < data.shape[0] and 0 <= x < data.shape[1]):
            return False

        filled = flood_fill_region(data, y, x, int(state["label"]))
        if np.array_equal(filled, data):
            return False

        history = histories[class_id]
        history.push(data)
        history.suspend = True
        try:
            layer.data = filled.astype(np.uint8)
        finally:
            history.suspend = False

        markChange(f"class {class_id}")
        apply_colormaps()
        refresh_page_ui()
        set_window_title(extra=f"(class {class_id} fill @ x={x}, y={y}, label={state['label']})")
        return True

    def on_class_mouse_drag(layer, event):
        if state.get("tool_mode") != "fill":
            return
        filled = fill_active_class_at_current_point(getattr(event, "position", None))
        if filled:
            event.handled = True
        return

    def set_active_layer(class_id: int, *_args):
        class_id = int(class_id)
        if class_id not in CLASS_DEFINITIONS:
            return
        remember_current_brush_size()
        state["active"] = class_id
        apply_active_state_to_layer()
        sync_brush_ui()

    def _viewer_active_layer_changed(event=None) -> None:
        """Synchronize state when a class is selected in Napari's layer list."""
        selected_layer = getattr(event, "value", None)
        if selected_layer is None:
            selected_layer = viewer.layers.selection.active
        selected_class_id = next(
            (
                class_id
                for class_id, layer in class_layers.items()
                if layer is selected_layer
            ),
            None,
        )
        if selected_class_id is None or selected_class_id == state["active"]:
            return
        remember_current_brush_size()
        state["active"] = selected_class_id
        apply_active_state_to_layer()
        sync_brush_ui()

    viewer.layers.selection.events.active.connect(_viewer_active_layer_changed)

    def connect_class_layer_events(class_id: int, layer) -> None:
        layer.events.data.connect(
            lambda _event=None, cid=class_id: markChange(f"class {cid}")
        )
        layer.events.paint.connect(
            lambda _event=None, cid=class_id: markChange(f"class {cid}")
        )
        layer.events.opacity.connect(
            lambda _event=None, cid=class_id: _class_opacity_changed(cid)
        )
        layer.events.visible.connect(
            lambda _event=None, cid=class_id: _class_visible_changed(cid)
        )
        layer.events.name.connect(
            lambda _event=None, cid=class_id: _class_name_changed(cid)
        )
        layer.events.brush_size.connect(
            lambda _event=None, cid=class_id: _label_layer_brush_size_changed(cid)
        )
        layer.mouse_drag_callbacks.append(on_class_mouse_drag)

    def add_new_class(*_args) -> None:
        if img_layer is None:
            return
        class_id = max(CLASS_DEFINITIONS, default=0) + 1
        if class_id > 255:
            QtWidgets.QMessageBox.warning(
                viewer.window._qt_window,
                "Class limit",
                "A uint8 segmentation mask supports class IDs only up to 255.",
            )
            return
        name, accepted = QtWidgets.QInputDialog.getText(
            viewer.window._qt_window,
            "Add class",
            f"Name for class {class_id}:",
            text=f"class_{class_id}",
        )
        name = name.strip()
        if not accepted or not name:
            return

        CLASS_DEFINITIONS[class_id] = {
            "name": name,
            "color": default_class_color(class_id),
        }
        state[f"class_{class_id}_opacity"] = 0.85
        state[f"class_{class_id}_visible"] = True
        histories[class_id] = History(MAX_HISTORY)

        state["suspend"] = True
        try:
            layer = viewer.add_labels(
                np.zeros(img_layer.data.shape[:2], dtype=np.uint8),
                name=f"{name} (class {class_id})",
            )
            class_layers[class_id] = layer
            connect_class_layer_events(class_id, layer)
        finally:
            state["suspend"] = False

        class_button = QtWidgets.QPushButton(f"Active: {name} ({class_id})")
        class_button.setCheckable(True)
        grp_layer.addButton(class_button)
        class_button.clicked.connect(
            lambda _checked=False, cid=class_id: set_active_layer(cid)
        )
        class_buttons[class_id] = class_button
        layer_row.addWidget(class_button)

        visibility_button = QtWidgets.QPushButton(f"Show {name}")
        visibility_button.setCheckable(True)
        visibility_button.setChecked(True)
        visibility_button.toggled.connect(
            lambda checked, cid=class_id: toggle_class_visibility(cid, checked)
        )
        visibility_buttons[class_id] = visibility_button
        vis_row.addWidget(visibility_button)

        if class_id <= 12:
            bindMany([f"F{class_id}"], lambda cid=class_id: set_active_layer(cid))
        refresh_class_name_ui(class_id)
        write_dataset_metadata(mask_dir.parent)
        apply_colormaps()
        set_active_layer(class_id)

    def set_selected_label(lbl: int, *_args):
        # Store the current tool's size before switching.  The new tool then
        # restores its own saved size instead of inheriting the old one.
        remember_current_brush_size()
        state["label"] = 0 if int(lbl) == 0 else 1
        apply_active_state_to_layer()

    def set_tool_mode(mode: str, *_args):
        remember_current_brush_size()
        normalized = str(mode).lower()
        if normalized not in {"paint", "pan_zoom", "fill"}:
            normalized = "paint"
        state["tool_mode"] = normalized
        apply_active_state_to_layer()

    def toggle_tool_mode(*_args):
        cycle = [
            ("paint", 0),
            ("paint", 1),
            ("pan_zoom", None),
        ]
        current = (state.get("tool_mode"), state.get("label"))
        try:
            idx = cycle.index(current)
        except ValueError:
            idx = -1
        next_mode, next_label = cycle[(idx + 1) % len(cycle)]
        state["tool_mode"] = next_mode
        if next_label is not None:
            state["label"] = int(next_label)
        apply_active_state_to_layer()

    # ======================
    # ✅ napari 기본 Undo/Redo를 "키입력으로" 실행
    # ======================
    def _send_shortcut(key: int, mods: QtCore.Qt.KeyboardModifiers):
        qt_viewer = getattr(viewer.window, "_qt_viewer", None)
        target = qt_viewer if qt_viewer is not None else viewer.window._qt_window

        try:
            target.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        except Exception:
            try:
                target.setFocus()
            except Exception:
                pass

        press = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key, mods)
        release = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyRelease, key, mods)
        QtWidgets.QApplication.postEvent(target, press)
        QtWidgets.QApplication.postEvent(target, release)

    def undo_via_ctrl_z(*_args):
        _send_shortcut(QtCore.Qt.Key.Key_Z, QtCore.Qt.KeyboardModifier.ControlModifier)

    def redo_via_ctrl_shift_z(*_args):
        _send_shortcut(
            QtCore.Qt.Key.Key_Z,
            QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
        )

    # ======================
    # History hooks (snapshots) - 유지
    # ======================
    def syncMaskSnapshot() -> None:
        nonlocal mask_snapshot
        if set(class_layers) != set(CLASS_DEFINITIONS):
            mask_snapshot = None
            return
        mask_snapshot = compose_multiclass_mask(
            {class_id: layer.data for class_id, layer in class_layers.items()}
        )

    def checkMaskChange() -> bool:
        if set(class_layers) != set(CLASS_DEFINITIONS):
            return False
        if mask_snapshot is None:
            return False

        mask_data = compose_multiclass_mask(
            {class_id: layer.data for class_id, layer in class_layers.items()}
        )
        shape_match = mask_data.shape == mask_snapshot.shape
        data_match = shape_match and np.array_equal(mask_data, mask_snapshot)
        return not data_match

    def scanMaskChange() -> None:
        if state["suspend"] or set(class_layers) != set(CLASS_DEFINITIONS):
            return
        if mask_snapshot is None:
            syncMaskSnapshot()
            return
        if not checkMaskChange():
            return
        markChange("multi-class mask")

    def scheduleAutosave() -> None:
        autosave_timer.start(AUTOSAVE_DELAY_MS)

    def markChange(layer: str) -> None:
        if state["suspend"]:
            return
        state["dirty"] = True
        syncMaskSnapshot()
        scheduleAutosave()
        set_window_title()
        refresh_page_ui()

    # ======================
    # IO / navigation
    # ======================
    def writeMask() -> Path | None:
        if set(class_layers) != set(CLASS_DEFINITIONS):
            return None
        frame_key = keys[state["idx"]]
        mask_path = mask_map[frame_key]
        mask_data = compose_multiclass_mask(
            {class_id: layer.data for class_id, layer in class_layers.items()}
        )
        try:
            iio.imwrite(mask_path, mask_data)
            write_mask_preview(mask_path, mask_data)
        except Exception as exc:
            raise RuntimeError(f"Mask save failed: {mask_path}") from exc
        write_dataset_metadata(mask_dir.parent)
        dropFrame(frame_key)
        syncMaskSnapshot()
        state["dirty"] = False
        return mask_path

    def saveMask(*_args: object) -> None:
        if autosave_timer.isActive():
            autosave_timer.stop()
        mask_path = writeMask()
        if mask_path is None:
            return

        # ✅ 세션 전체 누적 save idx + 로그 추가
        state["save_idx"] += 1
        log_append(f"Saved {state['save_idx']}: {mask_path.name}")

        set_window_title(extra=f"(saved classes 1/2: {mask_path.name})")
        refresh_page_ui()

    def flushAutosave(alert: bool = False) -> bool:
        if not state["dirty"]:
            return True
        try:
            saveMask()
            return True
        except Exception as exc:
            log_append(f"Auto-save failed: {type(exc).__name__}: {exc}")
            if alert:
                QtWidgets.QMessageBox.critical(
                    viewer.window._qt_window,
                    "Auto-save failed",
                    f"현재 프레임 저장 중 오류가 발생해서 이동을 취소합니다.\n\n{type(exc).__name__}: {exc}",
                )
            set_window_title(extra=f"(AUTO SAVE ERROR: {type(exc).__name__})")
            return False

    def openFrame() -> None:
        nonlocal img_layer

        if autosave_timer.isActive():
            autosave_timer.stop()
        k = keys[state["idx"]]
        rgb, class_masks = readFrame(k)

        for history in histories.values():
            history.clear()

        state["suspend"] = True
        try:
            if img_layer is None:
                img_layer = viewer.add_image(
                    rgb, name="RGB",
                    rgb=(rgb.ndim == 3 and rgb.shape[-1] in (3, 4)),
                )
            else:
                img_layer.data = rgb

            for class_id, definition in CLASS_DEFINITIONS.items():
                layer = class_layers.get(class_id)
                if layer is None:
                    layer = viewer.add_labels(
                        class_masks[class_id],
                        name=f"{definition['name']} (class {class_id})",
                    )
                    class_layers[class_id] = layer
                    connect_class_layer_events(class_id, layer)
                else:
                    layer.data = class_masks[class_id]
        finally:
            state["suspend"] = False

        state["dirty"] = False
        syncMaskSnapshot()
        apply_colormaps()
        apply_zoom(state["zoom"])
        restore_editor_state()
        QtCore.QTimer.singleShot(0, restore_editor_state)
        QtCore.QTimer.singleShot(0, cacheFrames)
        set_window_title()
        refresh_page_ui()

    def gotoFrame(frame_index: int) -> None:
        frame_index = int(np.clip(frame_index, 0, len(keys) - 1))
        source_index = state["idx"]
        state["idx"] = frame_index
        try:
            openFrame()
        except Exception as exc:
            state["idx"] = source_index
            set_window_title(extra=f"(OPEN ERROR: {type(exc).__name__})")
            LOGGER.exception("Open frame failed")

    def autosaveBeforeNavigation() -> bool:
        if not class_layers:
            return True
        scanMaskChange()
        if autosave_timer.isActive():
            autosave_timer.stop()
        if not state["dirty"]:
            return True
        return flushAutosave(alert=True)

    def navigateFrame(frame_index: int) -> None:
        remember_current_brush_size()
        if not autosaveBeforeNavigation():
            return
        gotoFrame(frame_index)

    def nextFrame(*_args: object) -> None:
        navigateFrame(state["idx"] + 1)

    def prevFrame(*_args: object) -> None:
        navigateFrame(state["idx"] - 1)

    # ======================
    # ops
    # ======================
    def copy_previous_to_active_class(*_args):
        class_id = int(state["active"])
        layer = class_layers.get(class_id)
        if img_layer is None or layer is None:
            return
        prev_idx = (state["idx"] - 1) % len(keys)
        prev_key = keys[prev_idx]
        prev_mask = load_multiclass_mask(mask_map[prev_key], img_layer.data.shape[:2])

        history = histories[class_id]
        history.push(np.asarray(layer.data))
        history.suspend = True
        try:
            layer.data = (prev_mask == class_id).astype(np.uint8)
        finally:
            history.suspend = False

        markChange(f"class {class_id}")
        apply_colormaps()
        set_window_title(extra=f"(copied prev {prev_key} -> class {class_id})")

    def convert_active_class_to_first(*_args):
        """Move the active class into the first configured class."""
        source_id = int(state["active"])
        target_id = next(iter(CLASS_DEFINITIONS))
        if source_id == target_id:
            return
        source = class_layers.get(source_id)
        target = class_layers.get(target_id)
        if source is None or target is None:
            return
        histories[source_id].push(np.asarray(source.data))
        histories[target_id].push(np.asarray(target.data))
        for history in (histories[source_id], histories[target_id]):
            history.suspend = True
        try:
            target.data = ((np.asarray(target.data) > 0) | (np.asarray(source.data) > 0)).astype(np.uint8)
            source.data = np.zeros_like(source.data, dtype=np.uint8)
        finally:
            for history in (histories[source_id], histories[target_id]):
                history.suspend = False
        markChange(f"class {source_id}")
        apply_colormaps()
        refresh_page_ui()


    def delete_current_pair(*_args):
        if not state["keys"]:
            return
        if len(state["keys"]) <= 1:
            QtWidgets.QMessageBox.warning(
                viewer.window._qt_window,
                "Delete blocked",
                "마지막 1개 프레임은 삭제할 수 없어요."
            )
            return

        k = state["keys"][state["idx"]]
        rgb_path = state["rgb_map"].get(k)
        mask_path = state["mask_map"].get(k)

        errors = []
        for p in [rgb_path, mask_path]:
            try:
                if p is not None and Path(p).exists():
                    Path(p).unlink()
            except Exception as e:
                errors.append(f"{p}: {e}")

        if errors:
            QtWidgets.QMessageBox.critical(
                viewer.window._qt_window,
                "Delete failed",
                "삭제 중 오류가 발생했어요.\n\n" + "\n".join(errors)
            )
            set_window_title(extra="(delete failed)")
            return

        old_idx = state["idx"]
        dropFrame(k)
        state["rgb_map"].pop(k, None)
        state["mask_map"].pop(k, None)
        state["keys"].pop(old_idx)

        for history in histories.values():
            history.clear()
        state["dirty"] = False

        if old_idx >= len(state["keys"]):
            state["idx"] = max(0, len(state["keys"]) - 1)
        else:
            state["idx"] = old_idx

        log_append(f"Deleted frame: {k}")
        if rgb_path is not None:
            log_append(f"RGB deleted: {Path(rgb_path).name}")
        if mask_path is not None:
            log_append(f"MASK deleted: {Path(mask_path).name}")

        try:
            openFrame()
            set_window_title(extra=f"(deleted: {k})")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                viewer.window._qt_window,
                "Open failed",
                f"삭제 후 다음 프레임을 여는 중 오류가 발생했어요.\n\n{type(e).__name__}: {e}"
            )

    # ======================
    # visibility toggles
    # ======================
    def toggle_class_visibility(class_id: int, checked: bool, *_args):
        state[f"class_{class_id}_visible"] = bool(checked)
        layer = class_layers.get(class_id)
        if layer is not None:
            layer.visible = bool(checked)
        sync_buttons()
        set_window_title()

    # ======================
    # UI
    # ======================
    dock = QtWidgets.QWidget()
    L = QtWidgets.QVBoxLayout(dock)
    L.setContentsMargins(10, 10, 10, 10)
    L.setSpacing(10)

    # Prev/Next
    nav = QtWidgets.QHBoxLayout()
    bprev = QtWidgets.QPushButton("Prev")
    bnext = QtWidgets.QPushButton("Next")
    bprev.clicked.connect(prevFrame)
    bnext.clicked.connect(nextFrame)
    nav.addWidget(bprev); nav.addWidget(bnext)
    L.addLayout(nav)

    # Page slider
    page_title = QtWidgets.QLabel("Page")
    L.addWidget(page_title)
    page_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    page_slider.setMinimum(1); page_slider.setMaximum(len(keys))
    page_slider.setSingleStep(1); page_slider.setPageStep(10)
    page_spin = QtWidgets.QSpinBox()
    page_spin.setMinimum(1); page_spin.setMaximum(len(keys))
    page_spin.setSingleStep(1)
    L.addWidget(page_slider); L.addWidget(page_spin)

    def changePage(value: int) -> None:
        navigateFrame(int(value) - 1)

    page_slider.valueChanged.connect(changePage)
    page_spin.valueChanged.connect(changePage)

    # Save
    save_btn = QtWidgets.QPushButton("SAVE multi-class mask")
    save_btn.clicked.connect(saveMask)
    L.addWidget(save_btn)

    delete_btn = QtWidgets.QPushButton("DELETE current frame + mask")
    delete_btn.clicked.connect(delete_current_pair)
    L.addWidget(delete_btn)

    # Undo/Redo
    ur = QtWidgets.QHBoxLayout()
    undo_btn = QtWidgets.QPushButton("Undo")
    redo_btn = QtWidgets.QPushButton("Redo")
    undo_btn.clicked.connect(undo_via_ctrl_z)
    redo_btn.clicked.connect(redo_via_ctrl_shift_z)
    ur.addWidget(undo_btn); ur.addWidget(redo_btn)
    L.addLayout(ur)

    # Class buttons are created from CLASS_DEFINITIONS.
    layer_row = QtWidgets.QHBoxLayout()
    grp_layer = QtWidgets.QButtonGroup()
    grp_layer.setExclusive(True)
    for class_id, definition in CLASS_DEFINITIONS.items():
        button = QtWidgets.QPushButton(f"Active: {definition['name']} ({class_id})")
        button.setCheckable(True)
        grp_layer.addButton(button)
        button.clicked.connect(lambda _checked=False, cid=class_id: set_active_layer(cid))
        class_buttons[class_id] = button
        layer_row.addWidget(button)
    add_class_button = QtWidgets.QPushButton("+ Add class")
    add_class_button.clicked.connect(add_new_class)
    layer_row.addWidget(add_class_button)
    L.addLayout(layer_row)

    tool_row = QtWidgets.QHBoxLayout()
    btnPaint = QtWidgets.QPushButton("Brush")
    btnFill = QtWidgets.QPushButton("Paint")
    btnPaint.setCheckable(True); btnFill.setCheckable(True)
    grp_tool = QtWidgets.QButtonGroup()
    grp_tool.setExclusive(True)
    grp_tool.addButton(btnPaint); grp_tool.addButton(btnFill)
    btnPaint.clicked.connect(lambda *_: set_tool_mode("paint"))
    btnFill.clicked.connect(lambda *_: set_tool_mode("fill"))
    tool_row.addWidget(btnPaint); tool_row.addWidget(btnFill)
    L.addLayout(tool_row)
    class_summary = ", ".join(
        f"{class_id}={definition['name']}" for class_id, definition in CLASS_DEFINITIONS.items()
    )
    class_summary_label = QtWidgets.QLabel(
        f"Saved mask values: background=0, {class_summary}."
    )
    L.addWidget(class_summary_label)

    # Visibility toggles
    vis_row = QtWidgets.QHBoxLayout()
    for class_id, definition in CLASS_DEFINITIONS.items():
        button = QtWidgets.QPushButton(f"Show {definition['name']}")
        button.setCheckable(True)
        button.setChecked(True)
        button.toggled.connect(
            lambda checked, cid=class_id: toggle_class_visibility(cid, checked)
        )
        visibility_buttons[class_id] = button
        vis_row.addWidget(button)
    L.addLayout(vis_row)

    # Label buttons (0/1)
    label_row = QtWidgets.QHBoxLayout()
    btnBG = QtWidgets.QPushButton("Label 0 (bg)")
    btnFG = QtWidgets.QPushButton("Label 1 (fg)")
    btnBG.setCheckable(True); btnFG.setCheckable(True)
    grp_lbl = QtWidgets.QButtonGroup()
    grp_lbl.setExclusive(True)
    grp_lbl.addButton(btnBG); grp_lbl.addButton(btnFG)
    btnBG.clicked.connect(lambda *_: set_selected_label(0))
    btnFG.clicked.connect(lambda *_: set_selected_label(1))
    label_row.addWidget(btnBG); label_row.addWidget(btnFG)
    L.addLayout(label_row)

    # Ops
    copy_btn = QtWidgets.QPushButton("Copy previous → active class")
    copy_btn.clicked.connect(copy_previous_to_active_class)
    L.addWidget(copy_btn)

    merge_btn = QtWidgets.QPushButton("Convert active class → first class")
    merge_btn.clicked.connect(convert_active_class_to_first)
    L.addWidget(merge_btn)

    # Brush blocks
    def add_brush_block(target_layout, title: str, init: int):
        lbl = QtWidgets.QLabel(title)
        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        s.setMinimum(BRUSH_MIN); s.setMaximum(BRUSH_MAX)
        s.setSingleStep(BRUSH_STEP); s.setPageStep(BRUSH_STEP * 5)
        sp = QtWidgets.QSpinBox()
        sp.setMinimum(BRUSH_MIN); sp.setMaximum(BRUSH_MAX)
        sp.setSingleStep(BRUSH_STEP)
        target_layout.addWidget(lbl); target_layout.addWidget(s); target_layout.addWidget(sp)
        s.setValue(init); sp.setValue(init)
        return lbl, s, sp

    def link_brush(s, sp, lbl: int):
        def _from_slider(v):
            sp.setValue(v)
            set_brush_value_for(lbl, v)
        def _from_spin(v):
            s.setValue(v)
            set_brush_value_for(lbl, v)
        s.valueChanged.connect(_from_slider)
        sp.valueChanged.connect(_from_spin)

    L.addWidget(QtWidgets.QLabel("Common brush sizes"))
    for label, text in ((0, "Eraser"), (1, "Brush")):
        _control_label, slider, spinbox = add_brush_block(
            L,
            text,
            state[f"brush_b{label}"],
        )
        link_brush(slider, spinbox, label)
        brush_controls[label] = (slider, spinbox)

    # Zoom
    zlbl = QtWidgets.QLabel("Zoom")
    zoom_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    zoom_slider.setMinimum(int(round(ZOOM_MIN / ZOOM_STEP)))
    zoom_slider.setMaximum(int(round(ZOOM_MAX / ZOOM_STEP)))
    zoom_slider.setSingleStep(1); zoom_slider.setPageStep(10)
    zoom_spin = QtWidgets.QDoubleSpinBox()
    zoom_spin.setMinimum(ZOOM_MIN); zoom_spin.setMaximum(ZOOM_MAX)
    zoom_spin.setSingleStep(ZOOM_STEP); zoom_spin.setDecimals(2)
    L.addWidget(zlbl); L.addWidget(zoom_slider); L.addWidget(zoom_spin)

    def sync_zoom_ui(z: float):
        z = float(np.clip(z, ZOOM_MIN, ZOOM_MAX))
        v = int(round(z / ZOOM_STEP))
        zoom_slider.blockSignals(True); zoom_spin.blockSignals(True)
        zoom_slider.setValue(v); zoom_spin.setValue(z)
        zoom_slider.blockSignals(False); zoom_spin.blockSignals(False)

    zoom_slider.valueChanged.connect(lambda v: (sync_zoom_ui(float(v) * ZOOM_STEP), apply_zoom(float(v) * ZOOM_STEP)))
    zoom_spin.valueChanged.connect(lambda z: (sync_zoom_ui(float(z)), apply_zoom(float(z))))

    try:
        def _camera_zoom_changed(_event=None):
            z = float(viewer.camera.zoom)
            state["zoom"] = float(np.clip(z, ZOOM_MIN, ZOOM_MAX))
            sync_zoom_ui(state["zoom"])
            set_window_title()
        viewer.camera.events.zoom.connect(_camera_zoom_changed)
    except Exception:
        pass

    # ✅ 로그 박스 (스크롤 자동 생김)
    log_dock = QtWidgets.QWidget()
    LOG_L = QtWidgets.QVBoxLayout(log_dock)
    LOG_L.setContentsMargins(10, 10, 10, 10)
    LOG_L.setSpacing(10)

    log_box = QtWidgets.QTextEdit()
    log_box.setReadOnly(True)
    log_box.setMinimumHeight(130)
    LOG_L.addWidget(log_box)

    dock_scroll = QtWidgets.QScrollArea()
    dock_scroll.setWidgetResizable(True)
    dock_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    dock_scroll.setWidget(dock)

    viewer.window.add_dock_widget(log_dock, name="LOG", area="left")
    viewer.window.add_dock_widget(dock_scroll, name="TOOLS", area="right")

    # ======================
    # Key bindings
    # ======================
    def bindMany(key_names: list[str], action: NavAction) -> None:
        def run_unless_typing(*_args: object) -> None:
            if text_input_has_focus():
                return
            action()

        for key_name in key_names:
            viewer.bind_key(key_name, overwrite=True)(run_unless_typing)

    application = QtWidgets.QApplication.instance()
    qt_window = viewer.window._qt_window
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    canvas = getattr(qt_viewer, "canvas", None)
    canvas_widget = getattr(canvas, "native", None)
    if not isinstance(canvas_widget, QtWidgets.QWidget):
        canvas_widget = None

    input_filter = InputFilter(
        qt_window,
        canvas_widget,
        nextFrame,
        prevFrame,
        adjustBrushRange,
        adjustImageZoom,
        adjustImagePan,
    )
    if application is not None:
        application.installEventFilter(input_filter)
        state["input_filter"] = input_filter

    bindMany(["Q", "Left", "Up"], prevFrame)
    bindMany(["W", "Enter", "Right", "Down"], nextFrame)
    bindMany(["PageDown", "PgDown", "Next"], nextFrame)
    bindMany(["PageUp", "PgUp", "Prior"], prevFrame)
    bindMany(["Control-S"], saveMask)
    bindMany(["Control-Delete"], delete_current_pair)

    bindMany(["End"], undo_via_ctrl_z)
    bindMany(["Delete"], redo_via_ctrl_shift_z)

    for class_id in sorted(CLASS_DEFINITIONS):
        if 1 <= class_id <= 12:
            bindMany([f"F{class_id}"], lambda cid=class_id: set_active_layer(cid))

    bindMany(["A"], lambda *_: set_selected_label(0))
    bindMany(["S"], lambda *_: set_selected_label(1))
    bindMany(["X"], lambda *_: set_tool_mode("paint"))
    bindMany(["B"], lambda *_: set_tool_mode("paint"))
    bindMany(["F"], lambda *_: set_tool_mode("fill"))
    bindMany(["Z"], lambda *_: set_tool_mode("pan_zoom"))
    bindMany(["/", "Slash"], toggle_tool_mode)

    # start
    autosave_timer.timeout.connect(flushAutosave)
    change_timer.timeout.connect(scanMaskChange)
    change_timer.start()

    openFrame()
    sync_zoom_ui(state["zoom"])
    sync_brush_ui()
    sync_buttons()
    apply_active_state_to_layer()
    refresh_page_ui()
    log_append("Ready")
    log_append(f"RGB dir: {rgb_dir.name}")
    log_append(f"MASK dir: {mask_dir.name}")

    try:
        napari.run()
    finally:
        if autosave_timer.isActive():
            autosave_timer.stop()
        try:
            if checkMaskChange():
                state["dirty"] = True
            if state["dirty"]:
                mask_path = writeMask()
                if mask_path is not None:
                    LOGGER.info("Final mask saved: %s", mask_path)
        except Exception:
            LOGGER.exception("Final mask save failed")
        frame_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
