# main.py
# pip install "napari[all]" imageio imageio-ffmpeg numpy qtpy tqdm

from __future__ import annotations
import argparse
import logging
import math
from collections import deque
from collections.abc import Callable
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

LOGGER = logging.getLogger(__name__)

# =========================
# input / output paths
# =========================
# 1) 이미지 폴더를 바로 쓰고 싶으면 폴더 경로 지정
# 2) 비디오를 쓰고 싶으면 mp4/avi/... 파일 경로 지정
INPUT_PATH = Path(r"D:\workspace\LabelingTool\__raw_data\c1_mono_cropped.png")

# 비디오 입력일 때 프레임 이미지 / 마스크를 저장할 기준 폴더
# 예: INPUT_PATH가 ex1.mp4 이면
#   AUTO_OUTPUT_ROOT/ex1/images
#   AUTO_OUTPUT_ROOT/ex1/masks
# 가 자동 생성됨
AUTO_OUTPUT_ROOT = Path(r"D:\workspace\LabelingTool\output")

# 이미지 폴더 입력일 때 사용할 마스크 폴더
# None이면 자동으로 INPUT_PATH의 형제 폴더에
# "<입력폴더명>_masks" 를 생성해서 사용
MASK_DIR = Path(r"D:\workspace\LabelingTool\pseudo_labeller\processed")

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
BRUSH_WHEEL_STEP = 5

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
FrameData: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]
FrameCache: TypeAlias = dict[str, FrameData]
FrameFuture: TypeAlias = dict[str, Future[FrameData]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an image file, image folder, or video file in the napari labeling tool."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Image file, image folder, or video file path. Defaults to INPUT_PATH in this file.",
    )
    parser.add_argument(
        "--mask-dir",
        dest="mask_dir",
        help="Mask directory to use when the input is an image folder.",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        help="Root directory for extracted video frames and masks.",
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
    input_path = _optional_path(args.input_path) or INPUT_PATH
    mask_dir = _optional_path(args.mask_dir) if args.mask_dir is not None else MASK_DIR
    output_root = _optional_path(args.output_root) if args.output_root is not None else AUTO_OUTPUT_ROOT
    return Path(input_path).expanduser(), mask_dir, output_root, args.fps


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


def load_mask255_as_bin01(mask_path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    """0/255 마스크 -> 0/1"""
    m = iio.imread(mask_path)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != shape_hw:
        raise ValueError(f"Shape mismatch: RGB {shape_hw} vs MASK {m.shape[:2]} ({mask_path.name})")
    return (m > 0).astype(np.uint8)


def loadFrame(
    frame: str,
    rgb_map: dict[str, Path],
    mask_map: dict[str, Path],
) -> FrameData:
    try:
        image = load_rgb(rgb_map[frame])
        mask = load_mask255_as_bin01(mask_map[frame], image.shape[:2])
    except Exception as exc:
        raise RuntimeError(f"Frame load failed: {frame}") from exc
    empty_mask = np.zeros_like(mask, dtype=np.uint8)
    return image, mask, empty_mask


def bin01_to_mask255(bin01: np.ndarray) -> np.ndarray:
    """0/1 -> 0/255"""
    return (np.asarray(bin01) > 0).astype(np.uint8) * 255


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
        else:
            mask_dir = input_path.parent / f"{input_path.name}_masks"
        rgb_map = dict(sorted(list_images(rgb_dir, RGB_EXTS).items()))
        if not rgb_map:
            raise RuntimeError(f"이미지 폴더에 읽을 수 있는 이미지가 없어요: {rgb_dir}")
    elif input_path.is_file() and input_path.suffix.lower() in RGB_EXTS:
        # A single image is handled as a one-frame labeling project.
        rgb_dir = input_path.parent
        if mask_dir_override is not None:
            mask_dir = Path(mask_dir_override)
        else:
            mask_dir = input_path.parent / f"{input_path.stem}_masks"
        rgb_map = {input_path.stem: input_path}
    else:
        raise FileNotFoundError(
            "INPUT_PATH가 유효한 이미지 폴더도 아니고 지원되는 비디오 파일도 아니에요: "
            f"{input_path}"
        )

    mask_map = dict(sorted(ensure_mask_files_for_rgb(rgb_dir, mask_dir, rgb_map).items()))
    keys = sorted(set(rgb_map.keys()) & set(mask_map.keys()))
    if not keys:
        raise RuntimeError(
            "매칭되는 파일이 없어요. RGB/MASK 파일명이 stem(확장자 제외) 기준으로 같은지 확인해줘.\n"
            f"RGB sample: {list(rgb_map.keys())[:10]}\n"
            f"MASK sample: {list(mask_map.keys())[:10]}"
        )

    return rgb_dir, mask_dir, rgb_map, mask_map, keys


def prepare_input_output_checked(
    input_path: Path,
    mask_dir_override: Path | None = None,
    output_root: Path | None = None,
    video_fps: float | None = None,
) -> tuple[Path, Path, dict[str, Path], dict[str, Path], list[str]]:
    input_path = Path(input_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(build_missing_input_message(input_path))
    return prepare_input_output(
        input_path,
        mask_dir_override=mask_dir_override,
        output_root=output_root,
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

        if event.type() == QtCore.QEvent.Type.KeyPress:
            return self.filterKey(event)
        if event.type() == QtCore.QEvent.Type.Wheel:
            return self.filterWheel(source, event)
        if event.type() in {
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseMove,
            QtCore.QEvent.Type.MouseButtonRelease,
        }:
            return self.filterMouse(source, event)
        return False

    def filterKey(self, event: QtCore.QEvent) -> bool:
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
        if focus_widget is None:
            return self.window.isActiveWindow()
        return focus_widget is self.window or self.window.isAncestorOf(focus_widget)

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
    input_path, mask_dir_override, output_root, video_fps = resolve_runtime_paths(argv)
    rgb_dir, mask_dir, rgb_map, mask_map, keys = prepare_input_output_checked(
        input_path,
        mask_dir_override=mask_dir_override,
        output_root=output_root,
        video_fps=video_fps,
    )

    state = {
        "idx": 0,
        "keys": keys,
        "rgb_map": rgb_map,
        "mask_map": mask_map,

        "dirty": False,
        "suspend": False,

        "active": "L1",
        "label": 1,

        "L1_b0": 100,
        "L1_b1": 60,
        "L2_b0": 100,
        "L2_b1": 60,

        "opacity_L1": 0.85,
        "opacity_L2": 0.65,
        "zoom": 1.0,
        "tool_mode": "paint",

        "vis_L1": True,
        "vis_L2": True,

        # ✅ 실행(세션) 전체에서 누적되는 저장 카운터
        "save_idx": 0,
    }

    frame_cache: FrameCache = {}
    frame_futures: FrameFuture = {}
    frame_lock = Lock()
    frame_executor = ThreadPoolExecutor(max_workers=2)

    viewer = napari.Viewer(title="Mask Editor (L1/L2 separate layers)")
    autosave_timer = QtCore.QTimer(viewer.window._qt_window)
    autosave_timer.setSingleShot(True)
    autosave_timer.setInterval(AUTOSAVE_DELAY_MS)
    change_timer = QtCore.QTimer(viewer.window._qt_window)
    change_timer.setInterval(CHANGE_SCAN_MS)
    mask_snapshot: np.ndarray | None = None

    img_layer = None
    l1_layer = None
    l2_layer = None

    hist_L1 = History(MAX_HISTORY)
    hist_L2 = History(MAX_HISTORY)

    # UI refs
    page_title = None
    page_slider = None
    page_spin = None
    btnL1 = btnL2 = None
    btnBG = btnFG = None
    btnPaint = btnFill = btnPan = None
    visL1_btn = visL2_btn = None

    s_L1_0 = sp_L1_0 = None
    s_L1_1 = sp_L1_1 = None
    s_L2_0 = sp_L2_0 = None
    s_L2_1 = sp_L2_1 = None

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
        return l1_layer if state["active"] == "L1" else l2_layer

    def set_window_title(extra: str = ""):
        try:
            k = keys[state["idx"]]
            star = " *" if state["dirty"] else ""
            base = (
                f"[ACTIVE={state['active']} label={state['label']} tool={state['tool_mode']}] "
                f"L1(b0={state['L1_b0']},b1={state['L1_b1']},vis={int(state['vis_L1'])}) "
                f"L2(b0={state['L2_b0']},b1={state['L2_b1']},vis={int(state['vis_L2'])}) "
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
        page_title.setText(f"Page  {v}/{len(keys)}" + ("  *unsaved(L1)" if state["dirty"] else ""))
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
        if l1_layer is not None:
            cmap1 = DirectLabelColormap(
                color_dict={None:(0,0,0,0), 0:(0,0,0,0), 1:(0.0,1.0,0.0,1.0)}
            )
            l1_layer.colormap = cmap1
            l1_layer.opacity = float(np.clip(state["opacity_L1"], 0.0, 1.0))
            try:
                l1_layer.blending = "translucent"
            except Exception:
                pass
            l1_layer.visible = bool(state["vis_L1"])

        if l2_layer is not None:
            cmap2 = DirectLabelColormap(
                color_dict={None:(0,0,0,0), 0:(0,0,0,0), 1:(1.0,0.0,1.0,1.0)}
            )
            l2_layer.colormap = cmap2
            l2_layer.opacity = float(np.clip(state["opacity_L2"], 0.0, 1.0))
            try:
                l2_layer.blending = "translucent"
            except Exception:
                pass
            l2_layer.visible = bool(state["vis_L2"])
            try:
                layers = viewer.layers
                i = layers.index(l2_layer)
                layers.move(i, len(layers) - 1)
            except Exception:
                pass

    def _l1_opacity_changed(_event=None):
        if state["suspend"] or l1_layer is None:
            return
        state["opacity_L1"] = float(np.clip(l1_layer.opacity, 0.0, 1.0))
        set_window_title()

    def _l2_opacity_changed(_event=None):
        if state["suspend"] or l2_layer is None:
            return
        state["opacity_L2"] = float(np.clip(l2_layer.opacity, 0.0, 1.0))
        set_window_title()

    def _l1_visible_changed(_event=None):
        if state["suspend"] or l1_layer is None:
            return
        state["vis_L1"] = bool(l1_layer.visible)
        sync_buttons()
        set_window_title()

    def _l2_visible_changed(_event=None):
        if state["suspend"] or l2_layer is None:
            return
        state["vis_L2"] = bool(l2_layer.visible)
        sync_buttons()
        set_window_title()

    def apply_tool_mode():
        layer = active_layer_obj()
        if state["tool_mode"] == "fill":
            layer = l1_layer
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

    def brush_value_for(active: str, lbl: int) -> int:
        return int(state[f"{active}_b{lbl}"])

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
        state[f"{state['active']}_b{state['label']}"] = v
        sync_brush_ui()

    def set_brush_value_for(active: str, lbl: int, v: int):
        v = int(np.clip(int(v), BRUSH_MIN, BRUSH_MAX))
        state[f"{active}_b{lbl}"] = v
        if state["active"] == active and state["label"] == lbl:
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

        layer_name = str(state["active"])
        label = int(state["label"])
        delta = BRUSH_WHEEL_STEP if direction > 0 else -BRUSH_WHEEL_STEP
        size = int(brush_value_for(layer_name, label)) + delta
        set_brush_value_for(layer_name, label, size)
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
        if btnL1 is not None:
            btnL1.setChecked(state["active"] == "L1")
            btnL2.setChecked(state["active"] == "L2")
        if btnBG is not None:
            btnBG.setChecked(state["label"] == 0)
            btnFG.setChecked(state["label"] == 1)
        if btnPaint is not None:
            btnPaint.setChecked(state["tool_mode"] == "paint")
        if btnFill is not None:
            btnFill.setChecked(state["tool_mode"] == "fill")
        if visL1_btn is not None:
            visL1_btn.setChecked(state["vis_L1"])
            visL2_btn.setChecked(state["vis_L2"])

    def sync_brush_ui():
        if s_L1_0 is None:
            return
        pairs = [
            (s_L1_0, sp_L1_0, "L1_b0"),
            (s_L1_1, sp_L1_1, "L1_b1"),
            (s_L2_0, sp_L2_0, "L2_b0"),
            (s_L2_1, sp_L2_1, "L2_b1"),
        ]
        for s, sp, key in pairs:
            v = int(state[key])
            s.blockSignals(True); sp.blockSignals(True)
            s.setValue(v); sp.setValue(v)
            s.blockSignals(False); sp.blockSignals(False)

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
            layer.brush_size = int(brush_value_for(state["active"], state["label"]))
        except Exception:
            pass
        sync_buttons()
        set_window_title()

    def restore_editor_state():
        sync_brush_ui()
        apply_active_state_to_layer()

    def _label_layer_brush_size_changed(layer_name: str):
        if state.get("suspend"):
            return
        layer = l1_layer if layer_name == "L1" else l2_layer
        if layer is None:
            return
        if state.get("active") != layer_name:
            return
        try:
            v = int(round(float(layer.brush_size)))
        except Exception:
            return
        v = int(np.clip(v, BRUSH_MIN, BRUSH_MAX))
        state[f"{layer_name}_b{state['label']}"] = v
        sync_brush_ui()
        set_window_title()

    def fill_l1_at_current_point(position) -> bool:
        if l1_layer is None:
            return False
        if position is None or len(position) < 2:
            return False

        data = np.asarray(l1_layer.data)
        y = int(round(float(position[0])))
        x = int(round(float(position[1])))
        if not (0 <= y < data.shape[0] and 0 <= x < data.shape[1]):
            return False

        filled = flood_fill_region(data, y, x, int(state["label"]))
        if np.array_equal(filled, data):
            return False

        hist_L1.push(data)
        hist_L1.suspend = True
        try:
            l1_layer.data = filled.astype(np.uint8)
        finally:
            hist_L1.suspend = False

        markChange("L1")
        apply_colormaps()
        refresh_page_ui()
        set_window_title(extra=f"(L1 fill @ x={x}, y={y}, label={state['label']})")
        return True

    def on_l1_mouse_drag(layer, event):
        if state.get("tool_mode") != "fill":
            return
        filled = fill_l1_at_current_point(getattr(event, "position", None))
        if filled:
            event.handled = True
        return

    def set_active_layer(name: str, *_args):
        remember_current_brush_size()
        state["active"] = "L1" if str(name).upper() == "L1" else "L2"
        apply_active_state_to_layer()

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
        if normalized == "fill":
            state["active"] = "L1"
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
        if l1_layer is None:
            mask_snapshot = None
            return
        mask_snapshot = np.asarray(l1_layer.data).copy()

    def checkMaskChange() -> bool:
        if l1_layer is None:
            return False
        if mask_snapshot is None:
            return False

        mask_data = np.asarray(l1_layer.data)
        shape_match = mask_data.shape == mask_snapshot.shape
        data_match = shape_match and np.array_equal(mask_data, mask_snapshot)
        return not data_match

    def scanMaskChange() -> None:
        if state["suspend"] or l1_layer is None:
            return
        if mask_snapshot is None:
            syncMaskSnapshot()
            return
        if not checkMaskChange():
            return
        markChange("L1")

    def scheduleAutosave() -> None:
        autosave_timer.start(AUTOSAVE_DELAY_MS)

    def markChange(layer: str) -> None:
        if state["suspend"]:
            return
        if layer != "L1":
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
        if l1_layer is None:
            return None
        frame_key = keys[state["idx"]]
        mask_path = mask_map[frame_key]
        mask_data = bin01_to_mask255(np.asarray(l1_layer.data))
        try:
            iio.imwrite(mask_path, mask_data)
        except Exception as exc:
            raise RuntimeError(f"Mask save failed: {mask_path}") from exc
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

        set_window_title(extra=f"(saved L1 only: {mask_path.name})")
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
        nonlocal img_layer, l1_layer, l2_layer

        if autosave_timer.isActive():
            autosave_timer.stop()
        k = keys[state["idx"]]
        rgb, l1, l2 = readFrame(k)

        hist_L1.clear(); hist_L2.clear()

        state["suspend"] = True
        try:
            if img_layer is None:
                img_layer = viewer.add_image(
                    rgb, name="RGB",
                    rgb=(rgb.ndim == 3 and rgb.shape[-1] in (3, 4)),
                )
            else:
                img_layer.data = rgb

            if l1_layer is None:
                l1_layer = viewer.add_labels(l1, name="L1 (main/save)")
                l1_layer.events.data.connect(lambda _event=None: markChange("L1"))
                l1_layer.events.paint.connect(lambda _event=None: markChange("L1"))
                l1_layer.events.opacity.connect(_l1_opacity_changed)
                l1_layer.events.visible.connect(_l1_visible_changed)
                l1_layer.events.brush_size.connect(lambda _event=None: _label_layer_brush_size_changed("L1"))
                l1_layer.mouse_drag_callbacks.append(on_l1_mouse_drag)
            else:
                l1_layer.data = l1

            if l2_layer is None:
                l2_layer = viewer.add_labels(l2, name="L2 (temp)")
                try:
                    l2_layer.editable = True
                except Exception:
                    pass
                l2_layer.events.data.connect(lambda _event=None: markChange("L2"))
                l2_layer.events.opacity.connect(_l2_opacity_changed)
                l2_layer.events.visible.connect(_l2_visible_changed)
                l2_layer.events.brush_size.connect(lambda _event=None: _label_layer_brush_size_changed("L2"))
            else:
                l2_layer.data = l2
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
        if l1_layer is None:
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
    def copy_prev_to_l2(*_args):
        if img_layer is None or l2_layer is None:
            return
        prev_idx = (state["idx"] - 1) % len(keys)
        prev_key = keys[prev_idx]
        prev_bin = load_mask255_as_bin01(mask_map[prev_key], img_layer.data.shape[:2])

        hist_L2.push(np.asarray(l2_layer.data))
        hist_L2.suspend = True
        try:
            l2_layer.data = prev_bin.astype(np.uint8)
        finally:
            hist_L2.suspend = False

        apply_colormaps()
        set_window_title(extra=f"(copied prev {prev_key} -> L2)")

    def merge_l1_l2_to_l1(*_args):
        if l1_layer is None or l2_layer is None:
            return
        hist_L1.push(np.asarray(l1_layer.data))
        hist_L2.push(np.asarray(l2_layer.data))

        l1 = (np.asarray(l1_layer.data) > 0)
        l2 = (np.asarray(l2_layer.data) > 0)
        merged = (l1 | l2).astype(np.uint8)

        hist_L1.suspend = True
        hist_L2.suspend = True
        try:
            l1_layer.data = merged
            l2_layer.data = np.zeros_like(merged, dtype=np.uint8)
        finally:
            hist_L1.suspend = False
            hist_L2.suspend = False

        markChange("L1")
        apply_colormaps()
        set_window_title(extra="(merged L2 into L1, cleared L2)")
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

        hist_L1.clear(); hist_L2.clear()
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
    def toggle_vis_L1(checked: bool, *_args):
        state["vis_L1"] = bool(checked)
        if l1_layer is not None:
            l1_layer.visible = bool(checked)
        sync_buttons()
        set_window_title()

    def toggle_vis_L2(checked: bool, *_args):
        state["vis_L2"] = bool(checked)
        if l2_layer is not None:
            l2_layer.visible = bool(checked)
        sync_buttons()
        set_window_title()

    # ======================
    # UI
    # ======================
    dock = QtWidgets.QWidget()
    L = QtWidgets.QVBoxLayout(dock)
    L.setContentsMargins(10, 10, 10, 10)
    L.setSpacing(10)

    l2_dock = QtWidgets.QWidget()
    L2P = QtWidgets.QVBoxLayout(l2_dock)
    L2P.setContentsMargins(10, 10, 10, 10)
    L2P.setSpacing(10)

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
    save_btn = QtWidgets.QPushButton("SAVE (L1 only)")
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

    # Active layer buttons
    layer_row = QtWidgets.QHBoxLayout()
    btnL1 = QtWidgets.QPushButton("Active: L1")
    btnL2 = QtWidgets.QPushButton("Active: L2")
    btnL1.setCheckable(True); btnL2.setCheckable(True)
    grp_layer = QtWidgets.QButtonGroup()
    grp_layer.setExclusive(True)
    grp_layer.addButton(btnL1); grp_layer.addButton(btnL2)
    btnL1.clicked.connect(lambda *_: set_active_layer("L1"))
    btnL2.clicked.connect(lambda *_: set_active_layer("L2"))
    layer_row.addWidget(btnL1); layer_row.addWidget(btnL2)
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
    L.addWidget(QtWidgets.QLabel("Paint uses only L1 and treats label 1 as boundary."))

    # Visibility toggles
    vis_row = QtWidgets.QHBoxLayout()
    visL1_btn = QtWidgets.QPushButton("Show L1")
    visL2_btn = QtWidgets.QPushButton("Show L2")
    visL1_btn.setCheckable(True); visL2_btn.setCheckable(True)
    visL1_btn.setChecked(True); visL2_btn.setChecked(True)
    visL1_btn.toggled.connect(toggle_vis_L1)
    visL2_btn.toggled.connect(toggle_vis_L2)
    vis_row.addWidget(visL1_btn); vis_row.addWidget(visL2_btn)
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
    copy_btn = QtWidgets.QPushButton("Copy Prev → L2")
    copy_btn.clicked.connect(copy_prev_to_l2)
    L.addWidget(copy_btn)

    merge_btn = QtWidgets.QPushButton("Merge L1+L2 → L1 (clear L2)")
    merge_btn.clicked.connect(merge_l1_l2_to_l1)
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
        return s, sp

    s_L1_0, sp_L1_0 = add_brush_block(L, "L1 brush for label 0", state["L1_b0"])
    s_L1_1, sp_L1_1 = add_brush_block(L, "L1 brush for label 1", state["L1_b1"])
    L2P.addWidget(QtWidgets.QLabel("L2 Brush"))
    s_L2_0, sp_L2_0 = add_brush_block(L2P, "L2 brush for label 0", state["L2_b0"])
    s_L2_1, sp_L2_1 = add_brush_block(L2P, "L2 brush for label 1", state["L2_b1"])

    L2P.addStretch(1)

    def link_brush(s, sp, layer_name: str, lbl: int):
        def _from_slider(v):
            sp.setValue(v)
            set_brush_value_for(layer_name, lbl, v)
        def _from_spin(v):
            s.setValue(v)
            set_brush_value_for(layer_name, lbl, v)
        s.valueChanged.connect(_from_slider)
        sp.valueChanged.connect(_from_spin)

    link_brush(s_L1_0, sp_L1_0, "L1", 0)
    link_brush(s_L1_1, sp_L1_1, "L1", 1)
    link_brush(s_L2_0, sp_L2_0, "L2", 0)
    link_brush(s_L2_1, sp_L2_1, "L2", 1)

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

    l2_dock_scroll = QtWidgets.QScrollArea()
    l2_dock_scroll.setWidgetResizable(True)
    l2_dock_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    l2_dock_scroll.setWidget(l2_dock)

    viewer.window.add_dock_widget(l2_dock_scroll, name="L2 BRUSH", area="left")
    viewer.window.add_dock_widget(log_dock, name="LOG", area="left")
    viewer.window.add_dock_widget(dock_scroll, name="TOOLS", area="right")

    # ======================
    # Key bindings
    # ======================
    def bindMany(key_names: list[str], action: NavAction) -> None:
        for key_name in key_names:
            viewer.bind_key(key_name, overwrite=True)(action)

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

    bindMany(["F1"], lambda *_: set_active_layer("L1"))
    bindMany(["F2"], lambda *_: set_active_layer("L2"))

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
