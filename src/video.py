"""Video metadata, FPS selection, and frame extraction."""

from __future__ import annotations

from .config import FPS_DIR_PREFIX
from .config import FRAME_EXT
from .config import FRAME_NAME_FMT
from .config import RGB_EXTS
from .config import VIDEO_EXTS
from .images import list_images
from pathlib import Path
from tqdm import tqdm
import imageio.v3 as iio
import imageio_ffmpeg
import math
import numpy as np


def is_video_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def default_video_project_root(video_path: Path, output_root: Path | None) -> Path:
    if output_root is None:
        return video_path.parent / video_path.stem
    return Path(output_root) / video_path.stem


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
