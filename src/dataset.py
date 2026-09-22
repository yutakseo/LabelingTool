"""Dataset preparation, image/mask pairing, and metadata export."""

from __future__ import annotations

from .classes import load_dataset_class_definitions
from .classes import register_mask_class_ids
from .config import CLASS_DEFINITIONS
from .config import MASK_EXTS
from .config import RGB_EXTS
from .config import VIDEO_EXTS
from .images import list_images
from .images import load_rgb
from .masks import load_multiclass_mask
from .masks import write_mask_preview
from .video import choose_video_target_fps
from .video import default_video_project_root
from .video import extract_video_frames
from .video import find_existing_video_project
from .video import format_fps_dirname
from .video import is_video_path
from collections.abc import Mapping
from pathlib import Path
import imageio.v3 as iio
import json
import numpy as np
import shutil


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
