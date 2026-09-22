"""Command-line arguments and runtime path resolution."""

from __future__ import annotations

from .config import DATASET_OUTPUT_PARENT_PATH
from .config import DATASET_SESSION_NAME
from .config import ORIGINAL_IMAGE_INPUT_PATH
from .config import PSEUDO_MASK_INPUT_PATH
from pathlib import Path
import argparse


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
