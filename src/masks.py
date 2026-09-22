"""Indexed masks, previews, frame loading, and region filling."""

from __future__ import annotations

from .config import CLASS_DEFINITIONS
from .images import load_rgb
from .types import FrameData
from collections import deque
from collections.abc import Mapping
from pathlib import Path
import imageio.v3 as iio
import numpy as np


FILL_STRUCTURE = np.array(
    [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ],
    dtype=bool,
)


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
