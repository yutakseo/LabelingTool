"""Image discovery and RGB loading."""

from __future__ import annotations

from pathlib import Path
import imageio.v3 as iio
import numpy as np


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
