"""Class validation, colours, and metadata restoration."""

from __future__ import annotations

from .config import CLASS_DEFINITIONS
from pathlib import Path
import json
import numpy as np


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
