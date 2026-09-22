"""Shared callback and frame data type aliases."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from typing import TypeAlias
import numpy as np


NavAction: TypeAlias = Callable[[], None]
RangeAction: TypeAlias = Callable[[int], bool]
PanAction: TypeAlias = Callable[[int, int], bool]
ClassMasks: TypeAlias = dict[int, np.ndarray]
FrameData: TypeAlias = tuple[np.ndarray, ClassMasks]
FrameCache: TypeAlias = dict[str, FrameData]
FrameFuture: TypeAlias = dict[str, Future[FrameData]]

