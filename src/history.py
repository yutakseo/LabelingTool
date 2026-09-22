"""Per-layer undo and redo snapshots."""

from __future__ import annotations

import numpy as np


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
