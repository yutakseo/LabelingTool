"""Qt keyboard, mouse, and focus handling."""

from __future__ import annotations

from .types import NavAction
from .types import PanAction
from .types import RangeAction
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from typing import cast


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
