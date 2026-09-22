"""Napari viewer, editing session, controls, and application lifecycle."""

from __future__ import annotations

from .classes import default_class_color
from .classes import validate_class_definitions
from .cli import resolve_runtime_paths
from .config import AUTOSAVE_DELAY_MS
from .config import BRUSH_MAX
from .config import BRUSH_MIN
from .config import BRUSH_STEP
from .config import BRUSH_WHEEL_STEP
from .config import CHANGE_SCAN_MS
from .config import CLASS_DEFINITIONS
from .config import FRAME_CACHE_LIMIT
from .config import MAX_HISTORY
from .config import ZOOM_MAX
from .config import ZOOM_MIN
from .config import ZOOM_STEP
from .config import ZOOM_WHEEL_RATIO
from .dataset import prepare_input_output_checked
from .dataset import write_dataset_metadata
from .history import History
from .input_events import InputFilter
from .input_events import text_input_has_focus
from .masks import compose_multiclass_mask
from .masks import flood_fill_region
from .masks import loadFrame
from .masks import load_multiclass_mask
from .masks import write_mask_preview
from .types import FrameCache
from .types import FrameData
from .types import FrameFuture
from .types import NavAction
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from napari.utils.colormaps import DirectLabelColormap
from pathlib import Path
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from threading import Lock
import imageio.v3 as iio
import logging
import napari
import numpy as np


LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    validate_class_definitions()
    input_path, pseudo_mask_input, dataset_output_root, video_fps = resolve_runtime_paths(argv)
    rgb_dir, mask_dir, rgb_map, mask_map, keys = prepare_input_output_checked(
        input_path,
        pseudo_mask_input=pseudo_mask_input,
        dataset_output_root=dataset_output_root,
        video_fps=video_fps,
    )

    state = {
        "idx": 0,
        "keys": keys,
        "rgb_map": rgb_map,
        "mask_map": mask_map,

        "dirty": False,
        "suspend": False,

        "active": next(iter(CLASS_DEFINITIONS)),
        "label": 1,
        "brush_b0": 100,
        "brush_b1": 60,
        "zoom": 1.0,
        "tool_mode": "paint",

        # ✅ 실행(세션) 전체에서 누적되는 저장 카운터
        "save_idx": 0,
    }
    for class_id in CLASS_DEFINITIONS:
        state[f"class_{class_id}_opacity"] = 0.85
        state[f"class_{class_id}_visible"] = True

    frame_cache: FrameCache = {}
    frame_futures: FrameFuture = {}
    frame_lock = Lock()
    frame_executor = ThreadPoolExecutor(max_workers=2)

    viewer = napari.Viewer(title="Multi-class Mask Editor")
    autosave_timer = QtCore.QTimer(viewer.window._qt_window)
    autosave_timer.setSingleShot(True)
    autosave_timer.setInterval(AUTOSAVE_DELAY_MS)
    change_timer = QtCore.QTimer(viewer.window._qt_window)
    change_timer.setInterval(CHANGE_SCAN_MS)
    mask_snapshot: np.ndarray | None = None

    img_layer = None
    class_layers: dict[int, object] = {}
    histories = {class_id: History(MAX_HISTORY) for class_id in CLASS_DEFINITIONS}

    # UI refs
    page_title = None
    page_slider = None
    page_spin = None
    class_buttons: dict[int, QtWidgets.QPushButton] = {}
    btnBG = btnFG = None
    btnPaint = btnFill = btnPan = None
    visibility_buttons: dict[int, QtWidgets.QPushButton] = {}
    brush_controls: dict[int, tuple[QtWidgets.QSlider, QtWidgets.QSpinBox]] = {}
    class_summary_label: QtWidgets.QLabel | None = None

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
        return class_layers.get(int(state["active"]))

    def set_window_title(extra: str = ""):
        try:
            k = keys[state["idx"]]
            star = " *" if state["dirty"] else ""
            class_state = " ".join(
                f"{CLASS_DEFINITIONS[class_id]['name']}"
                f"(vis={int(state[f'class_{class_id}_visible'])})"
                for class_id in CLASS_DEFINITIONS
            )
            base = (
                f"[ACTIVE={state['active']} label={state['label']} tool={state['tool_mode']}] "
                f"{class_state} "
                f"brush(erase={state['brush_b0']},paint={state['brush_b1']}) "
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
        page_title.setText(f"Page  {v}/{len(keys)}" + ("  *unsaved" if state["dirty"] else ""))
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
        for class_id, layer in class_layers.items():
            color = CLASS_DEFINITIONS[class_id]["color"]
            cmap = DirectLabelColormap(
                color_dict={None: (0, 0, 0, 0), 0: (0, 0, 0, 0), 1: color}
            )
            layer.colormap = cmap
            layer.opacity = float(np.clip(state[f"class_{class_id}_opacity"], 0.0, 1.0))
            try:
                layer.blending = "translucent"
            except Exception:
                pass
            layer.visible = bool(state[f"class_{class_id}_visible"])
            try:
                layers = viewer.layers
                i = layers.index(layer)
                layers.move(i, len(layers) - 1)
            except Exception:
                pass

    def _class_opacity_changed(class_id: int, _event=None):
        layer = class_layers.get(class_id)
        if state["suspend"] or layer is None:
            return
        state[f"class_{class_id}_opacity"] = float(np.clip(layer.opacity, 0.0, 1.0))
        set_window_title()

    def _class_visible_changed(class_id: int, _event=None):
        layer = class_layers.get(class_id)
        if state["suspend"] or layer is None:
            return
        state[f"class_{class_id}_visible"] = bool(layer.visible)
        sync_buttons()
        set_window_title()

    def refresh_class_name_ui(class_id: int) -> None:
        name = str(CLASS_DEFINITIONS[class_id]["name"])
        button = class_buttons.get(class_id)
        if button is not None:
            button.setText(f"Active: {name} ({class_id})")
        visibility_button = visibility_buttons.get(class_id)
        if visibility_button is not None:
            visibility_button.setText(f"Show {name}")
        if class_summary_label is not None:
            summary = ", ".join(
                f"{cid}={definition['name']}"
                for cid, definition in CLASS_DEFINITIONS.items()
            )
            class_summary_label.setText(f"Saved mask values: background=0, {summary}.")

    def _class_name_changed(class_id: int, _event=None) -> None:
        """Keep user-entered Napari layer names in the class UI for this session."""
        layer = class_layers.get(class_id)
        if layer is None:
            return
        name = str(layer.name).strip()
        default_suffix = f" (class {class_id})"
        if name.endswith(default_suffix):
            name = name[: -len(default_suffix)].strip()
        if not name:
            return
        CLASS_DEFINITIONS[class_id]["name"] = name
        refresh_class_name_ui(class_id)
        write_dataset_metadata(mask_dir.parent)
        set_window_title()

    def apply_tool_mode():
        layer = active_layer_obj()
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

    def brush_value_for(lbl: int) -> int:
        return int(state[f"brush_b{lbl}"])

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
        state[f"brush_b{state['label']}"] = v
        sync_brush_ui()

    def set_brush_value_for(lbl: int, v: int):
        v = int(np.clip(int(v), BRUSH_MIN, BRUSH_MAX))
        state[f"brush_b{lbl}"] = v
        if state["label"] == lbl:
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

        label = int(state["label"])
        delta = BRUSH_WHEEL_STEP if direction > 0 else -BRUSH_WHEEL_STEP
        size = brush_value_for(label) + delta
        set_brush_value_for(label, size)
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
        for class_id, button in class_buttons.items():
            button.setChecked(state["active"] == class_id)
        if btnBG is not None:
            btnBG.setChecked(state["label"] == 0)
            btnFG.setChecked(state["label"] == 1)
        if btnPaint is not None:
            btnPaint.setChecked(state["tool_mode"] == "paint")
        if btnFill is not None:
            btnFill.setChecked(state["tool_mode"] == "fill")
        for class_id, button in visibility_buttons.items():
            button.setChecked(state[f"class_{class_id}_visible"])

    def sync_brush_ui():
        for label, (slider, spinbox) in brush_controls.items():
            v = brush_value_for(label)
            slider.blockSignals(True); spinbox.blockSignals(True)
            slider.setValue(v); spinbox.setValue(v)
            slider.blockSignals(False); spinbox.blockSignals(False)

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
            layer.brush_size = brush_value_for(int(state["label"]))
        except Exception:
            pass
        sync_buttons()
        set_window_title()

    def restore_editor_state():
        sync_brush_ui()
        apply_active_state_to_layer()

    def _label_layer_brush_size_changed(class_id: int):
        if state.get("suspend"):
            return
        layer = class_layers.get(class_id)
        if layer is None:
            return
        if state.get("active") != class_id:
            return
        try:
            v = int(round(float(layer.brush_size)))
        except Exception:
            return
        v = int(np.clip(v, BRUSH_MIN, BRUSH_MAX))
        state[f"brush_b{state['label']}"] = v
        sync_brush_ui()
        set_window_title()

    def fill_active_class_at_current_point(position) -> bool:
        class_id = int(state["active"])
        layer = class_layers.get(class_id)
        if layer is None:
            return False
        if position is None or len(position) < 2:
            return False

        data = np.asarray(layer.data)
        y = int(round(float(position[0])))
        x = int(round(float(position[1])))
        if not (0 <= y < data.shape[0] and 0 <= x < data.shape[1]):
            return False

        filled = flood_fill_region(data, y, x, int(state["label"]))
        if np.array_equal(filled, data):
            return False

        history = histories[class_id]
        history.push(data)
        history.suspend = True
        try:
            layer.data = filled.astype(np.uint8)
        finally:
            history.suspend = False

        markChange(f"class {class_id}")
        apply_colormaps()
        refresh_page_ui()
        set_window_title(extra=f"(class {class_id} fill @ x={x}, y={y}, label={state['label']})")
        return True

    def on_class_mouse_drag(layer, event):
        if state.get("tool_mode") != "fill":
            return
        filled = fill_active_class_at_current_point(getattr(event, "position", None))
        if filled:
            event.handled = True
        return

    def set_active_layer(class_id: int, *_args):
        class_id = int(class_id)
        if class_id not in CLASS_DEFINITIONS:
            return
        remember_current_brush_size()
        state["active"] = class_id
        apply_active_state_to_layer()
        sync_brush_ui()

    def _viewer_active_layer_changed(event=None) -> None:
        """Synchronize state when a class is selected in Napari's layer list."""
        selected_layer = getattr(event, "value", None)
        if selected_layer is None:
            selected_layer = viewer.layers.selection.active
        selected_class_id = next(
            (
                class_id
                for class_id, layer in class_layers.items()
                if layer is selected_layer
            ),
            None,
        )
        if selected_class_id is None or selected_class_id == state["active"]:
            return
        remember_current_brush_size()
        state["active"] = selected_class_id
        apply_active_state_to_layer()
        sync_brush_ui()

    viewer.layers.selection.events.active.connect(_viewer_active_layer_changed)

    def connect_class_layer_events(class_id: int, layer) -> None:
        layer.events.data.connect(
            lambda _event=None, cid=class_id: markChange(f"class {cid}")
        )
        layer.events.paint.connect(
            lambda _event=None, cid=class_id: markChange(f"class {cid}")
        )
        layer.events.opacity.connect(
            lambda _event=None, cid=class_id: _class_opacity_changed(cid)
        )
        layer.events.visible.connect(
            lambda _event=None, cid=class_id: _class_visible_changed(cid)
        )
        layer.events.name.connect(
            lambda _event=None, cid=class_id: _class_name_changed(cid)
        )
        layer.events.brush_size.connect(
            lambda _event=None, cid=class_id: _label_layer_brush_size_changed(cid)
        )
        layer.mouse_drag_callbacks.append(on_class_mouse_drag)

    def add_new_class(*_args) -> None:
        if img_layer is None:
            return
        class_id = max(CLASS_DEFINITIONS, default=0) + 1
        if class_id > 255:
            QtWidgets.QMessageBox.warning(
                viewer.window._qt_window,
                "Class limit",
                "A uint8 segmentation mask supports class IDs only up to 255.",
            )
            return
        name, accepted = QtWidgets.QInputDialog.getText(
            viewer.window._qt_window,
            "Add class",
            f"Name for class {class_id}:",
            text=f"class_{class_id}",
        )
        name = name.strip()
        if not accepted or not name:
            return

        CLASS_DEFINITIONS[class_id] = {
            "name": name,
            "color": default_class_color(class_id),
        }
        state[f"class_{class_id}_opacity"] = 0.85
        state[f"class_{class_id}_visible"] = True
        histories[class_id] = History(MAX_HISTORY)

        state["suspend"] = True
        try:
            layer = viewer.add_labels(
                np.zeros(img_layer.data.shape[:2], dtype=np.uint8),
                name=f"{name} (class {class_id})",
            )
            class_layers[class_id] = layer
            connect_class_layer_events(class_id, layer)
        finally:
            state["suspend"] = False

        class_button = QtWidgets.QPushButton(f"Active: {name} ({class_id})")
        class_button.setCheckable(True)
        grp_layer.addButton(class_button)
        class_button.clicked.connect(
            lambda _checked=False, cid=class_id: set_active_layer(cid)
        )
        class_buttons[class_id] = class_button
        layer_row.addWidget(class_button)

        visibility_button = QtWidgets.QPushButton(f"Show {name}")
        visibility_button.setCheckable(True)
        visibility_button.setChecked(True)
        visibility_button.toggled.connect(
            lambda checked, cid=class_id: toggle_class_visibility(cid, checked)
        )
        visibility_buttons[class_id] = visibility_button
        vis_row.addWidget(visibility_button)

        if class_id <= 12:
            bindMany([f"F{class_id}"], lambda cid=class_id: set_active_layer(cid))
        refresh_class_name_ui(class_id)
        write_dataset_metadata(mask_dir.parent)
        apply_colormaps()
        set_active_layer(class_id)

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
        if set(class_layers) != set(CLASS_DEFINITIONS):
            mask_snapshot = None
            return
        mask_snapshot = compose_multiclass_mask(
            {class_id: layer.data for class_id, layer in class_layers.items()}
        )

    def checkMaskChange() -> bool:
        if set(class_layers) != set(CLASS_DEFINITIONS):
            return False
        if mask_snapshot is None:
            return False

        mask_data = compose_multiclass_mask(
            {class_id: layer.data for class_id, layer in class_layers.items()}
        )
        shape_match = mask_data.shape == mask_snapshot.shape
        data_match = shape_match and np.array_equal(mask_data, mask_snapshot)
        return not data_match

    def scanMaskChange() -> None:
        if state["suspend"] or set(class_layers) != set(CLASS_DEFINITIONS):
            return
        if mask_snapshot is None:
            syncMaskSnapshot()
            return
        if not checkMaskChange():
            return
        markChange("multi-class mask")

    def scheduleAutosave() -> None:
        autosave_timer.start(AUTOSAVE_DELAY_MS)

    def markChange(layer: str) -> None:
        if state["suspend"]:
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
        if set(class_layers) != set(CLASS_DEFINITIONS):
            return None
        frame_key = keys[state["idx"]]
        mask_path = mask_map[frame_key]
        mask_data = compose_multiclass_mask(
            {class_id: layer.data for class_id, layer in class_layers.items()}
        )
        try:
            iio.imwrite(mask_path, mask_data)
            write_mask_preview(mask_path, mask_data)
        except Exception as exc:
            raise RuntimeError(f"Mask save failed: {mask_path}") from exc
        write_dataset_metadata(mask_dir.parent)
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

        set_window_title(extra=f"(saved classes 1/2: {mask_path.name})")
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
        nonlocal img_layer

        if autosave_timer.isActive():
            autosave_timer.stop()
        k = keys[state["idx"]]
        rgb, class_masks = readFrame(k)

        for history in histories.values():
            history.clear()

        state["suspend"] = True
        try:
            if img_layer is None:
                img_layer = viewer.add_image(
                    rgb, name="RGB",
                    rgb=(rgb.ndim == 3 and rgb.shape[-1] in (3, 4)),
                )
            else:
                img_layer.data = rgb

            for class_id, definition in CLASS_DEFINITIONS.items():
                layer = class_layers.get(class_id)
                if layer is None:
                    layer = viewer.add_labels(
                        class_masks[class_id],
                        name=f"{definition['name']} (class {class_id})",
                    )
                    class_layers[class_id] = layer
                    connect_class_layer_events(class_id, layer)
                else:
                    layer.data = class_masks[class_id]
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
        if not class_layers:
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
    def copy_previous_to_active_class(*_args):
        class_id = int(state["active"])
        layer = class_layers.get(class_id)
        if img_layer is None or layer is None:
            return
        prev_idx = (state["idx"] - 1) % len(keys)
        prev_key = keys[prev_idx]
        prev_mask = load_multiclass_mask(mask_map[prev_key], img_layer.data.shape[:2])

        history = histories[class_id]
        history.push(np.asarray(layer.data))
        history.suspend = True
        try:
            layer.data = (prev_mask == class_id).astype(np.uint8)
        finally:
            history.suspend = False

        markChange(f"class {class_id}")
        apply_colormaps()
        set_window_title(extra=f"(copied prev {prev_key} -> class {class_id})")

    def convert_active_class_to_first(*_args):
        """Move the active class into the first configured class."""
        source_id = int(state["active"])
        target_id = next(iter(CLASS_DEFINITIONS))
        if source_id == target_id:
            return
        source = class_layers.get(source_id)
        target = class_layers.get(target_id)
        if source is None or target is None:
            return
        histories[source_id].push(np.asarray(source.data))
        histories[target_id].push(np.asarray(target.data))
        for history in (histories[source_id], histories[target_id]):
            history.suspend = True
        try:
            target.data = ((np.asarray(target.data) > 0) | (np.asarray(source.data) > 0)).astype(np.uint8)
            source.data = np.zeros_like(source.data, dtype=np.uint8)
        finally:
            for history in (histories[source_id], histories[target_id]):
                history.suspend = False
        markChange(f"class {source_id}")
        apply_colormaps()
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

        for history in histories.values():
            history.clear()
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
    def toggle_class_visibility(class_id: int, checked: bool, *_args):
        state[f"class_{class_id}_visible"] = bool(checked)
        layer = class_layers.get(class_id)
        if layer is not None:
            layer.visible = bool(checked)
        sync_buttons()
        set_window_title()

    # ======================
    # UI
    # ======================
    dock = QtWidgets.QWidget()
    L = QtWidgets.QVBoxLayout(dock)
    L.setContentsMargins(10, 10, 10, 10)
    L.setSpacing(10)

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
    save_btn = QtWidgets.QPushButton("SAVE multi-class mask")
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

    # Class buttons are created from CLASS_DEFINITIONS.
    layer_row = QtWidgets.QHBoxLayout()
    grp_layer = QtWidgets.QButtonGroup()
    grp_layer.setExclusive(True)
    for class_id, definition in CLASS_DEFINITIONS.items():
        button = QtWidgets.QPushButton(f"Active: {definition['name']} ({class_id})")
        button.setCheckable(True)
        grp_layer.addButton(button)
        button.clicked.connect(lambda _checked=False, cid=class_id: set_active_layer(cid))
        class_buttons[class_id] = button
        layer_row.addWidget(button)
    add_class_button = QtWidgets.QPushButton("+ Add class")
    add_class_button.clicked.connect(add_new_class)
    layer_row.addWidget(add_class_button)
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
    class_summary = ", ".join(
        f"{class_id}={definition['name']}" for class_id, definition in CLASS_DEFINITIONS.items()
    )
    class_summary_label = QtWidgets.QLabel(
        f"Saved mask values: background=0, {class_summary}."
    )
    L.addWidget(class_summary_label)

    # Visibility toggles
    vis_row = QtWidgets.QHBoxLayout()
    for class_id, definition in CLASS_DEFINITIONS.items():
        button = QtWidgets.QPushButton(f"Show {definition['name']}")
        button.setCheckable(True)
        button.setChecked(True)
        button.toggled.connect(
            lambda checked, cid=class_id: toggle_class_visibility(cid, checked)
        )
        visibility_buttons[class_id] = button
        vis_row.addWidget(button)
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
    copy_btn = QtWidgets.QPushButton("Copy previous → active class")
    copy_btn.clicked.connect(copy_previous_to_active_class)
    L.addWidget(copy_btn)

    merge_btn = QtWidgets.QPushButton("Convert active class → first class")
    merge_btn.clicked.connect(convert_active_class_to_first)
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
        return lbl, s, sp

    def link_brush(s, sp, lbl: int):
        def _from_slider(v):
            sp.setValue(v)
            set_brush_value_for(lbl, v)
        def _from_spin(v):
            s.setValue(v)
            set_brush_value_for(lbl, v)
        s.valueChanged.connect(_from_slider)
        sp.valueChanged.connect(_from_spin)

    L.addWidget(QtWidgets.QLabel("Common brush sizes"))
    for label, text in ((0, "Eraser"), (1, "Brush")):
        _control_label, slider, spinbox = add_brush_block(
            L,
            text,
            state[f"brush_b{label}"],
        )
        link_brush(slider, spinbox, label)
        brush_controls[label] = (slider, spinbox)

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

    viewer.window.add_dock_widget(log_dock, name="LOG", area="left")
    viewer.window.add_dock_widget(dock_scroll, name="TOOLS", area="right")

    # ======================
    # Key bindings
    # ======================
    def bindMany(key_names: list[str], action: NavAction) -> None:
        def run_unless_typing(*_args: object) -> None:
            if text_input_has_focus():
                return
            action()

        for key_name in key_names:
            viewer.bind_key(key_name, overwrite=True)(run_unless_typing)

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

    for class_id in sorted(CLASS_DEFINITIONS):
        if 1 <= class_id <= 12:
            bindMany([f"F{class_id}"], lambda cid=class_id: set_active_layer(cid))

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
