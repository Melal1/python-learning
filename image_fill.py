from enum import Enum
from typing import List, Optional, Tuple
import os

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtGui import QAction, QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QRubberBand,
    QScrollArea,
    QSpinBox,
    QToolBar,
)


class ClickableLabel(QLabel):
    def __init__(self, parent: Optional[QMainWindow] = None):
        super().__init__(parent)
        self.app: Optional["Editor"] = None

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:
        if self.app and ev is not None:
            self.app.on_click(ev)

    def mouseMoveEvent(self, ev: QMouseEvent | None) -> None:
        if self.app and ev is not None:
            self.app.on_move(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent | None) -> None:
        if self.app and ev is not None:
            self.app.on_release(ev)
        super().mouseReleaseEvent(ev)


class Modes(Enum):
    fill = 0
    draw = 1
    crop = 2


class Editor(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # State Variables
        self.setWindowTitle("Image flood fill")
        self._mode: Modes = Modes.fill
        self._cv_img: Optional[np.ndarray] = None
        self._cv_img_cropped_dim: Optional[np.ndarray] = None
        self._crop_coords: Optional[Tuple[int, int, int, int]] = None

        self._history: List[
            List[Tuple[int, int, Tuple[int, int, int] | np.ndarray]]
        ] = []
        self._his_indecies: List[np.ndarray] = []

       
        self._color = (255, 0, 0)
        self._reduce_cpu = False
        self._visualize = False
        self._preserve_gray_scale = False
        self._load_gray = False
        self._ui_locked = False
        self._normal_tolerance_val = 10
        self._gray_tolerance_val = 0

        self._init_ui()
        self._init_toolbars()

        self.rub_band = QRubberBand(QRubberBand.Shape.Rectangle, self._image_area)
        self.anchor = QPoint()

    def _init_ui(self):
        self._scroll_area = QScrollArea()
        self._image_area = ClickableLabel()

        self._image_area.app = self
        self._image_area.setText("Load an image to start")
        self._image_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._scroll_area.setWidget(self._image_area)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self._scroll_area)

    def _init_toolbars(self):
        self.main_toolbar = QToolBar("Main ToolBar")
        self.crop_toolbar = QToolBar("Crop ToolBar")

        self._setup_main_toolbar()
        self.setup_crop_toolbar()

        self.addToolBar(self.main_toolbar)
        self.addToolBar(self.crop_toolbar)
        self.crop_toolbar.hide()

    def _create_action(
        self, text: str, slot, checkable=False, checked=False, shortcut: Optional[str] = None
    ) -> QAction:
        action = QAction(text, self)
        action.setCheckable(checkable)
        if checkable:
            action.setChecked(checked)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        return action

    def _setup_main_toolbar(self):
        self._load_btn = self._create_action("Load", self.load_image, shortcut="Ctrl+O")
        self._load_gray_btn = self._create_action(
            "Load as Grayscale", self.toggle_load_gray, True, self._load_gray
        )
        self._save_btn = self._create_action("Save", self.save_image, shortcut="Ctrl+S")

        self._color_btn = self._create_action("Color", self.select_color)
        self._undo_btn = self._create_action("Undo", self.undo, shortcut="Ctrl+Z")
        self._fill_btn = self._create_action(
            "Fill", lambda: self.set_mode(Modes.fill), True, True, shortcut="F"
        )
        self._draw_btn = self._create_action(
            "Draw", lambda: self.set_mode(Modes.draw), True, False, shortcut="D"
        )
        self._crop_btn = self._create_action("Crop", self.convert_crop_bar, shortcut="C")

        self._cpu_btn = self._create_action(
            " Reduce Cpu Usage", self.toggle_cpu, True, self._reduce_cpu
        )
        self._vis_btn = self._create_action(
            " Visualize", self.toggle_vis, True, self._visualize
        )
        self._preserve_gray_scale_btn = self._create_action(
            "Preserve graysclae", self.toggle_pres_grayscale, True, False
        )

        # Build Main Toolbar
        self.main_toolbar.addActions(
            [
                self._load_btn,
                self._load_gray_btn,
                self._save_btn,
                self._color_btn,
                self._fill_btn,
                self._draw_btn,
                self._undo_btn,
            ]
        )

        self.main_toolbar.addSeparator()
        self._setup_params()

        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self._cpu_btn)
        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self._vis_btn)
        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self._preserve_gray_scale_btn)
        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self._crop_btn)

    def _setup_params(self):
        self.main_toolbar.addWidget(QLabel(" Stroke: "))
        self._stroke_spin = QSpinBox()
        self._stroke_spin.setRange(1, 100)
        self._stroke_spin.setValue(2)
        self.main_toolbar.addWidget(self._stroke_spin)

        self.main_toolbar.addSeparator()
        self._tolerance_lbl = QLabel(" Fill Tolerance ")
        self.main_toolbar.addWidget(self._tolerance_lbl)
        self._tolerance_spin = QSpinBox()
        self._tolerance_spin.setRange(0, 442)
        self._tolerance_spin.setValue(self._normal_tolerance_val)
        self.main_toolbar.addWidget(self._tolerance_spin)

    def setup_crop_toolbar(self):
        done_btn = QAction("Done", self)
        reset_btn = QAction("Reset", self)

        done_btn.triggered.connect(self.convert_main_bar)
        reset_btn.triggered.connect(self.reset_crop)

        self.crop_toolbar.addAction(done_btn)
        self.crop_toolbar.addAction(reset_btn)

    def convert_crop_bar(self):
        if self._cv_img is None:
            QMessageBox.warning(self, "Warn", "Nothing to crop!")
            return
        self._cv_img_cropped_dim = self._cv_img.copy()
        self.main_toolbar.hide()
        self.crop_toolbar.show()
        self.set_mode(Modes.crop)

    def convert_main_bar(self):
        self.confirm_crop()
        self._cv_img_cropped_dim = None
        self._crop_coords = None
        self.crop_toolbar.hide()
        self.main_toolbar.show()
        self.set_mode(Modes.fill)
        self.update_screen()

    def reset_crop(self):
        if self._cv_img is None:
            return
        self._cv_img_cropped_dim = self._cv_img.copy()
        self._crop_coords = None
        self.update_screen()

    def confirm_crop(self):
        if self._crop_coords is not None and self._cv_img is not None:
            reply = QMessageBox.question(
                self,
                "Confirm Crop",
                "Do you want to apply the crop to the main image?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                x1, y1, x2, y2 = self._crop_coords
                if x1 != x2 and y1 != y2:
                    self._cv_img = self._cv_img[y1:y2, x1:x2].copy()
                    self._history.clear()
                    self._his_indecies.clear()

    def toggle_pres_grayscale(self, _):
        t = not self._preserve_gray_scale
        self._preserve_gray_scale_btn.setChecked(t)
        self._preserve_gray_scale = t
        val = self._tolerance_spin.value()
        if t:
            self._normal_tolerance_val = val
            self._tolerance_lbl.setText(" Gray Tolerance ")
        else:
            self._gray_tolerance_val = val
            self._tolerance_lbl.setText(" Fill Tolerance ")

        x = self._gray_tolerance_val if t else self._normal_tolerance_val
        # print(f" x is : {x}")
        self._tolerance_spin.setValue(x)
        max_val = 255 if t else 411
        self._tolerance_spin.setRange(0, max_val)

    def toggle_cpu(self, _):
        self._reduce_cpu = not self._reduce_cpu
        self._cpu_btn.setChecked(self._reduce_cpu)

    def toggle_vis(self, _):
        self._visualize = not self._visualize
        self._vis_btn.setChecked(self._visualize)

    def toggle_load_gray(self, _):
        self._load_gray = not self._load_gray
        self._load_gray_btn.setChecked(self._load_gray)

    def select_color(self, _):
        color = QColorDialog.getColor()
        if color.isValid():
            self._color = (color.red(), color.green(), color.blue())

    def toggle_lock_ui(self):
        self._ui_locked = not self._ui_locked
        is_locked = self._ui_locked

        widgets: List[QAction | QSpinBox] = [
            self._tolerance_spin,
            self._stroke_spin,
            self._cpu_btn,
            self._vis_btn,
            self._fill_btn,
            self._draw_btn,
            self._crop_btn,
            self._undo_btn,
            self._color_btn,
            self._load_btn,
            self._load_gray_btn,
            self._save_btn,
        ]

        for w in widgets:
            w.setDisabled(is_locked)

        QApplication.processEvents()

    def load_image(self, _):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )

        if not file_path:
            return

        data = cv2.imread(file_path)
        if data is None:
            return
        self._his_indecies.clear()
        self._history.clear()
        if self._load_gray:
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            self._cv_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            self._cv_img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        self.update_screen()

    def save_image(self, _):
        if self._cv_img is None:
            return
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp);;All Files (*)",
        )

        if not file_path:
            return

        _, ext = os.path.splitext(file_path)
        if not ext:
            if "PNG" in selected_filter:
                file_path += ".png"
            elif "JPEG" in selected_filter:
                file_path += ".jpg"
            elif "BMP" in selected_filter:
                file_path += ".bmp"
            else:
                file_path += ".png"  

        data = cv2.cvtColor(self._cv_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, data)
        QMessageBox.information(
            self, "Success", f"Successfully saved file \n {file_path}"
        )

    def set_mode(self, mode: Modes):
        self._fill_btn.setChecked(mode == Modes.fill)
        self._draw_btn.setChecked(mode == Modes.draw)
        self._mode = mode

    def update_screen(self):
        if self._cv_img is None:
            return
        cv2_img = None
        if self._mode == Modes.crop and self._cv_img_cropped_dim is not None:
            cv2_img = self._cv_img_cropped_dim
        else:
            cv2_img = self._cv_img
        h, w, ch = cv2_img.shape

        q_img = QImage(
            cv2_img.data,  # type: ignore
            w,
            h,
            w * ch,
            QImage.Format.Format_RGB888,
        )

        q_pix = QPixmap.fromImage(q_img)
        self._image_area.setPixmap(q_pix)
        self._image_area.setFixedSize(q_pix.size())

    def _should_fill_pixel(
        self,
        y: int,
        x: int,
        target_color: tuple,
        base_gray: int,
        pres_gray: bool,
        tolerance: int,
        tolerance_sq: int,
        target_color_arr: np.ndarray,
    ) -> bool:
        """Helper to determine if a pixel matches the fill criteria."""
        if self._cv_img is None:
            return False

        if tolerance == 0:
            return tuple(self._cv_img[y][x]) == target_color

        if pres_gray:
            col = self._cv_img[y, x]
            current_gray = int(self._cv_img[y, x, 0])
            return bool(
                np.all(col == current_gray)
                and abs(int(base_gray) - current_gray) <= tolerance
            )

        pixel_color_arr = self._cv_img[y, x].astype(np.int32)
        dist_sq = np.sum((pixel_color_arr - target_color_arr) ** 2)
        return bool(dist_sq <= tolerance_sq)

    def _apply_fill_color(self, y: int, x: int, base_gray: int, pres_gray: bool):
        """Helper to apply the fill color to a pixel, respecting grayscale preservation."""
        if self._cv_img is None:
            return

        if pres_gray:
            ratio = self._cv_img[y, x, 0] / base_gray
            self._cv_img[y, x] = self.handle_pres_grayscale(self._color, ratio)
        else:
            self._cv_img[y][x] = self._color

    def flood_fill(self, y: int, x: int):
        if self._cv_img is None:
            return

        target_color = tuple(self._cv_img[y][x])
        if target_color == self._color:
            return

        pres_gray = False
        base_gray = int(self._cv_img[y, x, 0])
        # print(base_gray)
        # print(self._cv_img[y, x])
        if (
            self._preserve_gray_scale
            and base_gray != 0
            and np.all(self._cv_img[y, x] == base_gray)
        ):
            pres_gray = True

        self.toggle_lock_ui()
        h, w, _ = self._cv_img.shape
        self._history.append([])
        self._his_indecies.append(np.zeros((h, w), dtype=bool))
        history = self._history[-1]
        visited_mask = self._his_indecies[-1]
        visited_mask[y, x] = True

        stack: List[Tuple[int, int]] = [(y, x)]

        viz_count = 0
        tolerance = self._tolerance_spin.value()
        if tolerance == 411:
            history.append((-1, -1, self._cv_img.copy()))
            self._cv_img[:, :] = self._color
            self.toggle_lock_ui()
            return

        tolerance_sq = tolerance**2
        target_color_arr = np.array(target_color, dtype=np.int32)

        while stack:
            cy, cx = stack.pop()

            history.append((cy, cx, tuple(self._cv_img[cy, cx])))
            self._apply_fill_color(cy, cx, base_gray, pres_gray)

            if self._visualize:
                viz_count += 1
                if viz_count >= 500:
                    self.update_screen()
                    QApplication.processEvents()
                    viz_count = 0

            for dy, dx in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited_mask[ny, nx]:
                    if self._should_fill_pixel(
                        ny,
                        nx,
                        target_color,
                        base_gray,
                        pres_gray,
                        tolerance,
                        tolerance_sq,
                        target_color_arr,
                    ):
                        stack.append((ny, nx))
                        visited_mask[ny, nx] = True

        self.toggle_lock_ui()

    def handle_pres_grayscale(
        self, color: tuple[int, int, int], gray_level_ratio: float
    ):
        R = min(255, int(color[0] * gray_level_ratio))
        G = min(255, int(color[1] * gray_level_ratio))
        B = min(255, int(color[2] * gray_level_ratio))
        return (R, G, B)

    def undo(self, _):
        if self._cv_img is None or not self._history:
            QMessageBox.information(self, "Info", "No previous steps to undo!")
            return
        to_change = self._history.pop()
        self._his_indecies.pop()
        if to_change[0][0] == -1:
            self._cv_img[:] = to_change[0][2]
            self.update_screen()
            return

        for y, x, color in to_change:
            self._cv_img[y, x] = color
        self.update_screen()

    def draw(self, y: int, x: int):
        img = self._cv_img
        if img is None or not self._history:
            return

        h, w, _ = img.shape
        stroke = self._stroke_spin.value()

        y_min, y_max = max(0, y - stroke), min(h, y + stroke + 1)
        x_min, x_max = max(0, x - stroke), min(w, x + stroke + 1)

        c_mask = self._his_indecies[-1]
        n_mask = c_mask[y_min:y_max, x_min:x_max]
        not_painted = ~n_mask

        if np.any(not_painted):
            current_changes = self._history[-1]
            yy, xx = np.where(not_painted)
            for i, j in zip(yy, xx):
                gy, gx = i + y_min, j + x_min
                current_changes.append((gy, gx, tuple(img[gy, gx])))

            img[y_min:y_max, x_min:x_max][not_painted] = self._color
            n_mask[not_painted] = True

    def on_click(self, ev: QMouseEvent):
        if self._cv_img is None:
            return

        pos = ev.position()
        x, y = int(pos.x()), int(pos.y())
        h, w, _ = self._cv_img.shape

        if not (0 <= x < w and 0 <= y < h):
            return

        if self._mode == Modes.draw:
            self._handle_draw_click(y, x, h, w)
        elif self._mode == Modes.fill:
            self._handle_fill_click(y, x)
        elif self._mode == Modes.crop:
            self._handle_crop_click(x, y)

    def _handle_draw_click(self, y, x, h, w):
        self._history.append([])
        self._his_indecies.append(np.zeros((h, w), dtype=bool))
        self.draw(y, x)
        if not self._reduce_cpu:
            self.update_screen()

    def _handle_fill_click(self, y, x):
        if not self._ui_locked:
            self.flood_fill(y, x)
            self.update_screen()

    def _handle_crop_click(self, x, y):
        self.anchor = QPoint(x, y)
        self.rub_band.setGeometry(QRect(self.anchor, QSize()))
        self.rub_band.show()

    def on_move(self, ev: QMouseEvent):
        if self._cv_img is None or self._mode == Modes.fill:
            return

        pos = ev.position()
        x, y = int(pos.x()), int(pos.y())
        h, w, _ = self._cv_img.shape

        if not (0 <= x < w and 0 <= y < h):
            return

        if self._mode == Modes.draw:
            self.draw(y, x)
            if not self._reduce_cpu:
                self.update_screen()
        elif self._mode == Modes.crop:
            self.rub_band.setGeometry(QRect(self.anchor, QPoint(x, y)).normalized())

    def on_release(self, _):
        if self._mode == Modes.draw and self._reduce_cpu:
            self.update_screen()
        elif self._mode == Modes.crop:
            rect = self.rub_band.geometry()
            self.rub_band.hide()
            self.dim_crop(rect)

    def dim_crop(self, rect: QRect):
        if self._cv_img_cropped_dim is None or self._cv_img is None:
            return

        norm_rect = rect.normalized()
        h, w, _ = self._cv_img_cropped_dim.shape
        mask = np.ones((h, w), dtype=bool)

        if norm_rect.isValid():
            x1 = max(0, norm_rect.left())
            y1 = max(0, norm_rect.top())
            x2 = min(w, norm_rect.right())
            y2 = min(h, norm_rect.bottom())

            self._crop_coords = (x1, y1, x2, y2)
            mask[y1:y2, x1:x2] = False
        else:
            self._crop_coords = None

        self._cv_img_cropped_dim = self._cv_img.copy()

        if np.any(mask):
            pixels_to_dim = self._cv_img_cropped_dim[mask]
            weights = np.array([0.299, 0.587, 0.114])
            grayscale = np.dot(pixels_to_dim, weights).astype(np.uint8)
            self._cv_img_cropped_dim[mask] = np.stack([grayscale] * 3, axis=-1)

        self.update_screen()


if __name__ == "__main__":
    app = QApplication([])
    window = Editor()
    window.show()
    app.exec()