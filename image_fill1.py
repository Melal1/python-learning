from enum import Enum
from typing import List, Optional, Tuple

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
        # Vars
        self.setWindowTitle("Image flood fill")
        self._mode: Modes = Modes.fill
        self._cv_img: Optional[np.ndarray] = None
        self._cv_img_croped_dim: Optional[np.ndarray] = None
        self._history: List[List[Tuple[int, int, Tuple[int, int, int]]]] = []
        self._his_indecies = []
        self._color = (255, 0, 0)
        self._reduce_cpu = False
        self._visualize = False
        self._crop_coords: Optional[Tuple[int, int, int, int]] = None
        self._ui_locked = False

        self._scroll_area = QScrollArea()
        self._image_area = ClickableLabel()

        self._image_area.app = self
        self._image_area.setText("Load an image to start")
        self._image_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._scroll_area.setWidget(self._image_area)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self._scroll_area)

        self.main_toolbar: QToolBar = QToolBar("Main ToolBar")
        self.crop_toolbar: QToolBar = QToolBar("Crop ToolBar")

        self.setup_main_tool_bar()
        self.setup_crop_toolbar()

        self.addToolBar(self.main_toolbar)
        self.addToolBar(self.crop_toolbar)
        self.crop_toolbar.hide()

        self.rub_band = QRubberBand(QRubberBand.Shape.Rectangle, self._image_area)
        self.anchor = QPoint()

    def setup_main_tool_bar(self):
        self._load_btn = QAction("Load", self)
        self._save_btn = QAction("Save", self)
        self._color_btn = QAction("Color", self)
        self._undo_btn = QAction("Undo", self)
        self._fill_btn = QAction("Fill", self)
        self._draw_btn = QAction("Draw", self)
        self._cpu_btn = QAction(" Reduce Cpu Usage", self)
        self._vis_btn = QAction(" Visualize", self)

        self._fill_btn.setCheckable(True)
        self._draw_btn.setCheckable(True)
        self._cpu_btn.setCheckable(True)
        self._vis_btn.setCheckable(True)

        self._fill_btn.setChecked(True)
        self._cpu_btn.setChecked(self._reduce_cpu)
        self._vis_btn.setChecked(self._visualize)

        self._load_btn.triggered.connect(self.load_image)
        self._save_btn.triggered.connect(self.save_image)
        self._color_btn.triggered.connect(self.select_color)
        self._undo_btn.triggered.connect(self.undo)
        self._fill_btn.triggered.connect(lambda: self.set_mode(Modes.fill))
        self._draw_btn.triggered.connect(lambda: self.set_mode(Modes.draw))
        self._cpu_btn.triggered.connect(self.toggle_cpu)
        self._vis_btn.triggered.connect(self.toggle_vis)

        self.main_toolbar.addActions(
            [
                self._load_btn,
                self._save_btn,
                self._color_btn,
                self._fill_btn,
                self._draw_btn,
                self._undo_btn,
            ]
        )

        self.main_toolbar.addSeparator()
        self.main_toolbar.addWidget(QLabel(" Stroke: "))

        self._stroke_spin = QSpinBox()
        self._stroke_spin.setRange(1, 100)
        self._stroke_spin.setValue(2)
        self.main_toolbar.addWidget(self._stroke_spin)

        self.main_toolbar.addSeparator()
        self.main_toolbar.addWidget(QLabel(" Fill Tolerance: "))
        self._tolerance_spin = QSpinBox()
        self._tolerance_spin.setRange(
            0, int(np.linalg.norm(np.array([0, 0, 0]) - np.array([255, 255, 255])))
        )
        self._tolerance_spin.setValue(10)
        self.main_toolbar.addWidget(self._tolerance_spin)

        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self._cpu_btn)

        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self._vis_btn)

        self._crop_btn = QAction("Crop", self)
        self.main_toolbar.addSeparator()
        self._crop_btn.triggered.connect(self.convert_crop_bar)
        self.main_toolbar.addAction(self._crop_btn)

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
        self._cv_img_croped_dim = self._cv_img.copy()
        self.main_toolbar.hide()
        self.crop_toolbar.show()
        self.set_mode(Modes.crop)

    def convert_main_bar(self):
        self.confirm_crop()
        self._cv_img_croped_dim = None
        self._crop_coords = None
        self.crop_toolbar.hide()
        self.main_toolbar.show()
        self.set_mode(Modes.fill)
        self.update_screen()

    def reset_crop(self):
        if self._cv_img is None:
            return
        self._cv_img_croped_dim = self._cv_img.copy()
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

    def toggle_cpu(self, _):
        self._reduce_cpu = not self._reduce_cpu
        self._cpu_btn.setChecked(self._reduce_cpu)

    def toggle_vis(self, _):
        self._visualize = not self._visualize
        self._vis_btn.setChecked(self._visualize)

    def select_color(self, _):
        color = QColorDialog.getColor()
        if color.isValid():
            self._color = (color.red(), color.green(), color.blue())

    def toggle_lock_ui(self):
        self._ui_locked = not self._ui_locked
        m = self._ui_locked
        self._tolerance_spin.setDisabled(m)
        self._stroke_spin.setDisabled(m)
        self._cpu_btn.setDisabled(m)
        self._vis_btn.setDisabled(m)
        self._fill_btn.setDisabled(m)
        self._draw_btn.setDisabled(m)
        self._crop_btn.setDisabled(m)
        self._undo_btn.setDisabled(m)
        self._color_btn.setDisabled(m)
        self._load_btn.setDisabled(m)
        self._save_btn.setDisabled(m)
        QApplication.processEvents()

    def load_image(self, _):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "", "", "Jpeg images ( *.jpg *.jpeg )"
        )

        if not file_path:
            return

        data = cv2.imread(file_path)
        if data is None:
            return
        self._his_indecies.clear()
        self._history.clear()
        self._cv_img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        self.update_screen()

    def save_image(self, _):
        if self._cv_img is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save file", "", "Images (*.jpg *.jpeg)"
        )

        if not file_path:
            return

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
        cv2_img: Optional[np.ndarray] = None
        if self._mode == Modes.crop and self._cv_img_croped_dim is not None:
            cv2_img = self._cv_img_croped_dim
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

    def flood_fill(self, y: int, x: int):
        if self._cv_img is None:
            return

        target_color = tuple(self._cv_img[y][x])
        if target_color == self._color:
            return

        self.toggle_lock_ui()
        self._history.append([])
        self._his_indecies.append(set())
        current_changes = self._history[-1]
        current_indices = self._his_indecies[-1]

        stack: List[Tuple[int, int]] = [(y, x)]
        h, w, _ = self._cv_img.shape

        count = 0
        while stack:
            cy, cx = stack.pop()

            if (cy, cx) in current_indices:
                continue

            current_changes.append((cy, cx, tuple(self._cv_img[cy][cx])))
            current_indices.add((cy, cx))

            self._cv_img[cy][cx] = self._color

            if self._visualize:
                count += 1
                if count >= 500:
                    self.update_screen()
                    QApplication.processEvents()
                    count = 0

            if self._tolerance_spin.value == 0:
                for dy, dx in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if (ny, nx) not in current_indices:
                            if tuple(self._cv_img[ny][nx]) == target_color:
                                stack.append((ny, nx))
            else:
                for dy, dx in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if (ny, nx) not in current_indices:
                            if (
                                np.linalg.norm(target_color - self._cv_img[ny][nx])
                                <= self._tolerance_spin.value()
                            ):
                                stack.append((ny, nx))

        self.toggle_lock_ui()

    def undo(self, _):
        if self._cv_img is None or not self._history:
            QMessageBox.information(self, "Info", "No previous steps to undo!")
            return
        to_change = self._history.pop()
        self._his_indecies.pop()
        for y, x, color in to_change:
            self._cv_img[y][x] = color
        self.update_screen()

    def draw(self, y: int, x: int):
        # Guard clause with local variable for LSP happiness
        img = self._cv_img
        if img is None or not self._history:
            return

        h, w, _ = img.shape
        # Assuming self._stroke_spin is a QSpinBox
        strok = self._stroke_spin.value()

        # 1. Define ROI boundaries (inclusive/exclusive logic for NumPy)
        y_min, y_max = max(0, y - strok), min(h, y + strok + 1)
        x_min, x_max = max(0, x - strok), min(w, x + strok + 1)

        # 2. Reference the current stroke's boolean mask (H x W)
        current_mask = self._his_indecies[-1]

        # Get a view of the area we are currently painting
        roi_mask = current_mask[y_min:y_max, x_min:x_max]

        # Find pixels in this square that haven't been touched in this stroke yet
        unpainted_in_roi = ~roi_mask

        if np.any(unpainted_in_roi):
            # 3. Log History (Store original pixels for Undo)
            current_changes = self._history[-1]

            # Get local coordinates of new pixels
            yy, xx = np.where(unpainted_in_roi)

            # This loop only runs for NEWLY touched pixels, making it much faster
            for i, j in zip(yy, xx):
                gy, gx = i + y_min, j + x_min
                # Save: (y, x, (B, G, R))
                current_changes.append((gy, gx, tuple(img[gy, gx])))

            # 4. Vectorized Painting (C-speed)
            # Update the actual image pixels
            img[y_min:y_max, x_min:x_max][unpainted_in_roi] = self._color

            # Update the mask so we don't process these pixels again during this stroke
            roi_mask[unpainted_in_roi] = True

    def on_click(self, ev: QMouseEvent):
        if self._cv_img is None:
            return

        # Get coordinates
        x, y = int(ev.position().x()), int(ev.position().y())
        h, w, _ = self._cv_img.shape

        if not (0 <= x < w and 0 <= y < h):
            return

        if self._mode == "draw":
            # Start new history entry
            self._history.append([])

            # Initialize a fresh boolean mask for this specific stroke
            # This prevents redrawing/re-logging the same pixel 100 times per second
            self._his_indecies.append(np.zeros((h, w), dtype=bool))

            self.draw(y, x)
            if not self._reduce_cpu:
                self.update_screen()
            return

        if self._mode == "fill":
            self.flood_fill(y, x)
            self.update_screen()
            return

        # Rubber band logic
        self.anchor = QPoint(x, y)
        self.rub_band.setGeometry(QRect(self.anchor, QSize()))
        self.rub_band.show()

    def on_move(self, ev: QMouseEvent):
        if self._cv_img is None or self._mode == "fill":
            return

        x, y = int(ev.position().x()), int(ev.position().y())
        h, w, _ = self._cv_img.shape

        # Boundary check
        if not (0 <= x < w and 0 <= y < h):
            return

        if self._mode == "draw":
            self.draw(y, x)
            if not self._reduce_cpu:
                # This schedules a repaint; if draw() is fast,
                # Qt won't feel the need to compress events.
                self.update_screen()
            return

        # Rubber band update
        self.rub_band.setGeometry(QRect(self.anchor, QPoint(x, y)).normalized())

    def draw(self, y: int, x: int):
        # Guard clause
        img = self._cv_img
        if img is None or not self._history:
            return

        h, w, _ = img.shape
        strok = self._strok

        # 1. Define ROI boundaries (inclusive/exclusive logic for NumPy)
        y_min, y_max = max(0, y - strok), min(h, y + strok + 1)
        x_min, x_max = max(0, x - strok), min(w, x + strok + 1)

        # 2. Reference the current stroke's boolean mask (H x W)
        current_mask = self._his_indecies[-1]

        # Get a view of the area we are currently painting
        roi_mask = current_mask[y_min:y_max, x_min:x_max]

        # Find pixels in this square that haven't been touched in this stroke yet
        unpainted_in_roi = ~roi_mask

        if np.any(unpainted_in_roi):
            # 3. Log History (Store original pixels for Undo)
            current_changes = self._history[-1]

            # Get local coordinates of new pixels
            yy, xx = np.where(unpainted_in_roi)

            # This loop only runs for NEWLY touched pixels
            for i, j in zip(yy, xx):
                gy, gx = i + y_min, j + x_min
                current_changes.append((gy, gx, tuple(img[gy, gx])))

            # 4. Vectorized Painting (C-speed)
            # Update the actual image pixels
            img[y_min:y_max, x_min:x_max][unpainted_in_roi] = self._color

            # Update the mask so we don't process these pixels again during this stroke
            roi_mask[unpainted_in_roi] = True
    #
    def on_click(self, ev: QMouseEvent):
        if self._cv_img is None:
            return
        x, y = int(ev.position().x()), int(ev.position().y())
        h, w, _ = self._cv_img.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        if self._mode == Modes.draw:
            self._history.append([])
            # Initialize a fresh boolean mask for this specific stroke
            self._his_indecies.append(np.zeros((h, w), dtype=bool))
            self.draw(y, x)
            if not self._reduce_cpu:
                self.update_screen()
            return
        if self._mode == Modes.fill:
            self.flood_fill(y, x)
            self.update_screen()
            return

        self.anchor = QPoint(x, y)
        self.rub_band.setGeometry(QRect(self.anchor, QSize()))
        self.rub_band.show()
    #
    # def on_move(self, ev: QMouseEvent):
    #     if self._cv_img is None or self._mode == Modes.fill:
    #         return
    #     x, y = int(ev.position().x()), int(ev.position().y())
    #     h, w, _ = self._cv_img.shape
    #     if not (0 <= x < w and 0 <= y < h):
    #         return
    #     if self._mode == Modes.draw:
    #         self.draw(y, x)
    #         if not self._reduce_cpu:
    #             self.update_screen()
    #         return
    #     self.rub_band.setGeometry(QRect(self.anchor, QPoint(x, y)).normalized())
    #
    def on_release(self, ev: QMouseEvent):
        if self._mode == Modes.draw and self._reduce_cpu:
            self.update_screen()
        if self._mode == Modes.crop:
            rect = self.rub_band.geometry()
            self.rub_band.hide()
            self.dim_crop(rect)

    def dim_crop(self, rect: QRect):
        if self._cv_img_croped_dim is None or self._cv_img is None:
            return

        norm_rect = rect.normalized()
        h, w, _ = self._cv_img_croped_dim.shape

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

        self._cv_img_croped_dim = self._cv_img.copy()

        if np.any(mask):
            pixels_to_dim = self._cv_img_croped_dim[mask]
            weights = np.array([0.299, 0.587, 0.114])
            grayscale = np.dot(pixels_to_dim, weights).astype(np.uint8)
            self._cv_img_croped_dim[mask] = np.stack([grayscale] * 3, axis=-1)

        self.update_screen()


app = QApplication([])
window = Editor()
window.show()
app.exec()
