import sys
from typing import List, Optional, Tuple

# OpenCV for image processing, Numpy for fast grid/array math
import cv2
import numpy as np

# PyQt6 components for building the Window and handling User Input
from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtGui import QAction, QCursor, QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QRubberBand,
    QScrollArea,
    QToolBar,
)


# -------------------------------------------------------------------------
# BRIDGE CLASS: ClickableLabel
# -------------------------------------------------------------------------
# A standard QLabel (used to show images) doesn't "listen" for mouse clicks
# by default. We create this class to "catch" clicks, moves, and releases
# and pass that information to our main FloodFillApp class.
class ClickableLabel(QLabel):
    def __init__(self, parent: Optional[QMainWindow] = None):
        super().__init__(parent)
        self.app: Optional["FloodFillApp"] = None

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:
        # Triggered when user clicks the image
        if self.app and ev:
            self.app.on_image_click(ev)

    def mouseMoveEvent(self, ev: QMouseEvent | None) -> None:
        # Triggered when user drags the mouse (useful for the Crop Tool)
        if self.app and ev:
            self.app.on_image_move(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent | None) -> None:
        # Triggered when user lets go of the mouse button
        if self.app and ev:
            self.app.on_image_release(ev)


# -------------------------------------------------------------------------
# MAIN APPLICATION: FloodFillApp
# -------------------------------------------------------------------------
class FloodFillApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flood Fill & Image Editor (PyQt6)")
        self.setMinimumSize(800, 600)

        # Basic variables
        self.cv_image: Optional[np.ndarray] = (
            None  # The image stored as a 3D Grid (Height x Width x BGR)
        )
        self.current_color: Tuple[int, int, int] = (
            255,
            0,
            0,
        )  # Default: Blue (BGR format)
        self.history: List[
            np.ndarray
        ] = []  # A 'stack' to store old versions of the image for Undo
        self.mode = "FILL"  # Tracks if we are currently PAINTING or CROPPING
        self.preserve_gradients = False  # Toggle for the advanced shading feature

        self.setup_ui()

    def setup_ui(self):
        """Builds the buttons, toolbar, and the image display area."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # --- 1. Basic Actions ---
        load_action = QAction("1. Load Image", self)
        load_action.triggered.connect(self.load_image)
        toolbar.addAction(load_action)

        color_action = QAction("2. Choose Color", self)
        color_action.triggered.connect(self.choose_color)
        toolbar.addAction(color_action)

        save_action = QAction("3. Save Image", self)
        save_action.triggered.connect(self.save_image)
        toolbar.addAction(save_action)

        undo_action = QAction("Undo", self)
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)

        toolbar.addSeparator()

        # --- 2. Mode Toggles ---
        self.fill_mode_action = QAction("Fill Mode", self)
        self.fill_mode_action.setCheckable(True)
        self.fill_mode_action.setChecked(True)
        self.fill_mode_action.triggered.connect(lambda: self.set_mode("FILL"))
        toolbar.addAction(self.fill_mode_action)

        self.crop_mode_action = QAction("Crop Mode", self)
        self.crop_mode_action.setCheckable(True)
        self.crop_mode_action.triggered.connect(lambda: self.set_mode("CROP"))
        toolbar.addAction(self.crop_mode_action)

        toolbar.addSeparator()

        # --- 3. Advanced Feature Toggle ---
        self.grad_action = QAction("Preserve Gradients", self)
        self.grad_action.setCheckable(True)
        self.grad_action.triggered.connect(self.toggle_gradients)
        toolbar.addAction(self.grad_action)

        # --- 4. Central Image Area ---
        self.scroll_area = QScrollArea()
        self.image_label = ClickableLabel()
        self.image_label.app = self
        self.image_label.setText("Load an image to start")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)  # Allows scrolling if image is huge
        self.setCentralWidget(self.scroll_area)

        # Rubber band is the dotted rectangle you see when cropping
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self.image_label)
        self.origin = QPoint()

    def set_mode(self, mode: str):
        """Switches between Fill (Paint Bucket) and Crop (Scissors) modes."""
        self.mode = mode
        self.fill_mode_action.setChecked(mode == "FILL")
        self.crop_mode_action.setChecked(mode == "CROP")
        # Change cursor look to help user know which mode they are in
        if mode == "CROP":
            self.image_label.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def toggle_gradients(self, checked: bool):
        self.preserve_gradients = checked

    def load_image(self):
        """Opens a file dialog and reads the image using OpenCV."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg *.bmp)"
        )
        if not filepath:
            return

        self.cv_image = cv2.imread(filepath)
        if self.cv_image is None:
            QMessageBox.critical(self, "Error", "Could not load image.")
            return

        self.history = []  # Reset history for new image
        self.update_display()

    def save_image(self):
        """Saves the current modified image back to the hard drive."""
        if self.cv_image is None:
            QMessageBox.warning(self, "Warning", "No image to save.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg)"
        )
        if filepath:
            cv2.imwrite(filepath, self.cv_image)
            QMessageBox.information(self, "Success", "Image saved successfully.")

    def choose_color(self):
        """Opens a standard Color Picker window."""
        color = QColorDialog.getColor()
        if color.isValid():
            # Note: QColor gives us RGB, but OpenCV needs BGR
            self.current_color = (color.blue(), color.green(), color.red())

    def undo(self):
        """Restores the image to the state it was in before the last action."""
        if self.history:
            self.cv_image = self.history.pop()  # Take the last saved version
            self.update_display()
        else:
            QMessageBox.information(self, "Notice", "No previous steps to undo.")

    def update_display(self):
        """Converts the OpenCV Image (Array) into a format the Window can draw."""
        if self.cv_image is None:
            return

        # 1. OpenCV uses BGR, PyQt needs RGB. We swap them here.
        img_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w

        # 2. Create a QImage from the raw data.
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)  # type: ignore

        # 3. QPixmap is the actual "drawable" object for the label.
        # fromImage() makes a COPY of the data, which is safe for memory.
        pixmap = QPixmap.fromImage(q_img)

        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())

    def on_image_click(self, event: QMouseEvent):
        """Handles the logic when the user clicks on the image."""
        if self.cv_image is None:
            return

        # Get exact X,Y coordinate of the click relative to the image
        pos = event.position()
        x, y = int(pos.x()), int(pos.y())

        h, w = self.cv_image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return  # Click was outside image bounds

        if self.mode == "FILL":
            self.save_to_history()  # Save current state before we mess it up
            self.custom_flood_fill(x, y, self.current_color)
            self.update_display()
        elif self.mode == "CROP":
            # Start drawing the crop rectangle
            self.origin = QPoint(x, y)
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()

    def on_image_move(self, event: QMouseEvent):
        """Updates the visual rectangle while the user is dragging the mouse."""
        if self.mode == "CROP" and not self.origin.isNull():
            pos = event.position().toPoint()
            # normalized() ensures the box is valid even if dragged 'backwards'
            self.rubber_band.setGeometry(QRect(self.origin, pos).normalized())

    def on_image_release(self, _event: QMouseEvent):
        """Finalizes the crop action once the user lets go of the mouse."""
        if (
            self.mode == "CROP"
            and not self.origin.isNull()
            and self.cv_image is not None
        ):
            rect = self.rubber_band.geometry()
            self.rubber_band.hide()

            # If the selected area is big enough, crop it
            if rect.width() > 1 and rect.height() > 1:
                self.save_to_history()
                h, w = self.cv_image.shape[:2]
                # Slice the Numpy Array: image[startY : endY, startX : endX]
                x1, y1 = max(0, rect.x()), max(0, rect.y())
                x2, y2 = (
                    min(w, rect.x() + rect.width()),
                    min(h, rect.y() + rect.height()),
                )

                if x2 > x1 and y2 > y1:
                    self.cv_image = self.cv_image[y1:y2, x1:x2].copy()
                    self.update_display()
            self.origin = QPoint()

    def save_to_history(self):
        """Saves a copy of the current image to the undo list."""
        if self.cv_image is not None:
            # We only keep the last 20 steps to avoid using too much RAM
            if len(self.history) > 20:
                self.history.pop(0)
            self.history.append(self.cv_image.copy())

    def custom_flood_fill(
        self, start_x: int, start_y: int, new_color: Tuple[int, int, int]
    ):
        """
        THE CORE ALGORITHM: Manually implemented DFS (Depth-First Search) using a Loop.
        Logic: Use a Stack (LIFO) to explore the deepest neighbors first.
        """
        if self.cv_image is None:
            return

        h, w = self.cv_image.shape[:2]
        target_color = self.cv_image[
            start_y, start_x
        ].copy()  # The color we want to replace
        new_color_arr = np.array(new_color, dtype=np.uint8)

        # If the pixel clicked is already the new color, we stop immediately.
        if np.array_equal(target_color, new_color_arr):
            return

        # Tolerance: allows the fill to spread across slightly different shades.
        tolerance = 30 if self.preserve_gradients else 0

        # 'visited' prevents infinite loops and re-processing
        visited = np.zeros((h, w), dtype=bool)
        # 'stack' for DFS (LIFO)
        stack = [(start_x, start_y)]
        visited[start_y, start_x] = True

        # Pre-calculating brightness for the 'Preserve Gradients' feature
        target_lum = 1.0

        def get_lum(bgr: np.ndarray) -> float:
            return float(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2])

        if self.preserve_gradients:
            target_lum = get_lum(target_color)
            if target_lum == 0:
                target_lum = 1.0

        # --- THE MAIN LOOP (Iterative DFS) ---

        while stack:
            cx, cy = stack.pop()  # LIFO behavior for DFS

            if self.preserve_gradients:
                current_pixel = self.cv_image[cy, cx]
                ratio = get_lum(current_pixel) / target_lum
                pixel_new_color = np.clip(np.array(new_color) * ratio, 0, 255).astype(
                    np.uint8
                )
                self.cv_image[cy, cx] = pixel_new_color
            else:
                self.cv_image[cy, cx] = new_color_arr

            # Check the 4 neighbors: Up, Down, Left, Right
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    visited[ny, nx] = True
                    neighbor_color = self.cv_image[ny, nx]

                    match: bool = False
                    if tolerance == 0:
                        match = bool(np.array_equal(neighbor_color, target_color))
                    else:
                        dist = np.linalg.norm(
                            neighbor_color.astype(float) - target_color.astype(float)
                        )
                        match = bool(dist <= tolerance)

                    if match:
                        stack.append((nx, ny))


# Standard Python boilerplate to start the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FloodFillApp()
    window.show()
    sys.exit(app.exec())
