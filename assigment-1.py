from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# def compute_distance(a: np.ndarray, b: np.ndarray) -> float:
#     return float(np.sqrt(np.sum((a - b) ** 2)))


def compute_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def run_k_means(
    data: np.ndarray,
    clusters: int,
    iterations: int = 1,
    on_iter: Optional[Callable[[int], None]] = None,
):
    total_pixels = len(data)
    is_uniqe = total_pixels < clusters
    initial_indices = np.random.choice(total_pixels, clusters, replace=is_uniqe)

    centers = data[initial_indices].copy()
    final_labels = np.zeros(total_pixels, dtype=np.int32)

    for i in range(iterations):
        if on_iter:
            on_iter(i + 1)
        groups: list[list[np.ndarray]] = [[] for _ in range(clusters)]
        for pixel_idx, pixel in enumerate(data):
            distances = [compute_distance(c, pixel) for c in centers]
            closest_idx = np.argmin(distances)
            groups[closest_idx].append(pixel)
            final_labels[pixel_idx] = closest_idx
        for j in range(clusters):
            if len(groups[j]) == 0:
                centers[j] = data[np.random.randint(0, total_pixels)]
            else:
                centers[j] = np.mean(groups[j], axis=0)
    return centers, final_labels


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.cv_img = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("Multimedia assigment")
        self.setMinimumWidth(500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        header = QLabel("24-bit to indexed 8-bit imgage")
        header.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px 0;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout.addWidget(header)

        self.btn_select = QPushButton("Select JPG image")
        self.btn_select.setToolTip("Choose a JPEG image to process.")
        self.btn_select.clicked.connect(self.on_select_image)

        main_layout.addWidget(self.btn_select)

        info_group = QVBoxLayout()

        self.lbl_path = QLabel("Path: no file selected")
        self.lbl_path.setWordWrap(True)
        self.lbl_size = QLabel("Dimensions: N/A")

        scr_box = QHBoxLayout()
        scr_box.addWidget(QLabel("Iterations:"))

        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(1, 100)
        self.spin_iterations.setValue(1)

        scr_box.addWidget(self.spin_iterations)
        scr_box.addStretch()

        self.lbl_status = QLabel("Status: ready")
        self.lbl_status.setStyleSheet("color: #555; font-style: italic;")

        info_group.addWidget(self.lbl_path)
        info_group.addWidget(self.lbl_size)
        info_group.addLayout(scr_box)
        info_group.addWidget(self.lbl_status)
        main_layout.addLayout(info_group)
        main_layout.addStretch()

        self.btn_process = QPushButton("Process & save indexed image")
        self.btn_process.setEnabled(False)
        self.btn_process.setMinimumHeight(50)

        self.btn_process.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; border-radius: 5px; }
            QPushButton:disabled { background-color: #cccccc; }
        """)

        self.btn_process.clicked.connect(self.on_process_image)

        main_layout.addWidget(self.btn_process)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.on_exit)

        main_layout.addWidget(self.btn_exit)

    def on_exit(self):
        self.close()
        exit()

    def on_select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select JPEG image", "", "JPG Files (*.jpg *.jpeg)"
        )
        if file_path:
            self.cv_img = cv2.imread(file_path)
            if self.cv_img is not None:
                self.image_path = file_path
                h, w, _ = self.cv_img.shape
                self.lbl_path.setText(f"Path: {file_path}")
                self.lbl_size.setText(f"Dimensons: {w}x{h}")
                self.lbl_status.setText("Status: Image loaded, Ready.")
                self.lbl_status.setStyleSheet("color: blue;")
                self.btn_process.setEnabled(True)
            else:
                QMessageBox.critical(
                    self, "Load Error", "Failed to read the image file."
                )
                self.btn_process.setEnabled(False)

    def update_progress_label(self, current_iteration: int):
        perc: float = current_iteration / self.spin_iterations.value() * 100
        self.lbl_status.setText(f"Processing: progress {perc:.2f} %")
        QApplication.processEvents()

    def on_process_image(self):
        if self.cv_img is None:
            QMessageBox.warning(self, "Error", "No image loaded to process.")
            return
        self._toggle_ui_lock(True)
        self.lbl_status.setText("Status: Processing... ")
        self.lbl_status.setStyleSheet("color: orange; font-weight: bold;")
        QApplication.processEvents()
        try:
            h, w = self.cv_img.shape[:2]
            data = self.cv_img.reshape((-1, 3)).astype(np.float32)
            iters = self.spin_iterations.value()
            centers, labels = run_k_means(
                data, clusters=256, iterations=iters, on_iter=self.update_progress_label
            )
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Indexed PNG", "indexed_result.png", "PNG Files (*.png)"
            )
            if save_path:
                self._save_indexed_png(save_path, centers, labels, w, h)
                self.lbl_status.setText(
                    f"Status: Success! Saved to {save_path.split('/')[-1]}"
                )
                self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
                QMessageBox.information(
                    self, "Task Complete", f"Indexed image saved to:\n{save_path}"
                )
            else:
                self.lbl_status.setText("Status: Save cancelled.")
                self.lbl_status.setStyleSheet("color: blue;")
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An unexpected error occurred: {str(e)}"
            )
            self.lbl_status.setText("Status: Processing failed.")
            self.lbl_status.setStyleSheet("color: red;")
        finally:
            self._toggle_ui_lock(False)

    def _save_indexed_png(self, path, centers, labels, w, h):
        palette_rgb = centers.astype(np.uint8)[:, ::-1]
        palette_flat = palette_rgb.flatten().tolist()
        pixel_indices = labels.reshape((h, w)).astype(np.uint8)
        img = Image.fromarray(pixel_indices, mode="P")
        img.putpalette(palette_flat)
        img.save(path)

    def _toggle_ui_lock(self, locked: bool):
        self.btn_process.setEnabled(not locked)
        self.btn_select.setEnabled(not locked)
        self.spin_iterations.setEnabled(not locked)
        # self.btn_exit.setEnabled(not locked) // jst un comment this if you want to lock the exit btn when prossesing


app = QApplication([])
app.setStyle("Fusion")
window = ImageProcessorApp()
window.show()
app.exec()
