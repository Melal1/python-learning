import sys

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


def compute_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def run_k_means(data: np.ndarray, clusters: int, iterations: int = 1):
    total_pixels = len(data)
    use_replacement = total_pixels < clusters
    initial_indices = np.random.choice(total_pixels, clusters, replace=use_replacement)
    centers = data[initial_indices].copy()
    final_labels = np.zeros(total_pixels, dtype=np.int32)

    print(f"Starting K-Means: {clusters} clusters, {iterations} iterations...")

    for i in range(iterations):
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

        print(f"Iteration {i + 1}/{iterations} complete.")

    return centers, final_labels


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Stores the path of the selected image file
        self.image_path = None
        # Stores the image data loaded via OpenCV (NumPy array)
        self.cv_img = None
        # Calls the method to set up the Graphical User Interface
        self._init_ui()

    def _init_ui(self):
        # Sets the text displayed in the window's title bar
        self.setWindowTitle("K-Means Image Indexer")
        # Ensures the window has a minimum width of 500 pixels
        self.setMinimumWidth(500)

        # Creates a generic widget to serve as the container for all UI elements
        central_widget = QWidget()
        # Tells the QMainWindow to use this widget as its primary content area
        self.setCentralWidget(central_widget)
        # Creates a vertical box layout (elements stacked top to bottom) for the container
        main_layout = QVBoxLayout(central_widget)

        # Creates a text label to act as a title/header
        header = QLabel("Image Quantization Tool")
        # Applies CSS-like styling to the header (font size, weight, and vertical margin)
        header.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px 0;")
        # Centers the text within the label's space
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Adds the header label to the vertical stack
        main_layout.addWidget(header)

        # Creates a button for file selection
        self.btn_select = QPushButton("Select JPG Image")
        # Sets the "hint" text that appears when hovering over the button
        self.btn_select.setToolTip("Choose a JPEG image to process.")
        # Connects the button's 'clicked' event to the 'on_select_image' method
        self.btn_select.clicked.connect(self.on_select_image)
        # Adds the button to the vertical layout
        main_layout.addWidget(self.btn_select)

        # Creates a secondary vertical layout to group info labels together
        info_group = QVBoxLayout()
        # Label to show the file path; defaults to 'No file selected'
        self.lbl_path = QLabel("Path: No file selected")
        # Allows the path text to wrap onto multiple lines if it's too long
        self.lbl_path.setWordWrap(True)
        # Label to show image resolution (width x height)
        self.lbl_size = QLabel("Dimensions: N/A")

        # Creates a horizontal layout for the iteration input (Label + SpinBox)
        iter_box = QHBoxLayout()
        # Adds a descriptive label for the spin box
        iter_box.addWidget(QLabel("Optimization Iterations:"))
        # Creates a numeric input field (SpinBox)
        self.spin_iterations = QSpinBox()
        # Limits the input range between 1 and 100
        self.spin_iterations.setRange(1, 100)
        # Sets the default starting value to 1
        self.spin_iterations.setValue(1)
        # Adds the spin box to the horizontal layout
        iter_box.addWidget(self.spin_iterations)
        # Adds elastic space to push the label and spin box to the left
        iter_box.addStretch()

        # Label to provide feedback on the current app state (Ready, Processing, etc.)
        self.lbl_status = QLabel("Status: Ready")
        # Styles the status label with a specific color and italic font
        self.lbl_status.setStyleSheet("color: #555; font-style: italic;")

        # Adds all labels and the horizontal input box into the info group layout
        info_group.addWidget(self.lbl_path)
        info_group.addWidget(self.lbl_size)
        info_group.addLayout(iter_box)
        info_group.addWidget(self.lbl_status)
        # Adds the entire info group to the main vertical layout
        main_layout.addLayout(info_group)

        # Adds elastic space that grows to push items above it up and items below it down
        main_layout.addStretch()

        # Creates the main action button to start the K-Means process
        self.btn_process = QPushButton("Process & Save Indexed Image")
        # Disables the button by default (it stays gray until an image is loaded)
        self.btn_process.setEnabled(False)
        # Sets a fixed minimum height to make the button look prominent
        self.btn_process.setMinimumHeight(50)
        # Styles the button with a green background, white bold text, and rounded corners
        self.btn_process.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; border-radius: 5px; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        # Connects the click event to the 'on_process_image' method
        self.btn_process.clicked.connect(self.on_process_image)
        # Adds the button to the layout
        main_layout.addWidget(self.btn_process)

        # Creates an exit button
        self.btn_exit = QPushButton("Exit Application")
        # Connects the button to the built-in 'close' method of the window
        self.btn_exit.clicked.connect(self.close)
        # Adds the button to the layout
        main_layout.addWidget(self.btn_exit)

    def on_select_image(self):
        # Opens a system file dialog to pick a .jpg or .jpeg file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select JPEG Image", "", "JPG Files (*.jpg *.jpeg)"
        )

        # If the user didn't cancel the dialog
        if file_path:
            # Uses OpenCV to read the image file into a NumPy array
            self.cv_img = cv2.imread(file_path)
            # If reading was successful
            if self.cv_img is not None:
                self.image_path = file_path
                # Gets height, width, and number of channels from the array shape
                h, w, _ = self.cv_img.shape
                # Updates the UI labels with the new file information
                self.lbl_path.setText(f"Path: {file_path}")
                self.lbl_size.setText(f"Dimensions: {w}x{h}")
                self.lbl_status.setText("Status: Image Loaded. Ready.")
                self.lbl_status.setStyleSheet("color: blue;")
                # Enables the process button now that we have data
                self.btn_process.setEnabled(True)
            else:
                # Shows a popup error if the file format is invalid or unreadable
                QMessageBox.critical(
                    self, "Load Error", "Failed to read the image file."
                )
                self.btn_process.setEnabled(False)

    def on_process_image(self):
        # Safety check to ensure an image is actually loaded
        if self.cv_img is None:
            QMessageBox.warning(self, "Error", "No image loaded to process.")
            return

        # Prevents the user from clicking buttons while processing
        self._toggle_ui_lock(True)
        # Updates status label to inform the user that work is happening
        self.lbl_status.setText(
            "Status: Processing... Please wait (Check terminal for details)"
        )
        self.lbl_status.setStyleSheet("color: orange; font-weight: bold;")
        # Forces PyQt to update the screen immediately (so the status change shows up)
        QApplication.processEvents()

        try:
            # Gets height and width dimensions
            h, w = self.cv_img.shape[:2]
            # Flattens the 2D image into a long list of pixels (N, 3) for the algorithm
            data = self.cv_img.reshape((-1, 3)).astype(np.float32)

            # Retrieves the number of iterations from the spin box
            iters = self.spin_iterations.value()
            # Runs the custom K-Means function defined at the top of the file
            centers, labels = run_k_means(data, clusters=256, iterations=iters)

            # Opens a save dialog to ask the user where to put the resulting PNG
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Indexed PNG", "indexed_result.png", "PNG Files (*.png)"
            )

            # If the user selected a save path
            if save_path:
                # Calls the helper to format and write the PNG file
                self._save_indexed_png(save_path, centers, labels, w, h)
                # Updates UI for success
                self.lbl_status.setText(
                    f"Status: Success! Saved to {save_path.split('/')[-1]}"
                )
                self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
                QMessageBox.information(
                    self, "Task Complete", f"Indexed image saved to:\n{save_path}"
                )
            else:
                # Handle case where user clicked 'Cancel' on the save dialog
                self.lbl_status.setText("Status: Save cancelled.")
                self.lbl_status.setStyleSheet("color: blue;")

        except Exception as e:
            # Catches any unexpected crashes and shows the error message
            QMessageBox.critical(
                self, "Error", f"An unexpected error occurred: {str(e)}"
            )
            self.lbl_status.setText("Status: Processing failed.")
            self.lbl_status.setStyleSheet("color: red;")

        finally:
            # Unlocks the UI buttons whether the process succeeded or failed
            self._toggle_ui_lock(False)

    def _save_indexed_png(self, path, centers, labels, w, h):
        # OpenCV uses BGR; Pillow uses RGB. We flip the color channel order [::-1]
        palette_rgb = centers.astype(np.uint8)[:, ::-1]
        # Flatten the 256 colors into a single long list for the PNG palette
        palette_flat = palette_rgb.flatten().tolist()

        # Reshapes the flat label list back into the image's original dimensions
        pixel_indices = labels.reshape((h, w)).astype(np.uint8)

        # Creates a new Pillow image object in 'P' (Palette/Indexed) mode
        img = Image.fromarray(pixel_indices, mode="P")
        # Attaches our calculated 256 colors as the image's color palette
        img.putpalette(palette_flat)
        # Writes the image to the disk
        img.save(path)

    def _toggle_ui_lock(self, locked: bool):
        # Enables or disables all interactive widgets to prevent double-processing
        self.btn_process.setEnabled(not locked)
        self.btn_select.setEnabled(not locked)
        self.spin_iterations.setEnabled(not locked)
        self.btn_exit.setEnabled(not locked)


if __name__ == "__main__":
    # Creates the application object
    app = QApplication(sys.argv)

    # Sets the visual theme to 'Fusion' for a consistent look across OSs
    # app.setStyle("Fusion")

    # Creates and shows the main window
    window = ImageProcessorApp()
    window.show()
    # Starts the application event loop
    sys.exit(app.exec())
