import sys
# Import core Qt widgets used for the GUI layout and basic controls.
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QRubberBand, QToolBar,
    QSizePolicy, QPushButton, QScrollArea
)
# Import classes related to 2D graphics rendering and image representation.
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage
# Import core non-GUI Qt types like enums (Qt), 2D points, rectangles, and sizes.
from PyQt6.QtCore import Qt, QPoint, QRect, QSize
import cv2
import numpy as np

# ==============================================================================
# ImageLabel Class
# ==============================================================================
# In C++ Qt, this would be a class inheriting from QLabel. 
# We subclass QLabel to intercept mouse events directly on the image display area.
class ImageLabel(QLabel):
    def __init__(self, parent=None):
        # Call the base class constructor. Equivalent to `QLabel(parent)` in C++ initializer list.
        super().__init__(parent)
        
        # Align the contents (the image) to the top-left corner of the label.
        # Uses bitwise OR just like in C++ Qt (Qt::AlignTop | Qt::AlignLeft).
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Ignore size policies so the label can resize freely within the scroll area.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        
        # Disable automatic scaling. We want a 1:1 pixel mapping for accurate drawing/cropping.
        self.setScaledContents(False)

        # State machine variables to handle current interaction mode.
        self.mode = "view" # Valid modes: "view", "draw", "crop"
        
        # State variables for the "draw" mode.
        self.drawing = False # True while the left mouse button is held down.
        self.last_point = QPoint() # Stores the previous mouse position to draw line segments.
        
        # QRubberBand provides a native visual rectangle for selections (like cropping).
        # We pass `self` as the parent so its memory lifecycle is tied to this widget.
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        
        # State variable for the "crop" mode to store the initial click position.
        self.origin = QPoint()
        
        # QPixmap acts as the backing store for the image (hardware-optimized, unlike QImage).
        self._pixmap = QPixmap()
        
        # Default drawing parameters.
        self.pen_color = QColor(255, 0, 0) # RGB: Red
        self.pen_width = 3

    def set_mode(self, mode):
        """Changes the interaction mode and cleans up UI state."""
        self.mode = mode
        # If we switch away from crop mode, ensure the selection rectangle is hidden.
        if mode != "crop":
            self.rubber_band.hide()

    def load_image(self, file_path):
        """Loads an image from disk using OpenCV into the backing QPixmap."""
        # Read the image using OpenCV. cv2.imread returns a NumPy array in BGR format.
        # This is the equivalent of cv::imread in C++.
        cv_img = cv2.imread(file_path)
        if cv_img is None:
            return

        # OpenCV uses BGR layout in memory, but QImage expects RGB.
        # We must perform a color space conversion.
        # Equivalent to cv::cvtColor(img, img, cv::COLOR_BGR2RGB).
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Extract dimensions and calculate the memory stride (bytes per line).
        height, width, channel = cv_img_rgb.shape
        bytes_per_line = channel * width
        
        # Construct a QImage pointing to the NumPy array's memory block.
        # Note: QImage does NOT take ownership of the raw pointer it receives.
        # To avoid dangling pointers when `cv_img_rgb` is garbage collected at the end
        # of this scope, we immediately force a deep copy by calling `.copy()`.
        # Pyright complains about memoryview typing here, so we suppress it.
        q_img = QImage(cv_img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy() # type: ignore
        
        # Convert the CPU-bound QImage into a hardware-optimized QPixmap for rendering.
        self._pixmap = QPixmap.fromImage(q_img)
        
        # Update the QLabel's internal pixmap for rendering.
        self.setPixmap(self._pixmap)
        
        # Resize the QLabel widget to exactly match the loaded image dimensions.
        self.resize(self._pixmap.size())

    def save_image(self, file_path):
        """Serializes the current QPixmap buffer back to disk using OpenCV."""
        if not self._pixmap.isNull():
            # Convert the hardware-optimized QPixmap back into a CPU-accessible QImage.
            # We explicitly convert it to RGB888 format (24-bit, 3 channels) to simplify the NumPy mapping.
            q_img = self._pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
            
            width, height = q_img.width(), q_img.height()
            
            # Get a pointer to the underlying C++ pixel data array.
            ptr = q_img.bits()
            if ptr is None:
                return
            
            # In PyQt, the returned pointer doesn't inherently know its memory bounds.
            # We explicitly set the size in bytes to allow Python's buffer protocol to read it safely.
            ptr.setsize(height * width * 3) # type: ignore
            
            # Reinterpret the raw memory block as a NumPy array.
            # This is equivalent to wrapping a raw pointer in a cv::Mat in C++.
            img_rgb = np.frombuffer(ptr, np.uint8).reshape((height, width, 3)) # type: ignore
            
            # OpenCV expects BGR memory layout, so we convert back from RGB.
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Pass the NumPy array to OpenCV's C++ backend to write the file.
            cv2.imwrite(file_path, img_bgr)

    # ==============================================================================
    # Event Handlers (Overrides of QWidget virtual functions)
    # ==============================================================================

    def mousePressEvent(self, event):
        """Triggered when a mouse button is pressed inside this widget."""
        # Bail out early if there's no image loaded to prevent null dereference equivalents.
        if self._pixmap.isNull():
            return

        # Check if the left mouse button triggered the event.
        if self.mode == "draw" and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            # Cache the starting point. event.position() returns a floating-point QPointF,
            # so we cast it to an integer QPoint for pixel-perfect coordinates.
            self.last_point = event.position().toPoint()
            
        elif self.mode == "crop" and event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.position().toPoint()
            # Initialize the rubber band at the click position with zero size.
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event):
        """Triggered when the mouse moves while a button is held down."""
        if self._pixmap.isNull():
            return

        # event.buttons() returns a bitmask of ALL currently held buttons.
        if self.mode == "draw" and self.drawing and event.buttons() & Qt.MouseButton.LeftButton:
            # Initialize a QPainter bound to our QPixmap buffer.
            # In C++, you'd typically allocate this on the stack (`QPainter painter(&m_pixmap);`).
            # Here, the Python garbage collector manages it, but calling `painter.end()` ensures
            # the underlying C++ resources are released synchronously.
            painter = QPainter(self._pixmap)
            
            # Configure the drawing tool (Pen) with rounded caps/joins for smooth lines.
            pen = QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            
            # Draw a line from the last known point to the current mouse point.
            current_point = event.position().toPoint()
            painter.drawLine(self.last_point, current_point)
            
            # Update the last point for the next event frame.
            self.last_point = current_point
            
            # The QPixmap has been modified in memory, but the QLabel doesn't know yet.
            # Re-assign it to trigger a paint event (repaint).
            self.setPixmap(self._pixmap)
            
            # Explicitly release the drawing device context.
            painter.end()
            
        elif self.mode == "crop" and event.buttons() & Qt.MouseButton.LeftButton:
            # Update the selection rectangle dynamically.
            # QRect(origin, current).normalized() ensures the rectangle has positive width/height
            # regardless of which direction the user drags the mouse.
            self.rubber_band.setGeometry(QRect(self.origin, event.position().toPoint()).normalized())

    def mouseReleaseEvent(self, event):
        """Triggered when a mouse button is released."""
        if self._pixmap.isNull():
            return

        if self.mode == "draw" and event.button() == Qt.MouseButton.LeftButton:
            # End the drawing stroke.
            self.drawing = False
            
        elif self.mode == "crop" and event.button() == Qt.MouseButton.LeftButton:
            # Hide the selection UI element.
            self.rubber_band.hide()
            
            # Get the physical screen coordinates of the selection.
            selection = self.rubber_band.geometry()
            
            # Clamp the selection rectangle to the bounds of the actual image
            # to prevent out-of-bounds memory access during the copy operation.
            selection = selection.intersected(self._pixmap.rect())
            
            # Ensure the selection area is valid (non-zero).
            if selection.width() > 0 and selection.height() > 0:
                # Perform a deep copy of the selected pixel region.
                cropped = self._pixmap.copy(selection)
                
                # Replace our main backing buffer with the cropped version.
                self._pixmap = cropped
                
                # Update the display.
                self.setPixmap(self._pixmap)
                
                # Shrink the widget size to fit the new, smaller image.
                self.resize(self._pixmap.size())


# ==============================================================================
# Main Window Class
# ==============================================================================
# Inherits from QMainWindow, which provides a pre-defined layout suitable for 
# standard desktop applications (menu bar, toolbars, central widget, status bar).
class ImageEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Simple Image Editor")
        # Set initial window geometry: x, y, width, height.
        self.setGeometry(100, 100, 800, 600)

        # Instantiate our custom image viewing widget.
        # Passing `self` establishes the parent-child relationship for memory management.
        self.image_label = ImageLabel(self)
        
        # QScrollArea acts as a viewport. If `image_label` is larger than the window,
        # the scroll area automatically provides scrollbars.
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        
        # Disable resizability so the scroll area respects the explicit `resize()` calls
        # we make on the `image_label` when loading or cropping an image.
        self.scroll_area.setWidgetResizable(False) 
        
        # Assign the scroll area as the primary layout element of the QMainWindow.
        self.setCentralWidget(self.scroll_area)

        # Initialize the top toolbar.
        self.create_toolbar()

    def create_toolbar(self):
        """Constructs the UI toolbar and connects button signals to slots."""
        # In Qt, UI construction is typically done procedurally like this.
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Create the Load button.
        load_btn = QPushButton("Load")
        # Connect the `clicked` signal to the `load_image` slot (callback function).
        # This is PyQt's abstraction over C++ Qt's signal/slot mechanism or raw function pointers.
        load_btn.clicked.connect(self.load_image)
        toolbar.addWidget(load_btn)

        # Create the Save button.
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_image)
        toolbar.addWidget(save_btn)

        toolbar.addSeparator()

        # Create a toggleable (checkable) Draw button.
        self.draw_btn = QPushButton("Draw")
        self.draw_btn.setCheckable(True)
        self.draw_btn.clicked.connect(self.set_draw_mode)
        toolbar.addWidget(self.draw_btn)

        # Create a toggleable Crop button.
        self.crop_btn = QPushButton("Crop")
        self.crop_btn.setCheckable(True)
        self.crop_btn.clicked.connect(self.set_crop_mode)
        toolbar.addWidget(self.crop_btn)

        # Keep a list of mode buttons to implement pseudo-radio-button logic manually.
        self.buttons = [self.draw_btn, self.crop_btn]

    # ==============================================================================
    # Slots (Callback functions for UI signals)
    # ==============================================================================

    def set_draw_mode(self):
        self.update_buttons(self.draw_btn)
        # Check if the button was toggled ON or OFF.
        if self.draw_btn.isChecked():
            self.image_label.set_mode("draw")
        else:
            self.image_label.set_mode("view")

    def set_crop_mode(self):
        self.update_buttons(self.crop_btn)
        if self.crop_btn.isChecked():
            self.image_label.set_mode("crop")
        else:
            self.image_label.set_mode("view")

    def update_buttons(self, active_btn):
        """Helper to uncheck all mode buttons except the one just clicked."""
        for btn in self.buttons:
            if btn != active_btn:
                btn.setChecked(False)

    def load_image(self):
        """Opens a native file dialog to select an image."""
        # Static method call to get a file path. Returns a tuple (file_path, selected_filter).
        # We use `_` to discard the filter string, similar to `std::ignore` in C++.
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "JPEG Images (*.jpeg *.jpg);;All Files (*)"
        )
        if file_path:
            # Delegate the actual loading logic to the custom label.
            self.image_label.load_image(file_path)
            # Re-evaluate layout constraints based on the new image size.
            self.image_label.adjustSize()

    def save_image(self):
        """Opens a native file dialog to save the image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "JPEG Images (*.jpeg *.jpg)"
        )
        if file_path:
            self.image_label.save_image(file_path)

# ==============================================================================
# Application Entry Point
# ==============================================================================
# Equivalent to `int main(int argc, char *argv[])` in C++.
if __name__ == '__main__':
    # Initialize the core application state and GUI event loop.
    # Must be instantiated before any QWidgets are created.
    app = QApplication(sys.argv)
    
    # Create the main window instance. Memory is managed by Python's GC.
    window = ImageEditorWindow()
    
    # Widgets are hidden by default; show() schedules a paint event.
    window.show()
    
    # Enter the Qt event loop. `app.exec()` blocks until the main window is closed.
    # `sys.exit()` ensures the OS receives the application's exit code upon termination.
    sys.exit(app.exec())
