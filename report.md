# Flood Fill & Image Editor Report

## Implementation Overview
The application is a Desktop Graphical User Interface (GUI) built with **PyQt6** and **OpenCV**. It provides an interactive platform for coloring images using a custom-built Flood-Fill algorithm, along with advanced image editing features like cropping and grayscale gradient preservation.

## Core Mechanisms

### 1. Image Processing (OpenCV)
All image manipulation is performed using the OpenCV library. Images are loaded as NumPy arrays in BGR format. For display within the PyQt6 GUI, these arrays are converted to RGB format and then into `QImage` and `QPixmap` objects.

### 2. Custom Flood-Fill Algorithm
To comply with the requirement of not using `cv2.floodFill`, the algorithm was implemented from scratch using a **Breadth-First Search (BFS)** approach:
- **Starting Point**: Triggered by a mouse click on the image.
- **Queue-based Traversal**: A `collections.deque` is used to manage pixels to be processed.
- **Connectivity**: The algorithm checks 4-way connectivity (Up, Down, Left, Right).
- **Visited Matrix**: A NumPy boolean array tracks visited pixels to prevent infinite loops and redundant processing.

### 3. Advanced Features

#### A. Grayscale Gradient Preservation
When "Preserve Gradients" is enabled, the algorithm handles shading by:
1. **Tolerance Matching**: Instead of exact color matching, it uses Euclidean distance in BGR space to identify pixels within a specific tolerance range (e.g., gradients of the same color).
2. **Luminance Modulation**: It calculates the luminance of the starting pixel and each subsequent pixel in the region. The target color is then multiplied by the ratio of the current pixel's luminance to the starting pixel's luminance. This ensures that highlights and shadows from the original image are reflected in the new color.

#### B. Crop Tool
The cropping functionality uses `QRubberBand` to allow users to draw a selection rectangle over the image. Once the mouse is released, the selected region is extracted using NumPy slicing (`image[y1:y2, x1:x2]`) and the display is updated.

#### C. Undo System
The application maintains a history stack of previous image states. Before any destructive operation (filling or cropping), the current state is copied and pushed onto the stack. The "Undo" action simply pops the last state and restores it.

## Technical Constraints
- **Library**: Only OpenCV is used for image operations.
- **Exclusion**: `cv2.floodFill` is strictly avoided in favor of the manual BFS implementation.
- **GUI**: Transitioned from Tkinter to PyQt6 for a more modern and robust interface.
