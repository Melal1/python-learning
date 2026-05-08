import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QPixmap

app = QApplication(sys.argv)

cv_img = np.zeros((100, 100, 3), dtype=np.uint8)
cv_img[:] = (255, 0, 0) # Blue in BGR

cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
h, w, c = cv_img_rgb.shape
bpl = c * w

q_img = QImage(cv_img_rgb.data, w, h, bpl, QImage.Format.Format_RGB888).copy()
pixmap = QPixmap.fromImage(q_img)

print("Success")
