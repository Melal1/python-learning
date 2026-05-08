import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage
import numpy as np

app = QApplication(sys.argv)
img = QImage(10, 10, QImage.Format.Format_RGB888)
img.fill(0)
ptr = img.bits()
print(type(ptr))
try:
    ptr.setsize(10 * 10 * 3)
    arr = np.frombuffer(ptr, np.uint8).reshape((10, 10, 3))
    print("setsize successful", arr.shape)
except Exception as e:
    print("Error:", e)
    arr = np.array(ptr).reshape((10, 10, 3))
    print("np.array successful", arr.shape)
