import cv2
import matplotlib.pyplot as plt
import numpy

img_path: str = "./knighP_1.png"

img_BGR = cv2.imread(img_path)

if img_BGR is None:
    exit()

img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

img_CMY = 255 - img_RGB

c, m, y = cv2.split(img_CMY)

plt.figure(figsize=(12, 4))

plt.subplot(2, 2, 1)
plt.imshow(c, cmap="Blues")
plt.title("Cyan")


plt.subplot(2, 2, 2)
plt.imshow(m, cmap="Purples")
plt.title("Magenta")

plt.subplot(2, 2, 3)
plt.imshow(y, cmap="YlOrBr")

plt.title("Yellow")
img_4x1_list = [
    [[0, 0, 255], [255, 255, 255]],
    [[0, 255, 0], [255, 255, 255]],
    [[255, 0, 0], [255, 255, 255]],
    [[0, 0, 0], [255, 255, 255]],
]


img_4x1 = numpy.array(img_4x1_list)

plt.subplot(2, 2, 4)
plt.imshow(img_4x1[:, :, ::-1])
plt.title("Image")

plt.show()

# https://gemini.google.com/share/2098b41a9bac
