import cv2
import matplotlib.pyplot as plt
import numpy as np

raw_image = cv2.imread("./knighP_1.png")


if raw_image is None:
    exit()

image = raw_image

b, g, r = cv2.split(image)

zeros = np.zeros_like(g)
image_no_green = cv2.merge([b, zeros, r])


plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original (RGB)")

plt.subplot(2, 2, 2)
plt.imshow(b, cmap="Blues")
plt.title("Blue Channel")


plt.subplot(2, 2, 3)
plt.imshow(r, cmap="Reds")
plt.title("Red Channel")


plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(image_no_green, cv2.COLOR_BGR2RGB))
plt.title("No Green Channel")

plt.tight_layout()
plt.show()
