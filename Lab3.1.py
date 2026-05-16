import matplotlib.pyplot as plt
import cv2
import numpy as np

image = cv2.imread("home.jpg")
blue, green, red = cv2.split(image)

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(2, 2, 2)
plt.imshow(blue, cmap='Blues')
plt.title("Blue Channel")

# TODO: Plot the red and green channels

plt.show()

# TODO: Merge channels but remove the green channel
image_merged = cv2.merge([blue, np.zeros_like(), red])

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image_merged, cv2.COLOR_BGR2RGB))
plt.title("Image without Green Channel")
plt.show()