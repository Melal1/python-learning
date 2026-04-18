# Convert to YCbCr color space
import cv2
import matplotlib.pyplot as plt

# TODO: Read an RGB Image
image = None
ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)

# TODO: Show the orginal image

plt.title("Original Image")

plt.subplot(1, 2, 2)

# TODO: Show the ycbcr image

plt.title("YCbCr Image")
plt.show()

