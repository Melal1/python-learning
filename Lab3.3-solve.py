import cv2
import matplotlib.pyplot as plt

img_path: str = "./knighP_1.png"


img_BGR = cv2.imread(img_path)

if img_BGR is None:
    exit()


img_YCbCr = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.imshow(img_BGR[:, :, ::-1])
plt.title("RGB")

plt.subplot(1, 2, 2)
plt.imshow(img_YCbCr)
plt.title("YCbCr")

plt.show()
