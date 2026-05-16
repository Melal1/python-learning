# Convert to CMY color space
import matplotlib.pyplot as plt
import cv2

# TODO: Read an RGB Image
image= None

# TODO: Get the image in CMY color system
cmy_image = None

c, m, y = cv2.split(cmy_image)

plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.imshow(image[:, :, ::-1])
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(c, cmap='Blues')
plt.title('Cyan Component')

plt.subplot(1, 4, 3)
plt.imshow(m, cmap='Purples')
plt.title('Magenta Component')

plt.subplot(1, 4, 4)
plt.imshow(y, cmap='YlOrBr')
plt.title('Yellow Component')

# TODO: show the plot