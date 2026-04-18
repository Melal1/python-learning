# Image Processing
import cv2
import matplotlib.pyplot as plt
import os


A = cv2.imread('home.jpg')

# Define quality factors
# TODO: Define list of quality factors
quality_factors=None
CR = []
original_size = A.size

# Compress the image at different quality factors
for factor in quality_factors:
    compressed_name = f'compressedImage{factor}.jpg'
    # TODO: Write the image with different factor each time

    # Get the size of the compressed image
    compressed_size = os.path.getsize(compressed_name)
    # Calculate compression ratio
    cr = original_size / compressed_size
    CR.append(cr)

# Plot compression ratio vs. quality factor
plt.plot(quality_factors, CR, 'b-o')
plt.xlabel('Quality Factor')
plt.ylabel('CR Value')
plt.show()
