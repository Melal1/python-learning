# Vedio Processing

import cv2

# Open the video file
cap = cv2.VideoCapture('input_video.mp4')  # Replace with your video file path

# Read the frames
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
# TODO: Add each frame to frames list

# TODO: Reverse the frames


# Write the reversed frames to a new video
output_path = 'reverse.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter object
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Write each frame to the output video
for frame in frames:
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
