import cv2
import numpy as np
import os

# Background subtractor for detecting motion
fgbg = cv2.createBackgroundSubtractorMOG2()

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Define a function to process frames
def process_frame(frame):
    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Highlight moving objects with bounding boxes
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Ignore small contours
            continue
        x, y, w, h = cv2.boundingRect(contour)
        color = (255, 0, 0)  # Blue for moving objects
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame

# Process and display the video
output_path = "webcam_output.mp4"  # Path to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)
    out.write(frame)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output saved to {output_path}")
