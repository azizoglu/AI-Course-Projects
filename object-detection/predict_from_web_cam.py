# Import necessary libraries
import cv2
from ultralytics import YOLO

# Load the trained YOLO model (.pt file)
# Replace "best.pt" with the path to your trained weights file
model = YOLO("best.pt")  # Path to your trained YOLOv11 model

# Start video capture from webcam (0 = default camera, 1 = external camera, etc.)
cap = cv2.VideoCapture(1)

# Start an infinite loop to process video frames
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # If the frame was not successfully read, exit the loop
    if not ret:
        break

    # Run the YOLO model on the current frame
    results = model(frame)

    # Draw the detection results (bounding boxes, labels, etc.) on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame in a window
    cv2.imshow("YOLO - Rock Paper Scissors", annotated_frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
