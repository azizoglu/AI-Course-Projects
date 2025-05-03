# Import the YOLO class from the Ultralytics library
from ultralytics import YOLO

# Load a trained YOLO model from the file "best.pt"
# This is typically a model that has been fine-tuned on a custom dataset
model = YOLO("best.pt")

# Use the model to make predictions on the input image "test1.jpg"
# The parameters below customize how predictions are handled and displayed
model.predict(
    source="test1.jpg",      # Path to the image or video to be processed
    show=True,               # Display the image with predictions in a window
    save=True,               # Save the image with predictions drawn on it
    conf=0.7,                # Confidence threshold: only show predictions above 70% confidence
    line_width=2,            # Thickness of the bounding box lines
    save_crop=True,          # Save cropped images of detected objects
    save_txt=True,           # Save prediction results in a text file (YOLO format)
    show_labels=True,        # Display class labels on the bounding boxes
    show_conf=True,          # Display confidence scores on the bounding boxes
    classes=[0,1,2,3,4,5,6,7,8,9]  # Only detect and show objects belonging to these class IDs
)
