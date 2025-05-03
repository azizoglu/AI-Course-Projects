# Import the YOLO class from the Ultralytics library
from ultralytics import YOLO

# Load a YOLOv8 model pretrained for instance segmentation tasks
# "yolo11m-seg.pt" is a custom or specific segmentation model
model = YOLO("yolo11m-seg.pt")

# If you want to use a detection model instead of segmentation, uncomment the line below
# "yolo11m.pt" would be the detection version of the model
# model = YOLO("yolo11m.pt")

# Train the loaded YOLO model using the specified parameters
# - data: path to dataset configuration file in YAML format
# - imgsz: input image size (e.g., 640x640 pixels)
# - device: which GPU to use (0 = first GPU)
# - batch: number of images per training batch
# - epochs: number of training iterations over the full dataset
# - workers: number of CPU threads for data loading
model.train(data="dataset.yaml", imgsz=640, device=0, batch=64, epochs=10, workers=4)
