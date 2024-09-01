from ultralytics import YOLO


# Directory containing the YAML file and the images
data_path = "datasets/RockWebotsDetector.v1i.yolov8/data.yaml"

# Create a YOLO scratch model
model = YOLO("yolo_model/yolov8m.pt")

# Train the model
# model.train(data=data_path, epochs=100, imgsz=640)

# Load the best model
model = YOLO("runs/detect/train_m_100/weights/best.pt")

# Predict the image
model("datasets/raw_data_collection/pc0_rock_00002_100.jpeg")
