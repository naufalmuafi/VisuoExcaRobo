from ultralytics import YOLO


data_path = "../../datasets/RockWebotsDetector.v1i.yolov8/data.yaml"

model = YOLO("yolov5s.pt")
model.train(data=data_path, epochs=30, imgsz=640)
