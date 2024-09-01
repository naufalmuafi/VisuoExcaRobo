from ultralytics import YOLO


data_path = "datasets/RockWebotsDetector.v1i.yolov8/data.yaml"

model = YOLO("yolo_model/yolov8m.pt")
model.train(data=data_path, epochs=100, imgsz=640)
