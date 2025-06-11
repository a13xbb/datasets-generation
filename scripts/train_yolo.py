from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(
    data='data/generated_dataset/yolo_format/data.yaml',
    epochs=60,
    imgsz=640,
    batch=32,
    device=[0, 1, 2, 3]
)