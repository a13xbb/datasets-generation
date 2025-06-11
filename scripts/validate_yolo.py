from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

metrics = model.val(data='datasets/bdd100k/data.yaml', split='test', batch=16, device=0)