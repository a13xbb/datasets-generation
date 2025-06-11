import io
import supervision as sv
from PIL import Image
from rfdetr import RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES
from tqdm import tqdm
import numpy as np
import cv2
import os
import json

'''
Get RF-DETR predictions in COCO format
'''

def sorted_by_index(dir_path):
    return sorted(os.listdir(dir_path), key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

IMAGES_DIR = 'data/generated_dataset/yolo_format/images'

classes_of_interest = {'car': 3, 'truck': 8}

model = RFDETRLarge()

coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "car"},
                       {"id": 1, "name": "truck"}]
    }

ann_id = 0
img_id = 0

images_listdir = sorted_by_index(IMAGES_DIR)

for fname in tqdm(images_listdir):
    img_path = os.path.join(IMAGES_DIR, fname)
    image = Image.open(img_path)
    width, height = image.size

    coco["images"].append({
        "file_name": fname,
        "height": height,
        "width": width,
        "id": img_id
    })
    
    detections = model.predict(image, threshold=0.4)

    boxes = detections.xyxy
    conf = detections.confidence
    class_ids = detections.class_id

    boxes = boxes[np.isin(class_ids, list(classes_of_interest.values()))]
    conf = conf[np.isin(class_ids, list(classes_of_interest.values()))]
    class_ids = class_ids[np.isin(class_ids, list(classes_of_interest.values()))]
    
    for i in range(len(boxes)):
        cur_box, cur_conf, cur_class = boxes[i], conf[i], class_ids[i]
        if cur_class == 3:
            cur_class = 0
        elif cur_class == 8:
            cur_class = 1
        else:
            print('Error: unknown class')
            raise Exception
        
        xmin, ymin, xmax, ymax = cur_box
        w = xmax - xmin
        h = ymax - ymin
        
        coco["annotations"].append({
                    "image_id": img_id,
                    "category_id": int(cur_class),
                    "bbox": list(map(float, [xmin, ymin, w, h])),
                    "score": float(cur_conf)
                })
        ann_id += 1
        
    img_id += 1
    
output_path = 'data/generated_dataset/coco_format/rfdetr_preds.json'

with open(output_path, 'w') as fout:
    json.dump(coco['annotations'], fout)
