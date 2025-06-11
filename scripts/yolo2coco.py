import os
import json
from PIL import Image
from tqdm import tqdm

'''
Script to convert dataset from YOLO to COCO format
'''

def sorted_by_index(dir_path):
    return sorted(os.listdir(dir_path), key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

def yolo_to_coco(yolo_dir, img_dir, output_path):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "car"},
                       {"id": 1, "name": "truck"}]
    }

    ann_id = 0
    img_id = 0
    
    yolo_listdir = sorted_by_index(yolo_dir)

    for fname in tqdm(yolo_listdir):
        if not fname.endswith(".txt"):
            continue

        img_name = fname.replace(".txt", ".png")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue

        width, height = Image.open(img_path).size

        coco["images"].append({
            "file_name": img_name,
            "height": height,
            "width": width,
            "id": img_id
        })

        with open(os.path.join(yolo_dir, fname), "r") as f:
            for line in f.readlines():
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                x_min = x_center - w / 2
                y_min = y_center - h / 2

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(class_id),
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })

                ann_id += 1
        img_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f)

# Пример использования:
labels_dir = 'data/generated_dataset/yolo_format/labels'
images_dir = 'data/generated_dataset/yolo_format/images'
output_path = 'data/generated_dataset/coco_format/grounding_dino_gt.json'
yolo_to_coco(labels_dir, images_dir, output_path)