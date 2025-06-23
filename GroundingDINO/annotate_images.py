from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
from tqdm import tqdm
import numpy as np
import torch
import argparse

'''
Annotates images using Grounding DINO, saves annotations win YOLO format
and saves images with overlays
'''

parser = argparse.ArgumentParser(description='Annotate images for object detection task using GroundingDINO.')
parser.add_argument('-id', '--images_dir', type=str, required=True,
                        help='Directory with images to annotate')
parser.add_argument('-ld', '--labels_dir', type=str, required=True,
                        help='Directory to save labels')
parser.add_argument('-p', '--prompt', type=str, required=False, default='car . truck',
                        help='Text prompt with objects to detect, objects can be separated with dot')
parser.add_argument('-bt', '--box_threshold', type=float, required=False, default=0.35,
                        help='Bbox confidence threshold')
parser.add_argument('-tt', '--text_threshold', type=float, required=False, default=0.35,
                        help='Text confidence threshold')


args = parser.parse_args()

# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swinb_cogcoor.pth")
model = load_model("/home/alexblokh/diffusion/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                   "/home/alexblokh/diffusion/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
IMAGES_PATH = args.images_dir
LABELS_PATH = args.labels_dir
TEXT_PROMPT = args.prompt
BOX_THRESHOLD = args.box_threshold
TEXT_THRESHOLD = args.text_threshold

if not os.path.exists(LABELS_PATH):
    os.mkdir(LABELS_PATH)

classes = TEXT_PROMPT.split(' . ')

cls_mapping = {}
for idx, label in enumerate(classes):
    cls_mapping[label] = idx

images_dir = os.listdir(IMAGES_PATH)

for filename in tqdm(images_dir):
    txt_name = f'{os.path.splitext(filename)[0]}.txt'

    image_source, image = load_image(os.path.join(IMAGES_PATH, filename))

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    filtered_boxes = []
    filtered_conf = []
    filtered_labels = []
    with open(os.path.join(LABELS_PATH, txt_name), 'w') as fout:
        for i in range(len(boxes)):
            box = boxes[i]
            conf = logits[i]
            if phrases[i] not in list(cls_mapping.keys()):
                continue
            label = cls_mapping[phrases[i]]
            cx, cy, w, h = box
            if w > 0.85 or h > 0.85:
                continue
            filtered_boxes.append(box.tolist())
            filtered_conf.append(conf.tolist())
            filtered_labels.append(label)
            fout.write(f"{label} {cx} {cy} {w} {h}\n")