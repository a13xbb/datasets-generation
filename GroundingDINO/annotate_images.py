from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
from tqdm import tqdm
import numpy as np
import torch

'''
Annotates images using Grounding DINO, saves annotations win YOLO format
and saves images with overlays
'''

# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swinb_cogcoor.pth")
model = load_model("/home/alexblokh/diffusion/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                   "/home/alexblokh/diffusion/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
IMAGES_PATH = "/home/alexblokh/diffusion/generated_dataset/images"
LABELS_PATH = "/home/alexblokh/diffusion/generated_dataset/labels"
ANNOTATED_IMAGES_PATH = "/home/alexblokh/diffusion/generated_dataset/annotated_images"
TEXT_PROMPT = "car . truck"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.35

if not os.path.exists(LABELS_PATH):
    os.mkdir(LABELS_PATH)

if not os.path.exists(ANNOTATED_IMAGES_PATH):
    os.mkdir(ANNOTATED_IMAGES_PATH)

cls_mapping = {'car' : 0, 'truck': 1}

images_dir = os.listdir(IMAGES_PATH)

for filename in tqdm(images_dir):
    txt_name = f'{os.path.splitext(filename)[0]}.txt'

    image_source, image = load_image(os.path.join(IMAGES_PATH, filename))

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    filtered_boxes = []
    filtered_conf = []
    filtered_labels = []
    with open(os.path.join(LABELS_PATH, txt_name), 'w') as fout:
        for i in range(len(boxes)):
            box = boxes[i]
            conf = logits[i]
            if phrases[i] not in list(cls_mapping.keys()):
                if 'car' in phrases[i]:
                    phrases[i] = 'car'
                else:
                    continue
            label = cls_mapping[phrases[i]]
            cx, cy, w, h = box
            if w > 0.85 or h > 0.85:
                continue
            filtered_boxes.append(box.tolist())
            filtered_conf.append(conf.tolist())
            filtered_labels.append(label)
            fout.write(f"{label} {cx} {cy} {w} {h}\n")

    if len(filtered_boxes) == 0:
        cv2.imwrite(os.path.join(ANNOTATED_IMAGES_PATH, filename), image_source)
        continue

    filtered_boxes = torch.tensor(filtered_boxes)
    filtered_conf = torch.tensor(filtered_conf)

    annotated_frame = annotate(image_source=image_source, boxes=filtered_boxes,
                                logits=filtered_conf, phrases=filtered_labels)
    cv2.imwrite(os.path.join(ANNOTATED_IMAGES_PATH, filename), annotated_frame)

    # print(boxes)
    # print(filtered_boxes)
    # exit(0)