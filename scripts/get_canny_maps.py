import os
import cv2
import random
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np

''' 
Given a directory with images this script creates a directory
withCanny maps of those images
'''

def get_canny_image(image: Image.Image, low_threshold=100, high_threshold=200):
    image = image.resize((1024, 1024))
    image_np = np.array(image.convert("L"))
    edges = cv2.Canny(image_np, low_threshold, high_threshold)
    edges_3ch = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_3ch)

IMAGES_DIR = "data/seed_images/bdd_images"
CANNY_DIR = "data/seed_images/canny_maps"

for idx, filename in tqdm(enumerate(os.listdir(IMAGES_DIR))):
    input_image = Image.open(os.path.join(IMAGES_DIR, filename))
    control_image = get_canny_image(input_image)
    control_image.save(os.path.join(CANNY_DIR, f'canny_{idx}.jpg'))

