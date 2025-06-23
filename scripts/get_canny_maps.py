import os
import cv2
import random
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse

''' 
Given a directory with images this script creates a directory
with Canny maps of those images
'''

def get_canny_image(image: Image.Image, low_threshold=100, high_threshold=200):
    image = image.resize((1024, 1024))
    image_np = np.array(image.convert("L"))
    edges = cv2.Canny(image_np, low_threshold, high_threshold)
    edges_3ch = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_3ch)

parser = argparse.ArgumentParser(description='Generate Canny edge maps from images.')
parser.add_argument('-id', '--images_dir', type=str, required=True,
                        help='Directory containing input images')
parser.add_argument('-sd', '--save_dir', type=str, required=True,
                        help='Directory to save Canny edge maps')

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

for idx, filename in tqdm(enumerate(os.listdir(args.images_dir))):
    input_image = Image.open(os.path.join(args.images_dir, filename))
    control_image = get_canny_image(input_image)
    control_image.save(os.path.join(args.save_dir, f'canny_{idx}.jpg'))

