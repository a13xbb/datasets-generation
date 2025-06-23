import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, DDPMScheduler
from diffusers.utils import load_image
from transformers import AutoProcessor
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import random
import argparse

'''
Script to generate images using JuggernautXLv9 and Canny Controlnet,
GPU with 8-12GB VRAM is required
'''

parser = argparse.ArgumentParser(description='Generate images.')
parser.add_argument('-cd', '--canny_dir', type=str, required=True,
                        help='Directory containing Canny edge maps')
parser.add_argument('-sd', '--save_dir', type=str, required=True,
                        help='Directory to save generated images')
parser.add_argument('-n', '--n_images', type=int, required=True,
                        help='Amount of images to be generated')

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-XL-v9",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")


time_of_day = ["morning", "afternoon", "evening", "late night", "sunset", "sunrise"]
weather_normal = ["sunny", "cloudy", "foggy", "snowy", "rainy", "sunny and snowy", "rainy and foggy", "rainy and sunny"]
weather_night = ["clear", "cloudy", "foggy", "snowy", "rainy, rainy and snowy"]
location = ["urban", "suburban", "highway", "mountains", "coast", "countryside", "parking lot", "courtyard territory"]
traffic = ["light traffic", "moderate traffic", "traffic jam", "no traffic"]
cars_types = ["minivans", "trucks", "minivans and trucks"]
lighting = ["streetlights", "no street lights", "only light from cars headlights",
            "very dark, no light sources at all"]

canny_files = os.listdir(args.canny_dir)

output_dir_size = len(os.listdir(args.save_dir))

for i in tqdm(range(args.n_images)):  # или сколько нужно
    # Выбор случайных параметров
    loc_cond = random.choice(location)
    traffic_cond = random.choice(traffic)
    time_cond = random.choice(time_of_day)
    
    if time_cond in ["evening", "late night"]:
        weather_cond = random.choice(weather_night)
        lighting_cond = random.choice(lighting)
        prompt = f"A dashcam photo of a road, view from the driver's perspective,\
            camera near ground level, {time_cond}, {lighting_cond}, {weather_cond} weather,\
            {loc_cond} area, {traffic_cond}"
    else:
        weather_cond = random.choice(weather_normal)
        prompt = f"A dashcam photo of a road, view from the driver's perspective,\
            camera near ground level, {time_cond}, {weather_cond} weather, {loc_cond} area, {traffic_cond}"        
    
    gs = random.uniform(6, 10.0)
    cond_scale = random.uniform(0.15, 0.8)
    canny_path = os.path.join(args.canny_dir, random.choice(canny_files))
    canny_image = Image.open(canny_path).resize((1280, 720))

    negative = "unrealistic, blurry, distorted, cartoon, art, bright lighting"

    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=1280,
        height=720,
        image=canny_image,
        guidance_scale=gs,
        num_inference_steps=40,
        controlnet_conditioning_scale=cond_scale
    ).images[0]

    image.save(f"{args.save_dir}/img_{i + output_dir_size}.png")