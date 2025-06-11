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

'''
Script to generate images using JuggernautXLv9 and Canny Controlnet
'''

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


# time_of_day = ["morning", "afternoon", "evening", "dark evening", "sunset", "sunrise"]
time_of_day = ["late night"]
# weather = ["sunny", "cloudy", "foggy", "snowy", "rainy", "sunny and snowy", "rainy and foggy", "rainy and sunny"]
weather = ["clear", "cloudy", "foggy", "snowy", "rainy"]
location = ["urban", "suburban", "highway", "mountains", "coast", "countryside", "parking lot", "courtyard territory"]
traffic = ["light traffic", "moderate traffic", "traffic jam", "no traffic"]
cars_types = ["minivans", "trucks", "minivans and trucks"]
# lighting = ["streetlights", "medium lighting", "no street lights, only light from cars headlights", "very dark, only some light from car headlights",
#             "very dark, no streetlights, no light from other cars"]
lighting = ["very dark, no light around at all, shadows on vehicles"]

output_dir = "/home/alexblokh/diffusion/images_with_trucks"
os.makedirs(output_dir, exist_ok=True)

canny_dir = "/home/alexblokh/diffusion/data/seed_images/canny_maps"
canny_files = os.listdir(canny_dir)

# output_dir_size = len(os.listdir(output_dir))
output_dir_size = 12076

for i in tqdm(range(5000)):  # или сколько нужно
    # Выбор случайных параметров
    weather_cond = random.choice(weather)
    loc_cond = random.choice(location)
    traffic_cond = random.choice(traffic)
    time_cond = random.choice(time_of_day)
    lighting_cond = random.choice(lighting)
    cars_types_cond = random.choice(cars_types)
    gs = random.uniform(6, 10.0)
    cond_scale = random.uniform(0.15, 0.8)
    canny_path = os.path.join(canny_dir, random.choice(canny_files))
    canny_image = Image.open(canny_path).resize((1280, 720))

    # Сборка промпта
    # prompt = f"A dashcam photo of a road, view from the driver's perspective, camera near ground level, {time_cond}, {lighting_cond}, {weather_cond} weather,  {loc_cond} area, {traffic_cond}, {cars_types_cond} ahead"
    prompt = f"A dashcam photo of a road, view from the driver's perspective, camera near ground level, {time_cond}, {lighting_cond}, {loc_cond} area, {traffic_cond}, {cars_types_cond} ahead"
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

    image.save(f"{output_dir}/img_{i + output_dir_size}.png")