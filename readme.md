# Two-stage datasets generation pipeline
Pipeline generates images and then annotates them using JuggernautXL + ControlNet for generation ad GroundingDINO for annotation for detection task.<br>
You can run this pipeline to create road scenes dataset for detection task, if you want to specify some generation details you will need to change the prompt and it's random conditions in generation script. Other options like objects to annotate and annotation model thresholds can be specified using scripts parameters.<br>
Generative model requires GPU with minimum 8GB VRAM to work normally. <br>
All experiments including visualization in ipynb notebook were done using BDD100k dataset as a real data.

## 1. Get Canny edge maps from seed images for generation
To get Canny maps for generation prepare directory with some seed images (around 1-2% of desired dataset size) for example from BDD100k dataset, then use the script:
```bash
python scripts/get_canny_maps.py --images_dir [path_to_images] --save_dir [path_to_save_directory]
```
## 2. Generate images with JuggernautXL + ControlNet
To generate images run generation script:
```bash
python scripts/generate_images.py --canny_dir [path_to_canny_maps] --save_dir [path_to_save_directory] --n_images [amount of desired images]
```
If you want to generate some custom images for another domain, prompt and random conditions for it need to be modified in the script code.
## 3. Annotate images
1) Clone GroundingDINO github repository into "GroundingDINO" folder following all instruction from their repo: https://github.com/IDEA-Research/GroundingDINO
2) Run annotation script using following command:
```bash
python GroundingDINO/annotate_images.py \
  --images_dir [path_to_images] \
  --labels_dir [path_to_labels] \
  --prompt "car . truck" \
  --box_threshold 0.35 \
  --text_threshold 0.35
  ```
--prompt is optional, default is "car . truck", objects should be separated with " . " like in example<br>
--box_threshold is optional, default is 0.35<br>
--text_threshold is optional, default is 0.35<br>

Annotations will be saved in yolo format.

## Additional scripts for validation
For validation of generation and annotation quality you can use following scripts: <br>
1) rfdetr_predict.py - use to make predictions with RF-DETR on generated dataset in COCO format.
2) yolo2coco.py - use to convert GroundingDINO annotation from YOLO to COCO ground truth format.
3) check_annotation_quality.py - use to get detection metrics for RF-DETR on generated dataset.
4) variety_visualization.ipynb - notebook to calculate some generation quality metrics, can be modified just by changing paths to the directories.
5) train_yolo.py and validate_yolo.py - use to train and validate YOLOv8 model on generated and your real data.
To use these scripts you can modify all the paths in the code.

