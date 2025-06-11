from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

'''
Script to count object detection metrics given GT and prediction json files
in COCO format.
'''

gt_path = 'data/generated_dataset/coco_format/grounding_dino_gt.json'
preds_path = 'data/generated_dataset/coco_format/rfdetr_preds.json'

coco_gt = COCO(gt_path)
coco_dt = coco_gt.loadRes(preds_path)

coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()