import torch
import cv2
import os
import subprocess
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BitMasks, pairwise_iou
from ultralytics import YOLO  # For YOLOv8 model inference
import matplotlib.pyplot as plt
import cv2
from data import ImageLoader
from interfaces import InstanceSegmentationI
from models import Detectron
from models.Detectron import Detectron2InstanceSegmentation
from scenic_reasoning.data.ImageLoader import Bdd10kDataset, NuImagesDataset
from torch.utils.data import DataLoader

# TODO: torch metrics and validate comparison methods
#       implement onto YOLO and other datasets

bdd_dataset = Bdd10kDataset(split="val")

dataloader = DataLoader(bdd_dataset, batch_size=1, shuffle=False)
detectron2_segmenter = Detectron2InstanceSegmentation()

def compare(segmenter, dataloader):
    results = []

    for batch in dataloader:
        image = batch["image"][0]  # First image in the batch
        ground_truth_labels = batch["labels"]  # Ground truth labels

        # Get predictions
        predictions = segmenter.predict(image)

        # Compare ground truth and predictions
        for gt in ground_truth_labels:
            gt_bbox = gt["bbox"]  # Assuming ground truth includes bounding boxes
            best_iou, best_pred = 0, None

            for pred in predictions:
                pred_bbox = pred["bbox"]
                iou = pairwise_iou(gt_bbox, pred_bbox)

                if iou > best_iou:
                    best_iou, best_pred = iou, pred

            results.append({
                "gt_label": gt["label"],
                "pred_label": best_pred["cls"] if best_pred else None,
                "iou": best_iou,
            })

    return results

# Perform the benchmarking on detectron2
results = compare(detectron2_segmenter, dataloader)

# Summarize metrics
mean_iou = sum(r["iou"] for r in results) / len(results)
print(f"Mean IoU: {mean_iou:.2f}")

for idx, result in enumerate(results):
    print(f"Sample {idx}:")
    print(f"  Ground Truth: {result['gt_label']}")
    print(f"  Prediction:  {result['pred_label']}")
    print(f"  IoU:         {result['iou']:.2f}")
