import os
import subprocess

import cv2
import detectron2
import matplotlib.pyplot as plt
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.structures import BitMasks, pairwise_iou
from detectron2.utils.visualizer import Visualizer
from interfaces import InstanceSegmentationI

# from InstanceSegmentationI import InstanceSegmentationResultI, Mask_Format
from models import Detectron
from models.Detectron import Detectron2InstanceSegmentation
from torch.utils.data import DataLoader
from ultralytics import YOLO  # For YOLOv8 model inference

from data import ImageLoader
from graid.data.ImageLoader import Bdd10kDataset, NuImagesDataset

# Detectron.py and UltralyticsYolo.py need to have InstanceSegmentationI interface integrated
# def run_segmentation_pipeline(dataset: Bdd100kDataset, detectron: Detectron, yolo: UltralyticsYolo):
#     """
#     Run instance segmentation pipeline on the BDD100K dataset.

#     Args:
#         dataset (Bdd100kDataset): Loaded dataset instance.
#         detectron : file with methods that does the grunt work
#         yolo : file with methods that does the grunt work, but not facebook
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     d2_results, yolo_results = [], []
#     for idx, sample in enumerate(dataset):
#         image = sample["image"]
#         labels = sample["labels"]

#         # Perform segmentation from both files
#         # todo: make predict method in both Detectron.py and UltralyticsYolo.py
#         d2_results.append(detectron.predict(image))
#         yolo_results.append(yolo.predict(image))
#     # i believe this then feeds into visualize
#     return d2_results, yolo_results


# # take a sample image from Bdd100k and NuImages
# bdd_dataset = Bdd10kDataset(split="val")
# # nuimages_dataset = NuImagesDataset(split="train")

# dataloader = DataLoader(bdd_dataset, batch_size=1, shuffle=False)
# # testing if i can even access the bdd images and labels
# for batch in dataloader:
#         image = batch["image"][0]  # First image in the batch
#         # ground_truth_labels = batch["labels"]  # Ground truth labels
#         # print(ground_truth_labels, image)
#         break


detectron2_segmenter = Detectron2InstanceSegmentation(
    config_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    weights_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
)
# detectron2_segmenter.visualize(image)
# run_segmentation_pipeline(dataset, detectron2_segmenter)
# yolov8_segmenter = # from UltralyticsYolo.py
# #run_segmentation_pipeline(dataset, yolov8_segmenter)

# testing for visualize method, would use code above for bdd10k but I'm not sure if you have the correct dataset for demoing
url = "https://media.istockphoto.com/id/1145543931/video/nyc-manhattan-viii-synched-series-front-view-driving-studio-process-plate.jpg?s=480x480&k=20&c=2JgH-GCWt4Tsr-f7gdUKp0d5fMnnPsd1d4teJ9A-muE="
filename = "input.jpg"
subprocess.run(["wget", url, "-O", filename])
im = cv2.imread("./input.jpg")
detectron2_segmenter.visualize(im)


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

            results.append(
                {
                    "gt_label": gt["label"],
                    "pred_label": best_pred["cls"] if best_pred else None,
                    "iou": best_iou,
                }
            )

    return results


# # Perform the benchmarking on detectron2
# results = compare(detectron2_segmenter, dataloader)

# # Summarize metrics
# mean_iou = sum(r["iou"] for r in results) / len(results)
# print(f"Mean IoU: {mean_iou:.2f}")

# for idx, result in enumerate(results):
#     print(f"Sample {idx}:")
#     print(f"  Ground Truth: {result['gt_label']}")
#     print(f"  Prediction:  {result['pred_label']}")
#     print(f"  IoU:         {result['iou']:.2f}")
