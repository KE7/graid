import torch
import cv2
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BitMasks
from ultralytics import YOLO  # For YOLOv8 model inference

from InstanceSegmentationI import InstanceSegmentationResultI, Mask_Format
from ImageLoader import ImageDataset, Bdd100kDataset


# Detectron.py and UltralyticsYolo.py need to have InstanceSegmentationI interface integrated 
def run_segmentation_pipeline(dataset: Bdd100kDataset, detectron: Detectron, yolo: UltralyticsYolo):
    """
    Run instance segmentation pipeline on the BDD100K dataset.

    Args:
        dataset (Bdd100kDataset): Loaded dataset instance.
        detectron : file with methods that does the grunt work 
        yolo : file with methods that does the grunt work, but not facebook
    """
    os.makedirs(save_dir, exist_ok=True)
    d2_results, yolo_results = [], []
    for idx, sample in enumerate(dataset):
        image = sample["image"]
        labels = sample["labels"]

        # Perform segmentation from both files
        # todo: make predict method in both Detectron.py and UltralyticsYolo.py
        d2_results.append(detectron.predict(image))
        yolo_results.append(yolo.predict(image))
    # i believe this then feeds into visualize
    return d2_results, yolo_results




def visualize(image: torch.Tensor, results: List[InstanceSegmentationResultI], save_path: str):
    #todo
    #need to fish the Bdd100kdataset from ImageLoader
    #this should display bdd image on left (with all the instances labeled) and model prediction image on the right (YOLO or detectron2)
    #ask karim how to integrate, bc confused about how ImageLoader works

detectron2_config = "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
detectron2_weights = "/path/to/detectron2/weights.pth"

yolov8_weights = "/path/to/yolov8/weights.pt"


dataset = Bdd100kDataset(split="val")


detectron2_segmenter = # from Detectron.py
run_segmentation_pipeline(dataset, detectron2_segmenter)

yolov8_segmenter = # from UltralyticsYolo.py
run_segmentation_pipeline(dataset, yolov8_segmenter)