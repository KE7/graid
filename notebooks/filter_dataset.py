import json
from scenic_reasoning.utilities.common import project_root_dir
from itertools import islice
from pathlib import Path
import numpy as np
from PIL import Image
from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionUtils
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.models.Ultralytics import Yolo
from scenic_reasoning.utilities.common import (
    get_default_device,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
import cv2
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import re
from datetime import datetime
from datetime import time
import argparse


def run_bdd():
    bdd = Bdd100kDataset(
        split="train",
        transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(896, 1600)),
        rebuild=True,
        use_time_filtered=True,
    )


def is_time_in_working_hours(filename: str) -> bool:
    match = re.search(r"\d{4}-\d{2}-\d{2}-(\d{2})-(\d{2})-", filename)
    if not match:
        raise ValueError("Time not found in filename.")
    hour = int(match.group(1))
    minute = int(match.group(2))
    t = time(hour, minute)
    return time(8, 0) <= t < time(18, 0)


def run_nuimage():
    nu = NuImagesDataset(
        split="train",
        size="all",
        transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(896, 1600)),
        rebuild=True,
        use_time_filtered=True,
    )


def is_within_working_hours(timestamp_micro: str) -> bool:
    timestamp_sec = int(timestamp_micro) / 1e6
    dt = datetime.utcfromtimestamp(timestamp_sec)
    return time(8, 0) <= dt.time() < time(18, 0)


def run_waymo():
    import os
    import pickle

    source_dir = project_root_dir() / "data/waymo_training_interesting"
    dest_dir = project_root_dir() / "data/waymo_training_interesting_filtered"
    dest_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for file_name in tqdm(os.listdir(source_dir), desc="Processing files"):
        file_path = source_dir / file_name
        with open(file_path, "rb") as f:
            img_data = pickle.load(f)
        timestamp = img_data["timestamp"]
        if not is_within_working_hours(timestamp):
            print("skipping")
            continue
        with open(dest_dir / f"{idx}.pkl", "wb") as f:
            pickle.dump(img_data, f)
        idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset filtering tool.")
    parser.add_argument('--dataset', type=str, required=True, choices=['bdd', 'nuimage', 'waymo'],
                        help="Dataset to process: bdd, nuimage, or waymo")
    args = parser.parse_args()

    if args.dataset == 'bdd':
        run_bdd()
    elif args.dataset == 'nuimage':
        run_nuimage()
    elif args.dataset == 'waymo':
        run_waymo()