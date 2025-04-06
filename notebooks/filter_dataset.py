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



# # BDD
# # Path to the JSON file
# json_file_path = project_root_dir() / 'data/bdd100k/labels/det_20/det_train.json'

# # Load the JSON file
# with open(json_file_path, 'r') as file:
#     data = json.load(file)

# # import pdb
# # pdb.set_trace()

# weather_set = set(['partly cloudy', 'clear'])
# timeofday_set = set(['daytime'])
# print(weather_set)
# print(timeofday_set)
# print('total:', len(data))
# count = 0
# empty_count = 0
# new_data = []
# for d in data:
#     if 'labels' not in d or len(d['labels']) == 0:
#         empty_count += 1
#     if d['attributes']['weather'] not in weather_set and d['attributes']['timeofday'] not in timeofday_set:
#         continue
#     count += 1
#     new_data.append(d)

# print('filtered', count)
# print('empty', empty_count)
# # Save the filtered data to a new JSON file
# output_file_path = project_root_dir() / 'data/bdd100k/labels/det_20/det_train_filtered.json'
# with open(output_file_path, 'w') as output_file:
#     json.dump(new_data, output_file, indent=4)

# print(f'Filtered data saved to {output_file_path}')



# # Nuimage

# BATCH_SIZE = 1

# nu = NuImagesDataset(
#     split="val",
#     size="all",
#     transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(896, 1600)),
#     rebuild=False
# )

# data_loader = DataLoader(
#     nu,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=2,
#     collate_fn=lambda x: x,
# )



# def is_time_in_working_hours(filename: str) -> bool:
#     match = re.search(r"\d{4}-\d{2}-\d{2}-(\d{2})-(\d{2})-", filename)
#     if not match:
#         raise ValueError("Time not found in filename.")
    
#     hour = int(match.group(1))
#     minute = int(match.group(2))
#     t = time(hour, minute)

#     return time(8, 0) <= t < time(18, 0)

# print(is_time_in_working_hours("n013-2018-09-13-12-17-19+0800__CAM_FRONT_LEFT__1536812298604825.jpg"))

# # exit()


# name_set = set()
# desc_set = set()
# count = total_count = 0

# for idx, batch in enumerate(tqdm(data_loader, desc="Loading Batches")):
#     if total_count == 100: 
#         break
#     for b in batch:
#         total_count += 1    
#         if not is_time_in_working_hours(b['name']):
#             count += 1
#         # for att in b['attributes']:
#         #     if not att:
#         #         continue
#         #     name_set.add(att[0]['name'])
#         #     desc_set.add(att[0]['description'])

# print(count, total_count)



# waymo = WaymoDataset(split="training", transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920)))

# data_loader = DataLoader(
#     waymo,
#     batch_size=1,
#     shuffle=False,
#     num_workers=2,
#     collate_fn=lambda x: x,
# )


# def is_within_working_hours(timestamp_micro: str) -> bool:
#     # Convert string to int and microseconds to seconds
#     timestamp_sec = int(timestamp_micro) / 1e6
#     dt = datetime.utcfromtimestamp(timestamp_sec)  # Assuming UTC

#     # Check if time is within 08:00 to 18:00
#     return time(8, 0) <= dt.time() < time(18, 0)


# total_count = count = 0

# for idx, batch in enumerate(tqdm(data_loader, desc="Loading Batches")):
#     for b in batch:
#         total_count += 1    
#         t = b['timestamp']
#         if not is_within_working_hours(t):
#             count += 1

# print(count, total_count)



def run_bdd():
    json_file_path = project_root_dir() / 'data/bdd100k/labels/det_20/det_train.json'
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    weather_set = set(['partly cloudy', 'clear'])
    timeofday_set = set(['daytime'])

    count = 0
    empty_count = 0
    new_data = []

    for d in tqdm(data):
        if 'labels' not in d or len(d['labels']) == 0:
            empty_count += 1
        if d['attributes']['weather'] not in weather_set and d['attributes']['timeofday'] not in timeofday_set:
            continue
        count += 1
        new_data.append(d)

    output_file_path = project_root_dir() / 'data/bdd100k/labels/det_20/det_train_filtered.json'
    with open(output_file_path, 'w') as output_file:
        json.dump(new_data, output_file, indent=4)

    print(f'Filtered {count} entries (Empty: {empty_count})')
    print(f'Filtered data saved to {output_file_path}')


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
        split="val",
        size="all",
        transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(896, 1600)),
        rebuild=False
    )

    data_loader = DataLoader(
        nu,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: x,
    )

    count = total_count = 0
    for idx, batch in enumerate(tqdm(data_loader, desc="Loading NuImages Batches")):
        for b in batch:
            total_count += 1
            import pdb
            pdb.set_trace()
            
            if not is_time_in_working_hours(b['name']):
                count += 1

    print(f"Filtered {count} out of {total_count} images outside working hours.")


def is_within_working_hours(timestamp_micro: str) -> bool:
    timestamp_sec = int(timestamp_micro) / 1e6
    dt = datetime.utcfromtimestamp(timestamp_sec)
    return time(8, 0) <= dt.time() < time(18, 0)


def run_waymo():
    waymo = WaymoDataset(
        split="training",
        transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920))
    )

    data_loader = DataLoader(
        waymo,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: x,
    )

    total_count = count = 0
    for idx, batch in enumerate(tqdm(data_loader, desc="Loading Waymo Batches")):
        for b in batch:
            total_count += 1
            if not is_within_working_hours(b['timestamp']):
                count += 1

    print(f"Filtered {count} out of {total_count} frames outside working hours.")


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