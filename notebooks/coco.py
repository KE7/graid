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
    project_root_dir
)
import torch
from torch.utils.data import DataLoader
import cv2
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pathlib import Path
import json
from tqdm import tqdm


NUM_EXAMPLES_TO_SHOW = 20
BATCH_SIZE = 1

bdd = Bdd100kDataset(
    split="val",
    transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
    use_original_categories=False,
    use_extended_annotations=False,
)


model = Yolo(model="yolo11n.pt")

data_loader = DataLoader(
                bdd,
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: x,
            )


images = []
annotations = []
predictions = []
categories = [
  {"id": 0, "name": "person"},
  {"id": 1, "name": "bicycle"},
  {"id": 2, "name": "car"},
  {"id": 3, "name": "motorcycle"},
  {"id": 5, "name": "bus"},
  {"id": 6, "name": "train"},
  {"id": 7, "name": "truck"},
  {"id": 9, "name": "traffic light"},
  {"id": 11, "name": "stop sign"}
]


ann_id_counter = 1
image_id_counter = 1

for batch in tqdm(data_loader):
    x = torch.stack([sample["image"] for sample in batch])
    y = [sample["labels"] for sample in batch]

    x = x.to(device=get_default_device())

    sizes = [sample["image"].shape[-2:] for sample in batch]
    file_names = [sample["path"] for sample in batch]
    # Convert RGB to BGR because Ultralytics YOLO expects BGR
    # https://github.com/ultralytics/ultralytics/issues/9912
    # x = x[:, [2, 1, 0], ...]
    # x = x / 255.0
    prediction = model.identify_for_image(x, debug=False)
    # undo the conversion
    # x = x[:, [2, 1, 0], ...]
    # x = x * 255.0

    for idx, (odrs, gt) in enumerate(
            zip(prediction, y)
        ):
        height, width = sizes[idx]
        file_name = file_names[idx]
        image_id = image_id_counter
        image_id_counter += 1
        sizes = [sample["image"].shape[-2:] for sample in batch]

        # Add image record
        images.append({
            "id": image_id,
            "width": float(width),
            "height": float(height),
            "file_name": file_name,
            "zip_file": ""
        })

        # Ground Truth Annotations
        for obj in gt:
            annotations.append({
                "id": ann_id_counter,
                "image_id": image_id,
                "category_id": int(obj.cls),
                "bbox": obj.as_xywh().tolist()[0],
                "iscrowd": 0,
                "area": 0.0
            })
            ann_id_counter += 1


        for pred in odrs:
            predictions.append({
                "image_id": image_id,
                "category_id": int(pred.cls),
                "bbox": pred.as_xywh().tolist()[0],
                "score": float(pred.score),
                # "area": 0.0
            })


coco_gt_dict = {
    "images": images,
    "annotations": annotations,
    "categories": categories
    }

output_path = project_root_dir() / "notebooks/coco_gt.json"
with open(output_path, "w") as f:
    json.dump(coco_gt_dict, f)

output_path_dt = project_root_dir() / "notebooks/coco_dt.json"
with open(output_path_dt, "w") as f:
    json.dump(predictions, f)


cocoGt = COCO("/home/eecs/liheng/scenic-reasoning/notebooks/coco_gt.json")
cocoDt = cocoGt.loadRes("/home/eecs/liheng/scenic-reasoning/notebooks/coco_dt.json")
leval = COCOeval(cocoGt, cocoDt, iouType="bbox")

leval.evaluate()
leval.accumulate()
leval.summarize()
