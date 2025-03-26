import ijson
import json
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.models.Ultralytics import Yolo
from scenic_reasoning.utilities.common import (
    get_default_device,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
    project_root_dir
)
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# Constants
NUM_EXAMPLES_TO_SHOW = 20
BATCH_SIZE = 1

# Setup dataset with appropriate transform
bdd = Bdd100kDataset(
    split="val",
    transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
    use_original_categories=False,
    use_extended_annotations=False,
)

# Initialize the model
model = Yolo(model="yolo11n.pt")

# Create a DataLoader
data_loader = DataLoader(
    bdd,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: x,
)

# Define COCO category information
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

# Initialize counters for unique IDs
ann_id_counter = 1
image_id_counter = 1

proj_root = project_root_dir()
gt_images_temp_path = proj_root / "notebooks/gt_images.tmp"
gt_annotations_temp_path = proj_root / "notebooks/gt_annotations.tmp"
coco_gt_path = proj_root / "notebooks/coco_gt.json"
coco_dt_path = proj_root / "notebooks/coco_dt.json"

# Open temporary files and write opening brackets so that each is a valid JSON array.
images_file = open(gt_images_temp_path, "w")
annotations_file = open(gt_annotations_temp_path, "w")
pred_file = open(coco_dt_path, "w")
images_file.write("[")
annotations_file.write("[")
pred_file.write("[")

first_image = True
first_annotation = True
first_pred = True

for i, batch in tqdm(enumerate(data_loader)):
    x = torch.stack([sample["image"] for sample in batch])
    y = [sample["labels"] for sample in batch]
    x = x.to(device=get_default_device())
    sizes = [sample["image"].shape[-2:] for sample in batch]
    file_names = [sample["path"] for sample in batch]

    # Get predictions from the model.
    prediction = model.identify_for_image_batch(x, debug=False)

    for idx, (odrs, gt) in enumerate(zip(prediction, y)):
        height, width = sizes[idx]
        file_name = file_names[idx]
        image_id = image_id_counter
        image_id_counter += 1

        # Create image record and write to the temporary images file.
        image_record = {
            "id": image_id,
            "width": float(width),
            "height": float(height),
            "file_name": file_name,
            "zip_file": ""
        }
        if not first_image:
            images_file.write(",")
        else:
            first_image = False
        images_file.write(json.dumps(image_record))

        # Write ground truth annotations.
        for obj in gt:
            annotation_record = {
                "id": ann_id_counter,
                "image_id": image_id,
                "category_id": int(obj.cls),
                "bbox": obj.as_xywh().tolist()[0],
                "iscrowd": 0,
                "area": 0.0
            }
            if not first_annotation:
                annotations_file.write(",")
            else:
                first_annotation = False
            annotations_file.write(json.dumps(annotation_record))
            ann_id_counter += 1

        # Write predictions.
        for pred in odrs:
            prediction_record = {
                "image_id": image_id,
                "category_id": int(pred.cls),
                "bbox": pred.as_xywh().tolist()[0],
                "score": float(pred.score)
            }
            if not first_pred:
                pred_file.write(",")
            else:
                first_pred = False
            pred_file.write(json.dumps(prediction_record))

# Close the JSON arrays in the temporary files.
images_file.write("]")
annotations_file.write("]")
pred_file.write("]")
images_file.close()
annotations_file.close()
pred_file.close()

# Now, we build the final COCO ground-truth file by streaming through the temporary files with ijson.
with open(coco_gt_path, "w") as f_out:
    f_out.write("{\"images\": ")
    # Stream images from the temporary images file.
    with open(gt_images_temp_path, "r") as f_images:
        f_out.write("[")
        first = True
        for item in ijson.items(f_images, "item", use_float=True):
            if not first:
                f_out.write(",")
            else:
                first = False
            f_out.write(json.dumps(item))
        f_out.write("]")
    
    f_out.write(", \"annotations\": ")
    # Stream annotations from the temporary annotations file.
    with open(gt_annotations_temp_path, "r") as f_annotations:
        f_out.write("[")
        first = True
        for item in ijson.items(f_annotations, "item", use_float=True):
            if not first:
                f_out.write(",")
            else:
                first = False
            f_out.write(json.dumps(item))
        f_out.write("]")
    
    f_out.write(", \"categories\": ")
    f_out.write(json.dumps(categories))
    f_out.write("}")

gt_images_temp_path.unlink()
gt_annotations_temp_path.unlink()

cocoGt = COCO(str(coco_gt_path))
cocoDt = cocoGt.loadRes(str(coco_dt_path))
leval = COCOeval(cocoGt, cocoDt, iouType="bbox")

leval.evaluate()
leval.accumulate()
leval.summarize()
