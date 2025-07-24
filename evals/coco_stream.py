import argparse
import json
import os
import uuid

import ijson
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from graid.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from graid.utilities.common import (
    project_root_dir,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Constants
NUM_EXAMPLES_TO_SHOW = 20
BATCH_SIZE = 1
SEED = 7
torch.manual_seed(SEED)

args = argparse.ArgumentParser()
args.add_argument(
    "--dataset",
    "-d",
    type=str,
    choices=["bdd", "nuimage", "waymo"],
    default="bdd",
    help="Dataset to use: bdd, nuimage, or waymo",
)
args.add_argument(
    "--model",
    "-m",
    type=str,
    default="yolov8x-world",
    choices=[
        "DINO",
        "Co_DETR",
        "yolov10x",
        "yolov8x-world",
        "rtdetr",
        "retinanet_R_101_FPN_3x",
        "faster_rcnn_R_50_FPN_3x",
        "X101_FPN",
        "faster_rcnn_R_101_FPN_3x",
        "vitdet",
    ],
    help="Model to use",
)
args.add_argument(
    "--conf",
    "-c",
    type=float,
    default=0.5,
    help="Confidence threshold for predictions",
    choices=[x / 10 for x in range(0, 11)],
)
args.add_argument(
    "--device-id",
    "-d_id",
    type=int,
    default=7,
    help="Device ID for GPU (default: 0)",
    choices=[0, 1, 2, 3, 4, 5, 6, 7],
)

args = args.parse_args()

device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

dataset = args.dataset
if dataset == "bdd":
    # For vitdet, we don't apply a transform here because the model's
    # preprocessing is handled inside its class. For others, we might.
    transform = None
    if args.model != "vitdet":
        transform = lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280))

    dataset = Bdd100kDataset(
        split="val",  # Use validation set for evaluation
        transform=transform,
        use_original_categories=False,
        use_extended_annotations=False,
    )
elif dataset == "nuimage":
    dataset = NuImagesDataset(
        split="train",
        size="all",
        transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(896, 1600)),
    )
else:
    dataset = WaymoDataset(
        split="training",
        transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920)),
    )

data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    collate_fn=lambda x: x,
)


# Initialize the model
"""
Yolo(model="yolov8x-world.pt")"
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.094
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.161
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.093
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.143
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.090
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.142
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.144
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.144
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000"
"""
model = args.model
if "yolov6" in model:
    from graid.models.Ultralytics import Yolo

    model = Yolo(model=f"{model}.yaml")
elif "yolo" in model:
    from graid.models.Ultralytics import Yolo

    model = Yolo(model=f"{model}.pt")
elif model == "DINO":
    from graid.models.MMDetection import MMdetection_obj

    MMDETECTION_PATH = project_root_dir() / "install" / "mmdetection"
    DINO_config = str(
        MMDETECTION_PATH / "configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py"
    )
    DINO_checkpoint = str(
        "https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth"
    )
    model = MMdetection_obj(DINO_config, DINO_checkpoint)
    BATCH_SIZE = 1  # MMDetection does not support batch size > 1
elif model == "Co_DETR":
    from graid.models.MMDetection import MMdetection_obj

    MMDETECTION_PATH = project_root_dir() / "install" / "mmdetection"
    Co_DETR_config = str(
        MMDETECTION_PATH
        / "projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py"
    )
    Co_DETR_checkpoint = str(
        "https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth"
    )
    model = MMdetection_obj(Co_DETR_config, Co_DETR_checkpoint)
    BATCH_SIZE = 1  # MMDetection does not support batch size > 1
elif model == "retinanet_R_101_FPN_3x":
    from graid.models.Detectron import Detectron_obj

    retinanet_R_101_FPN_3x_config = (
        "COCO-Detection/retinanet_R_101_FPN_3x.yaml"  # 228MB
    )
    retinanet_R_101_FPN_3x_weights = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
    model = Detectron_obj(
        config_file=retinanet_R_101_FPN_3x_config,
        weights_file=retinanet_R_101_FPN_3x_weights,
    )
elif model == "faster_rcnn_R_50_FPN_3x":
    from graid.models.Detectron import Detectron_obj

    faster_rcnn_R_50_FPN_3x_config = (
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"  # 167MB
    )
    faster_rcnn_R_50_FPN_3x_weights = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    model = Detectron_obj(
        config_file=faster_rcnn_R_50_FPN_3x_config,
        weights_file=faster_rcnn_R_50_FPN_3x_weights,
    )
elif model == "rtdetr":
    from graid.models.Ultralytics import RT_DETR

    model = RT_DETR("rtdetr-x.pt")

elif model == "X101_FPN":
    from graid.models.Detectron import Detectron_obj

    X101_FPN_config = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"  # 167MB
    X101_FPN_weights = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    model = Detectron_obj(
        config_file=X101_FPN_config,
        weights_file=X101_FPN_weights,
    )
elif model == "faster_rcnn_R_101_FPN_3x":
    from graid.models.Detectron import Detectron_obj

    faster_rcnn_R_101_FPN_3x_config = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    faster_rcnn_R_101_FPN_3x_weights = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    model = Detectron_obj(
        config_file=faster_rcnn_R_101_FPN_3x_config,
        weights_file=faster_rcnn_R_101_FPN_3x_weights,
    )
elif model == "vitdet":
    from graid.models.Detectron import DetectronLazy

    CONFIG_FILE = "install/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_h_75ep.py"
    CHECKPOINT_FILE = "checkpoints/detectron2/model_final_f05665.pkl"
    model = DetectronLazy(
        config_file=CONFIG_FILE,
        weights_file=CHECKPOINT_FILE,
        threshold=args.conf,
        device=device,
    )

model.to(device)

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
    {"id": 11, "name": "stop sign"},
]

# Initialize counters for unique IDs
ann_id_counter = 1
image_id_counter = 1

proj_root = project_root_dir()
temp_id = uuid.uuid4().hex
gt_images_temp_path = proj_root / f"notebooks/{temp_id}_gt_images.tmp"
gt_annotations_temp_path = proj_root / f"notebooks/{temp_id}_gt_annotations.tmp"
coco_gt_path = proj_root / f"notebooks/{temp_id}_coco_gt.json"
coco_dt_path = proj_root / f"notebooks/{temp_id}_coco_dt.json"

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

total = min(2000, len(data_loader))  # Limit to 1000 batches for performance reasons

for i, batch in tqdm(enumerate(data_loader), total=total, desc="Processing batches"):
    if i >= total:
        break

    x = torch.stack([sample["image"] for sample in batch])
    y = [sample["labels"] for sample in batch]
    x = x.to(device=device)
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
            "zip_file": "",
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
                "area": 0.0,
            }
            if not first_annotation:
                annotations_file.write(",")
            else:
                first_annotation = False
            annotations_file.write(json.dumps(annotation_record))
            ann_id_counter += 1

        # Write predictions.
        for pred in odrs:
            if pred.score < args.conf:
                continue

            prediction_record = {
                "image_id": image_id,
                "category_id": int(pred.cls),
                "bbox": pred.as_xywh().tolist()[0],
                "score": float(pred.score),
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
    info_record = {
        "description": f"COCO-style dataset generated for {args.dataset} evaluation",
        "version": "1.0",
        "year": 2024,
    }
    f_out.write('{"info": ')
    f_out.write(json.dumps(info_record))
    f_out.write(', "images": ')

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

    f_out.write(', "annotations": ')
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

    f_out.write(', "categories": ')
    f_out.write(json.dumps(categories))
    f_out.write("}")

if os.path.exists(gt_images_temp_path):
    os.remove(str(gt_images_temp_path))
if os.path.exists(gt_annotations_temp_path):
    os.remove(str(gt_annotations_temp_path))

cocoGt = COCO(str(coco_gt_path))
cocoDt = cocoGt.loadRes(str(coco_dt_path))
leval = COCOeval(cocoGt, cocoDt, iouType="bbox")

leval.evaluate()
leval.accumulate()
leval.summarize()

if os.path.exists(coco_gt_path):
    os.remove(str(coco_gt_path))
if os.path.exists(coco_dt_path):
    os.remove(str(coco_dt_path))
