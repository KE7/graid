import argparse

from scenic_reasoning.data.Datasets import ObjDectDatasetBuilder
from scenic_reasoning.models.Detectron import Detectron_obj
from scenic_reasoning.models.MMDetection import MMdetection_obj
from scenic_reasoning.models.Ultralytics import RT_DETR, Yolo

# from scenic_reasoning.models.MMDetection import MMdetection_obj
from scenic_reasoning.utilities.common import (
    project_root_dir,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
import torch

bdd_transform = lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280))
nuimage_transform = lambda i, l: yolo_nuscene_transform(i, l, new_shape=(896, 1600))
waymo_transform = lambda i, l: yolo_waymo_transform(i, l, (1280, 1920))

BATCH_SIZE = 1
GPU_ID = 7
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# rtdetr = RT_DETR("rtdetr-x.pt")
MMDETECTION_PATH = project_root_dir() / "install" / "mmdetection"
Co_DETR_config = str(
    MMDETECTION_PATH
    / "projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py"
)
Co_DETR_checkpoint = str(
    "https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth"
)
model = MMdetection_obj(Co_DETR_config, Co_DETR_checkpoint, device=device)


def generate_db(dataset_name, split, conf, model=None):

    if model:
        model.set_threshold(conf)
        db_name = f"{dataset_name}_{split}_{conf}_{str(model)}"
        model.to(device)
    else:
        db_name = f"{dataset_name}_{split}_gt"

    if dataset_name == "nuimage":
        transform = nuimage_transform
    elif dataset_name == "bdd":
        transform = bdd_transform
    elif dataset_name == "waymo":
        transform = waymo_transform
    else:
        print("no such dataset")
        return

    db_builder = ObjDectDatasetBuilder(
        split=split, dataset=dataset_name, db_name=db_name, transform=transform
    )
    if not db_builder.is_built():
        db_builder.build(model=model, batch_size=BATCH_SIZE, conf=conf, device=device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Distributed dataset generator with Ray."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["bdd", "nuimage", "waymo"],
        default="bdd",
        help="Select which dataset to use: 'bdd', 'nuimage', or 'waymo'.",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="Select which split to use: 'train' or 'val'.",
    )

    args = parser.parse_args()

    # https://github.com/ray-project/ray/issues/3899
    # ray.init(_temp_dir='/tmp/ray/graid')

    # model = rtdetr

    # models = [rtdetr]
    # models = [None]
    # confs = [c for c in np.arange(0.05, 0.90, 0.05)]
    confs = [0.2]

    datasets = [args.dataset]
    split = args.split

    tasks = []

    for d in datasets:
        for conf in confs:
            task_val = generate_db(d, split, conf, model=model)
            # task_train = generate_db(d, "train", conf, model=model)
            # generate_db(d, "val", conf, model=model)
            # generate_db(d, "train", conf, model=model)

# python scenic_reasoning/src/scenic_reasoning/data/generate_db.py --dataset bdd --split val; echo "bdd val 0.2 conf"