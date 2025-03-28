import argparse

from scenic_reasoning.data.Datasets import ObjDectDatasetBuilder
from scenic_reasoning.models.Detectron import Detectron_obj
from scenic_reasoning.models.Ultralytics import RT_DETR, Yolo

# from scenic_reasoning.models.MMDetection import MMdetection_obj
from scenic_reasoning.utilities.common import (
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
import ray

bdd_transform = lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280))
nuimage_transform = lambda i, l: yolo_nuscene_transform(i, l, new_shape=(896, 1600))
waymo_transform = lambda i, l: yolo_waymo_transform(i, l, (1280, 1920))


rtdetr = RT_DETR("rtdetr-l.pt")

BATCH_SIZE = 64


@ray.remote(num_gpus=1)
def generate_db(dataset_name, split, conf, model=None):

    if model:
        model.set_threshold(conf)
        db_name = f"{dataset_name}_{split}_{str(model)}"
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
        db_builder.build(model=model, batch_size=BATCH_SIZE)



if __name__ == "__main__":

    ray.init()

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

    model = rtdetr

    # models = [rtdetr]
    # models = [None]
    # confs = [c for c in np.arange(0.05, 0.90, 0.05)]
    confs = [0.2]

    datasets = [args.dataset]
    split = args.split


    tasks = []

    for d in datasets:
        for conf in confs:
            task_val = generate_db.remote(d, split, conf, model=model)
            # task_train = generate_db(d, "train", conf, model=model)
            # generate_db(d, "val", conf, model=model)
            # generate_db(d, "train", conf, model=model)

            tasks.append(task_val)
            # tasks.append(task_train)

    ray.get(tasks)