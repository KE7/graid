import json

import ray
from scenic_reasoning.data.Datasets import ObjDectDatasetBuilder
from scenic_reasoning.models.Detectron import Detectron_obj
from scenic_reasoning.models.Ultralytics import RT_DETR, Yolo
from scenic_reasoning.utilities.common import (
    get_default_device,
    project_root_dir,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
from tqdm import tqdm

bdd_transform = lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280))
nuimage_transform = lambda i, l: yolo_nuscene_transform(i, l, new_shape=(896, 1600))
waymo_transform = lambda i, l: yolo_waymo_transform(i, l, (1280, 1920))


yolo_v8n = Yolo(model="yolov8n.pt")
yolo_11n = Yolo(model="yolo11n.pt")
rtdetr = RT_DETR("rtdetr-l.pt")

retinanet_R_101_FPN_3x_config = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
retinanet_R_101_FPN_3x_weights = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
retinanet_R_101_FPN_3x = Detectron_obj(
    config_file=retinanet_R_101_FPN_3x_config,
    weights_file=retinanet_R_101_FPN_3x_weights,
)

faster_rcnn_R_50_FPN_3x_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
faster_rcnn_R_50_FPN_3x_weights = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
faster_rcnn_R_50_FPN_3x = Detectron_obj(
    config_file=faster_rcnn_R_50_FPN_3x_config,
    weights_file=faster_rcnn_R_50_FPN_3x_weights,
)

BATCH_SIZE = 8


@ray.remote(num_gpus=1)
def generate_db(model, dataset_name, split, conf):

    model.set_threshold(conf)
    db_name = f"{dataset_name}_{split}_{str(model)}"

    if dataset_name == "nuimage":
        transform = nuimage_transform
    elif dataset_name == "bdd":
        transform = bdd_transform
    elif dataset_name == "waymo":
        transform = waymo_transform

    db_builder = ObjDectDatasetBuilder(split=split, dataset=dataset_name, db_name=db_name, transform=transform)
    if not db_builder.is_built():
        db_builder.build(model=model, batch_size=BATCH_SIZE)
    

if __name__ == "__main__":
    ray.init()

    models = [yolo_11n]
    # confs = [c for c in np.arange(0.05, 0.90, 0.05)]
    confs = [0.2]
    datasets = ["bdd"]
    
    tasks = []

    for d in datasets:
        for model in models:
            for conf in confs:
                task_train = generate_db.remote(model, d, "val", conf)
                # task_val = generate_db.remote(model, d, "val", conf)
                tasks.append(task_train)
                # tasks.append(task_val)
                break
            break
        break


    results = ray.get(tasks)
