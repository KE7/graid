import json

import ray
from scenic_reasoning.data.Datasets import ObjDectDatasetBuilder
from scenic_reasoning.models.Detectron import Detectron_obj
from scenic_reasoning.models.Ultralytics import RT_DETR, Yolo
# from scenic_reasoning.models.MMDetection import MMdetection_obj
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

# retinanet_R_101_FPN_3x_config = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
# retinanet_R_101_FPN_3x_weights = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
# retinanet_R_101_FPN_3x = Detectron_obj(
#     config_file=retinanet_R_101_FPN_3x_config,
#     weights_file=retinanet_R_101_FPN_3x_weights,
# )

# faster_rcnn_R_50_FPN_3x_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# faster_rcnn_R_50_FPN_3x_weights = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# faster_rcnn_R_50_FPN_3x = Detectron_obj(
#     config_file=faster_rcnn_R_50_FPN_3x_config,
#     weights_file=faster_rcnn_R_50_FPN_3x_weights,
# )


# MMDETECTION_PATH = project_root_dir() / "install" / "mmdetection"

# Co_DETR_config = str(MMDETECTION_PATH / "projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py")
# Co_DETR_checkpoint = str(MMDETECTION_PATH / "checkpoints/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth")
# Co_DETR = MMdetection_obj(Co_DETR_config, Co_DETR_checkpoint)


BATCH_SIZE = 8


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

    db_builder = ObjDectDatasetBuilder(split=split, dataset=dataset_name, db_name=db_name, transform=transform)
    if not db_builder.is_built():
        db_builder.build(model=model, batch_size=BATCH_SIZE)
    

if __name__ == "__main__":
    # https://github.com/ray-project/ray/issues/3899
    ray.init(_temp_dir='/tmp/ray/graid')

    models = [rtdetr]
    # models = [None]
    # confs = [c for c in np.arange(0.05, 0.90, 0.05)]
    confs = [0.8]
    datasets = ["bdd", "waymo"]
    
    tasks = []

    for d in datasets:
        for model in models:
            for conf in confs:
                task_train = generate_db.remote(d, "train", conf, model=model)
                task_val = generate_db.remote(d, "val", conf, model=model)
                tasks.append(task_train)
                tasks.append(task_val)
        


    results = ray.get(tasks)
