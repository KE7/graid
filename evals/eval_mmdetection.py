import json

import numpy as np
import ray
from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionUtils
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.models.Detectron import Detectron_obj
from scenic_reasoning.models.MMDetection import MMdetection_obj
from scenic_reasoning.models.Ultralytics import RT_DETR, Yolo
from scenic_reasoning.utilities.common import (
    get_default_device,
    project_root_dir,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
from tqdm import tqdm

GDINO_config = "../install/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg.py"
GDINO_checkpoint = "../install/mmdetection/checkpoints/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
GDINO = MMdetection_obj(GDINO_config, GDINO_checkpoint)

Co_DETR_config = "../install/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py"
Co_DETR_checkpoint = "../install/mmdetection/checkpoints/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth"
Co_DETR = MMdetection_obj(Co_DETR_config, Co_DETR_checkpoint)


@ray.remote(num_gpus=1)
def metric_per_dataset(model, dataset_name, conf):
    model.set_threshold(conf)

    if dataset_name == "NuImages":
        dataset = NuImagesDataset(
            split="mini",
            size="all",
            transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(896, 1600)),
        )
    elif dataset_name == "Waymo":
        dataset = WaymoDataset(
            split="validation",
            transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920)),
        )
    else:
        dataset = Bdd100kDataset(
            split="val",
            transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
            use_original_categories=False,
            use_extended_annotations=False,
        )

    measurements = ObjectDetectionMeasurements(
        model, dataset, batch_size=BATCH_SIZE, collate_fn=lambda x: x
    )
    mAPs = []
    TN_count = 0
    mAP_iterator = measurements.iter_measurements(
        bbox_offset=24,
        debug=False,
        conf=conf,
        class_metrics=True,
        extended_summary=True,
        agnostic_nms=True,
        fake_boxes=False,
    )
    for results in tqdm(mAP_iterator, desc="processing mAP measurements"):
        for result in results:
            if result["measurements"]["TN"] == 1:
                print("Both ground truth and predictions are empty. Ignore")
                TN_count += 1
                continue
            mAP = result["measurements"]["map"]
            mAPs.append(mAP.item())

    mAPs_fake = []
    TN_count_fake = 0
    mAP_iterator_fake = measurements.iter_measurements(
        bbox_offset=24,
        debug=False,
        conf=conf,
        class_metrics=True,
        extended_summary=True,
        agnostic_nms=True,
        fake_boxes=True,
    )
    for results in tqdm(mAP_iterator_fake, desc="processing fake mAP measurements"):
        for result in results:
            if result["measurements"]["TN"] == 1:
                print("Both ground truth and predictions are empty. Ignore")
                TN_count_fake += 1
                continue
            mAP = result["measurements"]["map"]
            mAPs_fake.append(mAP.item())

    return {
        "dataset": dataset_name,
        "model": str(model),
        "confidence": conf,
        "average_mAP": sum(mAPs) / len(mAPs),
        "fake_average_mAP": sum(mAPs_fake) / len(mAPs_fake),
        "TN": TN_count,
        "TN_fake": TN_count_fake,
    }


if __name__ == "__main__":
    ray.init()

    datasets = ["NuImages"]
    models = [GDINO, Co_DETR]
    # confs = [c for c in np.arange(0.05, 0.90, 0.05)]
    confs = [0.2, 0.5, 0.7]
    BATCH_SIZE = 8

    tasks = []
    for d in datasets:
        for model in models:
            for conf in confs:
                task_id = metric_per_dataset.remote(model, d, conf)
                tasks.append(task_id)

    results = ray.get(tasks)
    output_file = {}
    for result in results:
        k = (
            result["model"]
            + "//"
            + result["dataset"]
            + "//"
            + str(result["confidence"])
        )
        output_file[k] = result

    print(output_file)

    output_file_path = (
        project_root_dir()
        / "scenic_reasoning"
        / "src"
        / "scenic_reasoning"
        / "eval"
        / "results.json"
    )
    with open(output_file_path, "w") as f:
        json.dump(output_file, f, indent=4)
