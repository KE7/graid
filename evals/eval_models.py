import json
from collections import defaultdict
from typing import Callable, List

import numpy as np
import ray
import torch
from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    ImageDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionModelI, ObjectDetectionUtils
from scenic_reasoning.models.Detectron import Detectron_obj
from scenic_reasoning.models.Ultralytics import RT_DETR, Yolo
from scenic_reasoning.utilities.common import (
    get_default_device,
    persistent_cache,
    project_root_dir,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

yolo_v10x = Yolo(model="yolov10x.pt")  # 61.4 Mb
yolo_11x = Yolo(model="yolo11x.pt")  # 109 Mb
rtdetr = RT_DETR("rtdetr-x.pt")  # 129 Mb

retinanet_R_101_FPN_3x_config = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"  # 228MB
retinanet_R_101_FPN_3x_weights = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
retinanet_R_101_FPN_3x = Detectron_obj(
    config_file=retinanet_R_101_FPN_3x_config,
    weights_file=retinanet_R_101_FPN_3x_weights,
)

faster_rcnn_R_50_FPN_3x_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"  # 167MB
faster_rcnn_R_50_FPN_3x_weights = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
faster_rcnn_R_50_FPN_3x = Detectron_obj(
    config_file=faster_rcnn_R_50_FPN_3x_config,
    weights_file=faster_rcnn_R_50_FPN_3x_weights,
)


@persistent_cache(str(project_root_dir() / "evals" / "eval_models_cache.pkl"))
@ray.remote(num_gpus=1)
def metric_per_dataset(
    model: ObjectDetectionModelI,
    dataset_fn: Callable[[], ImageDataset],
    conf_thresholds: List[float],
):
    dataset = dataset_fn()
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    model.to(device=get_default_device())

    mAP_at_conf = defaultdict(list)
    mAPs_with_penalty_at_conf = defaultdict(list)
    TN_count = defaultdict(int)
    TN_count_with_penalty = defaultdict(int)

    for batch in tqdm(data_loader, desc="processing " + str(dataset)):
        x = torch.stack([sample["image"] for sample in batch])
        y = [sample["labels"] for sample in batch]

        x = x.to(device=get_default_device())
        prediction = model.identify_for_image_batch(x)
        x = x.cpu()

        # TODO: this loop should be split into another ray task since it's cpu bound
        for idx, (odrs, gt) in enumerate(
            zip(prediction, y)
        ):  # odr = object detection result, gt = ground truth
            for conf in conf_thresholds:
                relevant_odrs = list(filter(lambda x: x.score >= conf, odrs))

                measurements: dict = (
                    ObjectDetectionUtils.compute_metrics_for_single_img(
                        relevant_odrs,
                        gt,
                        class_metrics=True,
                        extended_summary=True,
                        image=x[idx],
                        penalize_for_extra_predicitions=False,
                    )
                )
                full_image_result = dict()
                full_image_result["image"] = x[idx]
                full_image_result["measurements"] = measurements
                full_image_result["predictions"] = relevant_odrs
                full_image_result["labels"] = gt

                if measurements["TN"] == 1:
                    print("Both ground truth and predictions are empty. Ignore")
                    TN_count[conf] += 1
                else:
                    mAP = measurements["map"]
                    mAP_at_conf[conf].append(mAP.item())

                measurements_with_penalty: dict = (
                    ObjectDetectionUtils.compute_metrics_for_single_img(
                        relevant_odrs,
                        gt,
                        class_metrics=True,
                        extended_summary=True,
                        image=x[idx],
                        penalize_for_extra_predicitions=True,
                    )
                )
                full_result_with_penalty = dict()
                full_result_with_penalty["image"] = x[idx]
                full_result_with_penalty["measurements"] = measurements_with_penalty
                full_result_with_penalty["predictions"] = relevant_odrs
                full_result_with_penalty["labels"] = gt

                if measurements_with_penalty["TN"] == 1:
                    print("Both ground truth and predictions are empty. Ignore")
                    TN_count_with_penalty[conf] += 1
                else:
                    mAP = measurements_with_penalty["map"]
                    mAPs_with_penalty_at_conf[conf].append(mAP.item())

    mAPs = {
        conf: sum(mAP_at_conf[conf]) / len(mAP_at_conf[conf]) for conf in mAP_at_conf
    }
    mAPs_with_penalty = {
        conf: sum(mAPs_with_penalty_at_conf[conf])
        / len(mAPs_with_penalty_at_conf[conf])
        for conf in mAPs_with_penalty_at_conf
    }

    model.to(device="cpu")

    return {
        "dataset": str(dataset),
        "model": str(model),
        "confidence_thresholds": conf_thresholds,
        "average_mAP": mAPs,
        "average_mAP_with_penalty": mAPs_with_penalty,
        "TNs": TN_count,
        "TN_with_penaltys": TN_count_with_penalty,
    }


if __name__ == "__main__":
    ray.init()

    datasets = []
    # NuImages
    for split in ["mini", "train", "val"]:
        for size in ["all"]:
            datasets.append(
                lambda: NuImagesDataset(
                    split=split,
                    size=size,
                    transform=lambda i, l: yolo_nuscene_transform(
                        i, l, new_shape=(896, 1600)
                    ),
                )
            )
    # Waymo
    for split in ["training", "validation"]:
        datasets.append(
            lambda: WaymoDataset(
                split=split,
                transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920)),
            )
        )
    # BDD100k
    for split in ["train", "val"]:
        datasets.append(
            lambda: Bdd100kDataset(
                split=split,
                transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
                use_original_categories=False,
                use_extended_annotations=False,
            )
        )

    models = [
        yolo_v10x,
        yolo_11x,
        rtdetr,
        retinanet_R_101_FPN_3x,
        faster_rcnn_R_50_FPN_3x,
    ]
    confs = [float(c) for c in np.arange(0.05, 0.90, 0.05)]
    # confs = [0.2, 0.5, 0.7]

    for model in models:
        model.set_threshold(confs[0])

    BATCH_SIZE = 16

    tasks = []
    for d in datasets:
        for model in models:
            task_id = metric_per_dataset.remote(model, d, confs)
            tasks.append(task_id)

    results = ray.get(tasks)
    output_file = {}
    for result in results:
        k = result["model"] + "//" + result["dataset"]
        output_file[k] = result

    print(output_file)

    output_file_path = project_root_dir() / "eval" / "results.json"
    with open(output_file_path, "w") as f:
        json.dump(output_file, f, indent=4)
