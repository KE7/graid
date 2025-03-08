import json

import ray
import torch
from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionUtils
from scenic_reasoning.models.Detectron import Detectron_obj
from scenic_reasoning.models.Ultralytics import RT_DETR, Yolo
from scenic_reasoning.utilities.common import (
    get_default_device,
    project_root_dir,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

yolo_v10x = Yolo(model="YOLOv10x.pt")
yolo_11x = Yolo(model="yolo11x.pt")
rtdetr = RT_DETR("rtdetr-x.pt")

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


@ray.remote(num_gpus=1)
def metric_per_dataset(model, dataset, conf_thresholds):
    dataset = dataset()
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    model.to(device=get_default_device())

    for batch in tqdm(data_loader, desc="processing dataset " + str(dataset)):
        x = torch.stack([sample["image"] for sample in batch])
        y = [sample["labels"] for sample in batch]

        x = x.to(device=get_default_device())
        if isinstance(model, Yolo) or isinstance(model, RT_DETR):
            # Convert RGB to BGR because Ultralytics models expect BGR
            # https://github.com/ultralytics/ultralytics/issues/9912
            x = x[:, [2, 1, 0], ...]
            x = x / 255.0
            prediction = model.identify_for_image(x)
            # undo the conversion
            x = x[:, [2, 1, 0], ...]
            x = x * 255.0
        elif isinstance(model, Detectron_obj):
            prediction = model.identify_for_image_as_tensor(x)
        else:
            prediction = model.identify_for_image(x)

        x = x.cpu()

        results = []
        mAPs = {}
        TN_counts = {}
        mAPs_with_penalty = {}
        TN_counts_with_penalty = {}

        for idx, (odrs, gt) in enumerate(
            zip(prediction, y)
        ):  # odr = object detection result, gt = ground truth

            mAP_at_conf = []
            mAPs_with_penalty_at_conf = []
            TN_count = 0
            TN_count_with_penalty = 0

            for conf in conf_thresholds:
                relevant_odrs = map(lambda x: x.score() >= conf, odrs)

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
                    TN_count += 1
                else:
                    mAP = measurements_with_penalty["map"]
                    mAP_at_conf.append(mAP.item())

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
                    TN_count_with_penalty += 1
                else:
                    mAP = measurements_with_penalty["map"]
                    mAPs_with_penalty_at_conf.append(mAP.item())

            mAPs[conf] = sum(mAP_at_conf) / len(mAP_at_conf)
            mAPs_with_penalty[conf] = sum(mAPs_with_penalty_at_conf) / len(
                mAPs_with_penalty_at_conf
            )
            TN_counts[conf] = TN_count
            TN_counts_with_penalty[conf] = TN_count_with_penalty

    model.to(device="cpu")

    return {
        "dataset": str(dataset),
        "model": str(model),
        "confidence_thresholds": conf_thresholds,
        "average_mAP": mAPs,
        "average_mAP_with_penalty": mAPs_with_penalty,
        "TNs": TN_counts,
        "TN_with_penaltys": TN_counts_with_penalty,
    }


if __name__ == "__main__":
    ray.init()

    datasets = []
    # NuImages
    for split in ["mini", "train", "val"]:
        for size in ["all", "mini"]:
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
    for split in ["train", "validation"]:
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
    # confs = [c for c in np.arange(0.05, 0.90, 0.05)]
    confs = [0.2, 0.5, 0.7]

    for model in models:
        model.set_threshold(confs[0])

    BATCH_SIZE = 8

    tasks = []
    for d in datasets:
        for model in models:
            task_id = metric_per_dataset.remote(model, d, confs)
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

    output_file_path = project_root_dir() / "eval" / "results.json"
    with open(output_file_path, "w") as f:
        json.dump(output_file, f, indent=4)
