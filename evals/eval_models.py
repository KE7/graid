import json

import ray
from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.models.Detectron import Detectron_obj
from scenic_reasoning.models.Ultralytics import RT_DETR, Yolo
from scenic_reasoning.utilities.common import (
    project_root_dir,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
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

    measurements = ObjectDetectionMeasurements(
        model, dataset, batch_size=BATCH_SIZE, collate_fn=lambda x: x
    )
    mAPs = []
    TN_count = 0
    mAP_iterator = measurements.iter_measurements(
        bbox_offset=24, # TODO: this should be calibrated per dataset and image size
        debug=False,
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
        class_metrics=True,
        extended_summary=True,
        agnostic_nms=True,
        fake_boxes=True,
    )
    for results in tqdm(mAP_iterator_fake, desc="processing COCO mAP with penalties for extra predictions"):
        for result in results:
            if result["measurements"]["TN"] == 1:
                print("Both ground truth and predictions are empty. Ignore")
                TN_count_fake += 1
                continue
            mAP = result["measurements"]["map"]
            mAPs_fake.append(mAP.item())

    return {
        "dataset": str(dataset),
        "model": str(model),
        "confidence": conf,
        "average_mAP": sum(mAPs) / len(mAPs),
        "fake_average_mAP": sum(mAPs_fake) / len(mAPs_fake),
        "TN": TN_count,
        "TN_fake": TN_count_fake,
    }


if __name__ == "__main__":
    ray.init()

    datasets = []
    # NuImages
    for split in ["mini", "train", "val"]:
        for size in ["all", "mini"]:
            datasets.append(
                lambda: 
                NuImagesDataset(
                    split=split,
                    size=size,
                    transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(896, 1600)),
                )
            )
    # Waymo
    for split in ["train", "validation"]:
        datasets.append(
            lambda:
            WaymoDataset(
                split=split,
                transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920)),
            )
        )
    # BDD100k
    for split in ["train", "val"]:
        datasets.append(
            lambda:
            Bdd100kDataset(
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
