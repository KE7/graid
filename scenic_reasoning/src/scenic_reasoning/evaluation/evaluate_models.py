from itertools import islice
from PIL import Image
import numpy as np
from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionUtils
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.models.UltralyticsYolo import Yolo
from scenic_reasoning.utilities.common import (
    get_default_device,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)


bdd = Bdd100kDataset(
    split="val",
    transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
    use_original_categories=False,
    use_extended_annotations=False,
)

nu = NuImagesDataset(split="mini", size="mini", transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(768, 1280)))

waymo = WaymoDataset(split="validation", transform=lambda i, l: yolo_waymo_transform(i, l, (768, 1280)))

yolo_v8n = Yolo(model="yolov8n.pt")
yolo_11n = Yolo(model="yolo11n.pt")


datasets = [bdd, nu, waymo]
models = [yolo_v8n, yolo_11n]
confs = [c for c in np.arange(0.05, 0.90, 0.05)]
BATCH_SIZE = 1

def metric_per_dataset(model, dataset, conf):
    measurements = ObjectDetectionMeasurements(
        model, dataset, batch_size=BATCH_SIZE, collate_fn=lambda x: x
    )
    mAPs = []
    for results in measurements.iter_measurements(
        imgsz=[768, 1280],
        bbox_offset=24,
        debug=False,
        conf=conf,
        class_metrics=True,
        extended_summary=True,
        agnostic_nms=True,
        ):
        for result in results:
            mAP = result["measurements"]['map']
            print("curr mAP:", mAP)
            mAPs.append(mAP.item())
    
    return sum(mAPs) / len(mAPs)


for d in datasets:
    for model in models:
        for conf in confs:
            average_mAP = metric_per_dataset(model, d, conf)
            print(f"Average mAP for {model} on {d} with conf {conf} is {average_mAP}")

