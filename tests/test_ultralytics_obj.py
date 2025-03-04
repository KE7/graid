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

NUM_EXAMPLES_TO_SHOW = 20
BATCH_SIZE = 1

bdd = Bdd100kDataset(
    split="val",
    transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
    use_original_categories=False,
    use_extended_annotations=False,
)

nu = NuImagesDataset(split="mini", size="mini", transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(768, 1280)))

# waymo = WaymoDataset(split="validation", transform=lambda i, l: yolo_waymo_transform(i, l, (768, 1280)))

# https://docs.ultralytics.com/models/yolov5/#performance-metrics
model = Yolo(model="yolo11n.pt")
# model = Yolo(model="yolovv8n.pt")

for d in [nu]:  # , nu, waymo]:

    measurements = ObjectDetectionMeasurements(
        model, d, batch_size=BATCH_SIZE, collate_fn=lambda x: x
    )  # hacky way to avoid RuntimeError: each element in list of batch should be of equal size

    # WARNING ⚠️ imgsz=[720, 1280] must be multiple of max stride 64, updating to [768, 1280]
    for results in islice(
        measurements.iter_measurements(
            # device=get_default_device(),
            imgsz=[768, 1280],
            bbox_offset=24,
            debug=False,
            conf=0.17,
            class_metrics=True,
            extended_summary=True,
            agnostic_nms=True,
            # iou=0.7, # which is the default
        ),
        NUM_EXAMPLES_TO_SHOW,
    ):
        for i in range(len(results)):
            print("gt classes:", [c._class for c in results[i]["labels"]])
            print("pred classes:", [c._class for c in results[i]["predictions"]])

            print(f"{i}th image")
            measurements = results[i]["measurements"]
            print(measurements)
            print("global map", measurements['map'])
            print("map 50", measurements['map_50'])
            print("map 75", measurements['map_75'])
            print("map_small", measurements['map_small'])
            print("map_medium", measurements['map_medium'])
            print("map_large", measurements['map_large'])
            print("mar_small", measurements['mar_small'])
            print("mar_medium", measurements['mar_medium'])
            print("mar_large", measurements['mar_large'])

            ObjectDetectionUtils.show_image_with_detections_and_gt(
                Image.fromarray(results[i]["image"].permute(1, 2, 0).numpy().astype(np.uint8)),
                detections=results[i]["predictions"],
                ground_truth=results[i]["labels"],   
            )
