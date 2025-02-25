from itertools import islice

from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
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

for d in [nu]:

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
            conf=0.1,
            class_metrics=True,
            extended_summary=True,
        ),
        NUM_EXAMPLES_TO_SHOW,
    ):
        print("global map", results[0]['map'])
        print("map 50", results[0]['map_50'])
        print("map 75", results[0]['map_75'])
        print("map_small", results[0]['map_small'])
        print("map_medium", results[0]['map_medium'])
        print("map_large", results[0]['map_large'])
        print("mar_small", results[0]['mar_small'])
        print("mar_medium", results[0]['mar_medium'])
        print("mar_large", results[0]['mar_large'])
