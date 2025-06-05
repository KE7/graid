from itertools import islice

import torch
from graid.data.ImageLoader import (
    Bdd10kDataset,
    NuImagesDataset_seg,
    WaymoDataset_seg,
)
from graid.measurements.InstanceSegmentation import (
    InstanceSegmentationMeasurements,
)
from graid.models.Ultralytics import Yolo_seg
from graid.utilities.common import (
    get_default_device,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
from ultralytics.data.augment import LetterBox

shape_transform = LetterBox(new_shape=(768, 1280))

NUM_EXAMPLES_TO_SHOW = 3
BATCH_SIZE = 1

bdd = Bdd10kDataset(
    split="val", transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280))
)

waymo = WaymoDataset_seg(
    split="validation", transform=lambda i, l: yolo_waymo_transform(i, l)
)

nuscene = NuImagesDataset_seg(
    split="test",
    transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(768, 1280)),
)

model = Yolo_seg(model="yolo11n-seg.pt")
for d in [bdd, nuscene, waymo]:
    # https://docs.ultralytics.com/models/yolov5/#performance-metrics
    measurements = InstanceSegmentationMeasurements(
        model, d, batch_size=BATCH_SIZE, collate_fn=lambda x: x
    )
    # WARNING ⚠️ imgsz=[720, 1280] must be multiple of max stride 64, updating to [768, 1280]
    for results, ims in islice(
        measurements.iter_measurements(
            device=get_default_device(),
            imgsz=[768, 1280],
            debug=True,
            conf=0.1,
            class_metrics=True,
            extended_summary=True,
        ),
        NUM_EXAMPLES_TO_SHOW,
    ):
        print("")
