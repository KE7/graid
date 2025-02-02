from itertools import islice
from scenic_reasoning.data.ImageLoader import Bdd100kDataset, NuImagesDataset, WaymoDataset
from scenic_reasoning.models.UltralyticsYolo import Yolo
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.utilities.common import get_default_device, yolo_transform
import torch
from ultralytics.data.augment import LetterBox

NUM_EXAMPLES_TO_SHOW = 3
BATCH_SIZE = 1

bdd = Bdd100kDataset(
    split="val", 
    transform=yolo_transform,  
    use_original_categories=False,
    use_extended_annotations=False,
)

niu = NuImagesDataset(split='test', transform=yolo_transform)

waymo = WaymoDataset(split="validation", transform=yolo_transform)

# https://docs.ultralytics.com/models/yolov5/#performance-metrics
model = Yolo(model="yolo11n.pt")

for d in [bdd, niu, waymo]:
        
    measurements = ObjectDetectionMeasurements(model, d, batch_size=BATCH_SIZE, collate_fn=lambda x: x) # hacky way to avoid RuntimeError: each element in list of batch should be of equal size

    # WARNING ⚠️ imgsz=[720, 1280] must be multiple of max stride 64, updating to [768, 1280]
    for (results, ims) in islice(measurements.iter_measurements(
            # device=get_default_device(), 
            imgsz=[768, 1280],
            bbox_offset=24,
            debug=True,
            conf=0.1,
            class_metrics=True,
            extended_summary=True,
            ), 
        NUM_EXAMPLES_TO_SHOW):
        print("")