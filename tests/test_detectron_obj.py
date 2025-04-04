from itertools import islice
from scenic_reasoning.data.ImageLoader import Bdd100kDataset
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionUtils
from scenic_reasoning.utilities.common import get_default_device
from scenic_reasoning.models.Detectron import Detectron_obj
from ultralytics.data.augment import LetterBox
import numpy as np
from PIL import Image

NUM_EXAMPLES_TO_SHOW = 3
BATCH_SIZE = 1

shape_transform = LetterBox(new_shape=(768, 1280))
bdd = Bdd100kDataset(
    split="val",
    use_original_categories=False,
    use_extended_annotations=False,
)


threshold = 0.5
config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
weights_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


model = Detectron_obj(
    config_file=config_file, weights_file=weights_file, threshold=threshold
)

measurements = ObjectDetectionMeasurements(
    model, bdd, batch_size=BATCH_SIZE, collate_fn=lambda x: x
)  # hacky way to avoid RuntimeError: each element in list of batch should be of equal size

# WARNING ⚠️ imgsz=[720, 1280] must be multiple of max stride 64, updating to [768, 1280]
from pprint import pprint

for results in islice(
    measurements.iter_measurements(
        device=get_default_device(),
        imgsz=[720, 1280],
        bbox_offset=24,
        debug=True,
        conf=0.1,
        class_metrics=True,
        extended_summary=True,
    ),
    NUM_EXAMPLES_TO_SHOW,
):
    for i in range(len(results)):
        print("gt classes:", [c._class for c in results[i]["labels"]])
        print("pred classes:", [c._class for c in results[i]["predictions"]])

        print(f"{i}th image")
        measurements = results[i]["measurements"]
        print(measurements)
        print("global map", measurements["map"])
        print("map 50", measurements["map_50"])
        print("map 75", measurements["map_75"])
        print("map_small", measurements["map_small"])
        print("map_medium", measurements["map_medium"])
        print("map_large", measurements["map_large"])
        print("mar_small", measurements["mar_small"])
        print("mar_medium", measurements["mar_medium"])
        print("mar_large", measurements["mar_large"])

        ObjectDetectionUtils.show_image_with_detections_and_gt(
            Image.fromarray(
                results[i]["image"].permute(1, 2, 0).numpy().astype(np.uint8)
            ),
            detections=results[i]["predictions"],
            ground_truth=results[i]["labels"],
        )
