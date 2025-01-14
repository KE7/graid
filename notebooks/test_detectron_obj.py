from itertools import islice
from scenic_reasoning.data.ImageLoader import Bdd100kDataset
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.utilities.common import get_default_device
from scenic_reasoning.models.Detectron import Detectron_obj
from ultralytics.data.augment import LetterBox

NUM_EXAMPLES_TO_SHOW = 1
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
    config_file=config_file, 
    weights_file=weights_file, 
    threshold=threshold
)

measurements = ObjectDetectionMeasurements(model, bdd, batch_size=BATCH_SIZE, collate_fn=lambda x: x) # hacky way to avoid RuntimeError: each element in list of batch should be of equal size

# WARNING ⚠️ imgsz=[720, 1280] must be multiple of max stride 64, updating to [768, 1280]
from pprint import pprint
for (results, ims) in islice(measurements.iter_measurements(
        device=get_default_device(), 
        imgsz=[720, 1280],
        bbox_offset=24,
        debug=True,
        conf=0.1,
        class_metrics=True,
        extended_summary=True,
        ), 
    NUM_EXAMPLES_TO_SHOW):
    pprint(results)
    [im.show() for im in ims]

