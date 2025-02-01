from scenic_reasoning.data.ImageLoader import Bdd10kDataset, NuImagesDataset_seg, WaymoDataset_seg
from scenic_reasoning.models.UltralyticsYolo import Yolo_seg
from scenic_reasoning.measurements.InstanceSegmentation import InstanceSegmentationMeasurements

from scenic_reasoning.utilities.common import get_default_device
import torch
from itertools import islice
from ultralytics.data.augment import LetterBox

shape_transform = LetterBox(new_shape=(768, 1280))

NUM_EXAMPLES_TO_SHOW = 3
BATCH_SIZE = 1

bdd = Bdd10kDataset(
    split="val", 
    # YOLO requires images to be 640x640 or 768x1280, 
    # but BDD100K images are 720x1280 so we need to resize
    # transform=transform_image_for_yolo,  
    use_original_categories=False,
    use_extended_annotations=False,
)

waymo = WaymoDataset_seg(split="validation")

model = Yolo_seg(model="yolo11n-seg.pt")

nuscene = NuImagesDataset_seg(
    split="test"
)


for d in [bdd, waymo, nuscene]:

    # https://docs.ultralytics.com/models/yolov5/#performance-metrics
    model = Yolo_seg(model="yolo11n-seg.pt") # v5 can handle 1280 while v8 can handle 640. makes no sense ><
    # measurements = InstanceSegmentationMeasurements(model, bdd, batch_size=BATCH_SIZE, collate_fn=lambda x: x) # hacky way to avoid RuntimeError: each element in list of batch should be of equal size
    measurements = InstanceSegmentationMeasurements(model, d, batch_size=BATCH_SIZE, collate_fn=lambda x: x)
    model.identify_for_image(['../demo/demo.jpg', '../demo/demo2.jpg'])
    # model.identify_for_image('../demo/demo.jpg')
    # WARNING ⚠️ imgsz=[720, 1280] must be multiple of max stride 64, updating to [768, 1280]
    from pprint import pprint
    for (results, ims) in islice(measurements.iter_measurements(
            device=get_default_device(), 
            imgsz=[768, 1280],
            debug=True,  
            conf=0.1,
            class_metrics=True,
            extended_summary=True,
            ), 
        NUM_EXAMPLES_TO_SHOW):
        print("")
        # pprint(results)
        # [im.show() for im in ims]  #TODO: need to write a custom function to display the ground truth instance segmentation result.
