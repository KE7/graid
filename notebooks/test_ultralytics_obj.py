from itertools import islice
from scenic_reasoning.data.ImageLoader import Bdd100kDataset
from scenic_reasoning.models.UltralyticsYolo import Yolo
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.utilities.common import get_default_device
import torch
from ultralytics.data.augment import LetterBox

NUM_EXAMPLES_TO_SHOW = 1
BATCH_SIZE = 1

shape_transform = LetterBox(new_shape=(768, 1280))
def transform_image_for_yolo(image : torch.Tensor):
    # 1) convert from tensor to cv2 image
    image_np  = image.permute(1, 2, 0).numpy()
    # 2) resize to 768x1280
    image_np = shape_transform(image=image_np)
    # 3) convert back to tensor
    image = torch.tensor(image_np).permute(2, 0, 1)
    # 4) normalize to 0-1
    image = image.to(torch.float32) / 255.0

    return image

bdd = Bdd100kDataset(
    split="val", 
    # YOLO requires images to be 640x640 or 768x1280, 
    # but BDD100K images are 720x1280 so we need to resize
    transform=transform_image_for_yolo,  
    use_original_categories=False,
    use_extended_annotations=False,
)
# https://docs.ultralytics.com/models/yolov5/#performance-metrics
# model = Yolo(model="yolov5x6u.pt") # v5 can handle 1280 while v8 can handle 640. makes no sense ><
model = Yolo(model="yolov5x6u.pt")
measurements = ObjectDetectionMeasurements(model, bdd, batch_size=BATCH_SIZE, collate_fn=lambda x: x) # hacky way to avoid RuntimeError: each element in list of batch should be of equal size

# WARNING ⚠️ imgsz=[720, 1280] must be multiple of max stride 64, updating to [768, 1280]
from pprint import pprint
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
    pprint(results)
    [im.show() for im in ims]