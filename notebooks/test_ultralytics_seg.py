from scenic_reasoning.data.ImageLoader import Bdd100kDataset
from scenic_reasoning.models.UltralyticsYolo import Yolo_seg
from scenic_reasoning.measurements.InstanceSegmentation import InstanceSegmentationMeasurements

from scenic_reasoning.utilities.common import get_default_device
import torch
from itertools import islice
from ultralytics.data.augment import LetterBox

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

NUM_EXAMPLES_TO_SHOW = 1
BATCH_SIZE = 1

bdd = Bdd100kDataset(
    split="val", 
    # YOLO requires images to be 640x640 or 768x1280, 
    # but BDD100K images are 720x1280 so we need to resize
    transform=transform_image_for_yolo,  
    use_original_categories=False,
    use_extended_annotations=False,
)


# https://docs.ultralytics.com/models/yolov5/#performance-metrics
model = Yolo_seg(model="yolo11n-seg.pt") # v5 can handle 1280 while v8 can handle 640. makes no sense ><
measurements = InstanceSegmentationMeasurements(model, bdd, batch_size=BATCH_SIZE, collate_fn=lambda x: x) # hacky way to avoid RuntimeError: each element in list of batch should be of equal size
model.identify_for_image(['../demo/demo.jpg', '../demo/demo2.jpg'])
# result = model._model.predict(source=['../demo/demo.jpg', '../demo/demo2.jpg'])[0]
# print(len(result))
# print(result.masks.data.shape)
# first_mask = result.masks.data[0].numpy()
# import matplotlib.pyplot as plt
# import numpy as np
# plt.figure(figsize=(8, 6))
# plt.title("First Mask")
# plt.imshow(first_mask, cmap="gray")  # Use grayscale for binary masks
# plt.axis("off")  # Remove axes for better visualization
# plt.show()

# print(result.masks.shape)
# print(result.boxes)



# WARNING ⚠️ imgsz=[720, 1280] must be multiple of max stride 64, updating to [768, 1280]
from pprint import pprint
for (results, ims) in islice(measurements.iter_measurements(
        device=get_default_device(), 
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