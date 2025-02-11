from itertools import islice
from scenic_reasoning.data.ImageLoader import Bdd100kDataset, NuImagesDataset, WaymoDataset
from scenic_reasoning.models.MMDetection import MMdetection_obj
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.utilities.common import get_default_device
import torch
from ultralytics.data.augment import LetterBox

NUM_EXAMPLES_TO_SHOW = 3
BATCH_SIZE = 2

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
    # transform=transform_image_for_yolo,  
    use_original_categories=False,
    use_extended_annotations=False,
)



# niu = NuImagesDataset(split='test')

# waymo = WaymoDataset(split="validation")

# # https://docs.ultralytics.com/models/yolov5/#performance-metrics
model_name = 'retinanet_r50-caffe_fpn_1x_coco'
checkpoint = '../install/mmdetection/checkpoints/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth'
model = MMdetection_obj(model=model_name, checkpoint=checkpoint)

import cv2
image = cv2.imread('../demo/demo.jpg')
print("???????", image.shape)

# results =  model.identify_for_image([image],  ['../demo/demo2.jpg'])


for d in [bdd]:
        
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