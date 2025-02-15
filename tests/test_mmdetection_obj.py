from itertools import islice
from scenic_reasoning.data.ImageLoader import Bdd100kDataset, NuImagesDataset, WaymoDataset
from scenic_reasoning.models.MMDetection import MMdetection_obj
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.utilities.common import get_default_device, yolo_transform, maskrcnn_waymo_transform
import torch
from ultralytics.data.augment import LetterBox

NUM_EXAMPLES_TO_SHOW = 3
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
    # transform=transform_image_for_yolo,  
    use_original_categories=False,
    use_extended_annotations=False,
)

niu = NuImagesDataset(split='test')

# waymo = WaymoDataset(split="validation", transform=lambda i, l: maskrcnn_waymo_transform(i, l, new_shape=(1333, 640)))
waymo = WaymoDataset(split="validation", transform=lambda i, l: maskrcnn_waymo_transform(i, l, (640, 1333), "bbox"))

config_file = '../install/mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'
checkpoint_file = '../install/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

model = MMdetection_obj(config_file, checkpoint_file)


for d in [waymo]:
        
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