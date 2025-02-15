from scenic_reasoning.data.ImageLoader import Bdd10kDataset, NuImagesDataset_seg, WaymoDataset_seg
from scenic_reasoning.models.MMDetection import MMdetection_seg
from scenic_reasoning.measurements.InstanceSegmentation import InstanceSegmentationMeasurements

from scenic_reasoning.utilities.common import get_default_device, yolo_waymo_transform
from itertools import islice
from ultralytics.data.augment import LetterBox

shape_transform = LetterBox(new_shape=(768, 1280))

NUM_EXAMPLES_TO_SHOW = 3
BATCH_SIZE = 1

bdd = Bdd10kDataset(
    split="val", 
)

waymo = WaymoDataset_seg(
    split="validation", transform=lambda i, l: yolo_waymo_transform(i, l, stride=32)
)

nuscene = NuImagesDataset_seg(
    split="test"
)


config_file = '../install/mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'
checkpoint_file = '../install/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

model = MMdetection_seg(config_file, checkpoint_file)

for d in [bdd]:
    # https://docs.ultralytics.com/models/yolov5/#performance-metrics
    measurements = InstanceSegmentationMeasurements(model, d, batch_size=BATCH_SIZE, collate_fn=lambda x: x)
    # WARNING ⚠️ imgsz=[720, 1280] must be multiple of max stride 64, updating to [768, 1280]
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
