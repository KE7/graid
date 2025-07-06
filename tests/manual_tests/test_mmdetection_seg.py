from itertools import islice
from pathlib import Path

from ultralytics.data.augment import LetterBox

from graid.data.ImageLoader import Bdd10kDataset, NuImagesDataset_seg, WaymoDataset_seg
from graid.measurements.InstanceSegmentation import InstanceSegmentationMeasurements
from graid.models.MMDetection import MMdetection_seg
from graid.utilities.common import (
    get_default_device,
    project_root_dir,
    yolo_bdd_seg_transform,
    yolo_nuscene_seg_transform,
    yolo_waymo_seg_transform,
)

shape_transform = LetterBox(new_shape=(768, 1280))

NUM_EXAMPLES_TO_SHOW = 1
NEW_SHAPE = (768, 1280)
BATCH_SIZE = 4

# Instantiate datasets with mask-aware transforms ------------------------------------------------------

bdd = Bdd10kDataset(
    split="val",
    transform=lambda img, lbls: yolo_bdd_seg_transform(img, lbls, NEW_SHAPE),
)

nuscene = NuImagesDataset_seg(
    split="test",
    transform=lambda img, lbls: yolo_nuscene_seg_transform(img, lbls, NEW_SHAPE),
)

waymo = WaymoDataset_seg(
    split="validation",
    transform=lambda img, lbls: yolo_waymo_seg_transform(img, lbls, NEW_SHAPE),
)

# Use project_root_dir() for robust path handling
MMDETECTION_PATH = project_root_dir() / "install" / "mmdetection"

config_file = str(
    MMDETECTION_PATH / "configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py"
)

# Check if checkpoint exists, if not, we'll need to download it
checkpoint_file = str(
    MMDETECTION_PATH
    / "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"
)

# Create checkpoints directory if it doesn't exist
checkpoint_dir = MMDETECTION_PATH / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

# Download checkpoint if it doesn't exist
if not Path(checkpoint_file).exists():
    print(f"Checkpoint file not found at {checkpoint_file}")
    print("You need to download the checkpoint file first.")
    print("Run the following command to download it:")
    print(
        f"wget -P {checkpoint_dir} https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"
    )
    exit(1)

model = MMdetection_seg(config_file, checkpoint_file)

for d in [bdd, nuscene, waymo]:
    # https://docs.ultralytics.com/models/yolov5/#performance-metrics
    print("Using dataset: ", d)
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
        print(results)
        print("")
