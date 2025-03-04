from itertools import islice
from scenic_reasoning.data.ImageLoader import Bdd100kDataset, NuImagesDataset, WaymoDataset
from scenic_reasoning.models.MMDetection import MMdetection_obj
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionUtils
from scenic_reasoning.utilities.common import yolo_waymo_transform
from PIL import Image
import numpy as np


NUM_EXAMPLES_TO_SHOW = 3
BATCH_SIZE = 1

bdd = Bdd100kDataset(
    split="val",
    use_original_categories=False,
    use_extended_annotations=False,
)

niu = NuImagesDataset(split="test", size="full")

waymo = WaymoDataset(
    split="validation", transform=lambda i, l: yolo_waymo_transform(i, l, (640, 1333))
)

config_file = "../install/mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py"
checkpoint_file = "../install/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"

model = MMdetection_obj(config_file, checkpoint_file)


for d in [bdd, niu, waymo]:

    measurements = ObjectDetectionMeasurements(
        model, d, batch_size=BATCH_SIZE, collate_fn=lambda x: x
    )  # hacky way to avoid RuntimeError: each element in list of batch should be of equal size

    # WARNING ⚠️ imgsz=[720, 1280] must be multiple of max stride 64, updating to [768, 1280]
    for results in islice(measurements.iter_measurements(
            # device=get_default_device(), 
            imgsz=[768, 1280],
            bbox_offset=24,
            debug=True,
            conf=0.1,
            class_metrics=True,
            extended_summary=True,
            ), 
        NUM_EXAMPLES_TO_SHOW):
        for i in range(len(results)):
            print("gt classes:", [c._class for c in results[i]["labels"]])
            print("pred classes:", [c._class for c in results[i]["predictions"]])

            print(f"{i}th image")
            measurements = results[i]["measurements"]
            print(measurements)
            print("global map", measurements['map'])
            print("map 50", measurements['map_50'])
            print("map 75", measurements['map_75'])
            print("map_small", measurements['map_small'])
            print("map_medium", measurements['map_medium'])
            print("map_large", measurements['map_large'])
            print("mar_small", measurements['mar_small'])
            print("mar_medium", measurements['mar_medium'])
            print("mar_large", measurements['mar_large'])

            ObjectDetectionUtils.show_image_with_detections_and_gt(
                Image.fromarray(results[i]["image"].permute(1, 2, 0).numpy().astype(np.uint8)),
                detections=results[i]["predictions"],
                ground_truth=results[i]["labels"],   
            )
