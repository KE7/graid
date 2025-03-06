import numpy as np
from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionUtils
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.models.Ultralytics import Yolo, RT_DETR
from scenic_reasoning.models.Detectron import Detectron_obj
from scenic_reasoning.utilities.common import (
    get_default_device,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
from scenic_reasoning.utilities.common import project_root_dir
import json
import ray
from tqdm import tqdm


config_file = "../install/mmdetection/configs/mm_grounding_dino/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py"
checkpoint_file = "../install/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"

model = MMdetection_obj(config_file, checkpoint_file)


@ray.remote(num_gpus=1)
def metric_per_dataset(model, dataset_name, conf):
    if dataset_name == "nu":
        dataset = NuImagesDataset(split="mini", size="all", transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(768, 1280)))
    elif dataset_name == "waymo":
        dataset = WaymoDataset(split="validation", transform=lambda i, l: yolo_waymo_transform(i, l, (768, 1280)))
    else:
        dataset = Bdd100kDataset(
            split="val",
            transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
            use_original_categories=False,
            use_extended_annotations=False,
            )

    measurements = ObjectDetectionMeasurements(
        model, dataset, batch_size=BATCH_SIZE, collate_fn=lambda x: x
    )
    mAPs = []
    mAP_iterator = measurements.iter_measurements(
        imgsz=[768, 1280],
        bbox_offset=24,
        debug=False,
        conf=conf,
        class_metrics=True,
        extended_summary=True,
        agnostic_nms=True,
        fake_boxes=False
        )
    for results in tqdm(mAP_iterator, desc="processing mAP measurements"):
        for result in results:
            mAP = result["measurements"]['map']
            mAPs.append(mAP.item())
    
    mAPs_fake = []
    mAP_iterator_fake = measurements.iter_measurements(
        imgsz=[768, 1280],
        bbox_offset=24,
        debug=False,
        conf=conf,
        class_metrics=True,
        extended_summary=True,
        agnostic_nms=True,
        fake_boxes=True
        )
    for results in tqdm(mAP_iterator_fake, desc="processing fake mAP measurements"):
        for result in results:
            mAP = result["measurements"]['map']
            mAPs_fake.append(mAP.item())
    
    stats = ({
        'dataset': type(d).__name__,
        'model': type(model).__name__,
        'confidence': conf,
        'average_mAP': sum(mAPs) / len(mAPs),
        'fake_average_mAP': sum(mAPs_fake) / len(mAPs_fake)
        })
    print(stats)
    return stats


if __name__ == "__main__":
    ray.init()

    datasets = ["nu"]
    models = [retinanet_R_101_FPN_3x]
    # confs = [c for c in np.arange(0.05, 0.90, 0.05)]
    confs = [0.2]
    BATCH_SIZE = 4

    tasks = []
    for d in datasets:
        for model in models:
            for conf in confs:
                task_id = metric_per_dataset.remote(model, d, conf)
                tasks.append(task_id)

    results = ray.get(tasks)

    output_file = project_root_dir() / "src" / "scenic_reasoning" / "evaluation" / "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
