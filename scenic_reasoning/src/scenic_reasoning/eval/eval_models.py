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




bdd = Bdd100kDataset(
    split="val",
    transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
    use_original_categories=False,
    use_extended_annotations=False,
)

nu = NuImagesDataset(split="mini", size="all", transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(768, 1280)))

waymo = WaymoDataset(split="validation", transform=lambda i, l: yolo_waymo_transform(i, l, (768, 1280)))

# yolo_v8n = Yolo(model="yolov8n.pt")
# yolo_11n = Yolo(model="yolo11n.pt")
# rtdetr = RT_DETR("rtdetr-l.pt")

retinanet_R_101_FPN_3x_config = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
retinanet_R_101_FPN_3x_weights = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
retinanet_R_101_FPN_3x = Detectron_obj(
    config_file=retinanet_R_101_FPN_3x_config, 
    weights_file=retinanet_R_101_FPN_3x_weights
)

faster_rcnn_R_50_FPN_3x_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
faster_rcnn_R_50_FPN_3x_weights = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
faster_rcnn_R_50_FPN_3x = Detectron_obj(
    config_file=faster_rcnn_R_50_FPN_3x_config, 
    weights_file=faster_rcnn_R_50_FPN_3x_weights
)



@ray.remote(num_gpus=1)
def metric_per_dataset(model, dataset, conf):
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
    
    print({
        'dataset': type(d).__name__,
        'model': type(model).__name__,
        'confidence': conf,
        'average_mAP': sum(mAPs) / len(mAPs),
        'fake_average_mAP': sum(mAPs_fake) / len(mAPs_fake)
        })
    return {
        'dataset': type(d).__name__,
        'model': type(model).__name__,
        'confidence': conf,
        'average_mAP': sum(mAPs) / len(mAPs),
        'fake_average_mAP': sum(mAPs_fake) / len(mAPs_fake)
        }



if __name__ == "__main__":
    ray.init()

    waymo_ref = ray.put(waymo)
    bdd_ref = ray.put(bdd)
    nu_ref = ray.put(nu)
    datasets = [nu_ref]
    models = [retinanet_R_101_FPN_3x]
    # confs = [c for c in np.arange(0.05, 0.90, 0.05)]
    confs = [0.2, 0.3]
    BATCH_SIZE = 4

    tasks = []
    for d in datasets:
        for model in models:
            for conf in confs:
                task_id = metric_per_dataset.remote(model, d, conf)
                tasks.append(task_id)

    print(tasks)
    # results = ray.get([metric_per_dataset.remote(retinanet_R_101_FPN_3x, nu_ref, 0.2)])
    results = ray.get(tasks)

    output_file = project_root_dir() / "src" / "scenic_reasoning " / "evaluation" / "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
