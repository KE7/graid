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


yolo_v8n = Yolo(model="yolov8n.pt")
yolo_11n = Yolo(model="yolo11n.pt")
rtdetr = RT_DETR("rtdetr-l.pt")

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
def metric_per_dataset(model, dataset_name, conf):
    if dataset_name == "NuImages":
        dataset = NuImagesDataset(split="mini", size="all", transform=lambda i, l: yolo_nuscene_transform(i, l, new_shape=(896, 1600)))
    elif dataset_name == "Waymo":
        dataset = WaymoDataset(split="validation", transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920)))
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
    TN_count = 0
    mAP_iterator = measurements.iter_measurements(
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
            if result["measurements"]['TN'] == 1:
                print("Both ground truth and predictions are empty. Ignore")
                TN_count += 1
                continue
            mAP = result["measurements"]['map']
            mAPs.append(mAP.item())
    
    mAPs_fake = []
    TN_count_fake = 0
    mAP_iterator_fake = measurements.iter_measurements(
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
            if result["measurements"]['TN'] == 1:
                print("Both ground truth and predictions are empty. Ignore")
                TN_count_fake += 1
                continue
            mAP = result["measurements"]['map']
            mAPs_fake.append(mAP.item())
    
    return {
        'dataset': dataset_name,
        'model': str(model),
        'confidence': conf,
        'average_mAP': sum(mAPs) / len(mAPs),
        'fake_average_mAP': sum(mAPs_fake) / len(mAPs_fake),
        'TN': TN_count,
        'TN_fake': TN_count_fake
        }



if __name__ == "__main__":
    ray.init()

    datasets = ["NuImages"]
    models = [yolo_v8n, yolo_11n, rtdetr, retinanet_R_101_FPN_3x, faster_rcnn_R_50_FPN_3x]
    # confs = [c for c in np.arange(0.05, 0.90, 0.05)]
    confs = [0.2]
    BATCH_SIZE = 8

    tasks = []
    for d in datasets:
        for model in models:
            for conf in confs:
                task_id = metric_per_dataset.remote(model, d, conf)
                tasks.append(task_id)

    results = ray.get(tasks)
    output_file = {}
    for result in results:
        k = result["model"] + "//" + result["dataset"] + "//" + str(result["confidence"])
        output_file[k] = result
    
    print(output_file)
    
    output_file_path = project_root_dir() / "scenic_reasoning"/ "src" / "scenic_reasoning" / "eval" / "results.json"
    with open(output_file_path, 'w') as f:
        json.dump(output_file, f, indent=4)
