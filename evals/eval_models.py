import gc
import json

import psutil
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List

import numpy as np

import ray
import torch
from ray.util.queue import Queue
from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    ImageDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.interfaces.ObjectDetectionI import (
    ObjectDetectionModelI,
    ObjectDetectionUtils,
)
from scenic_reasoning.models.Detectron import Detectron_obj
from scenic_reasoning.models.Ultralytics import RT_DETR, Yolo
from scenic_reasoning.utilities.common import (
    get_default_device,
    project_root_dir,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pickle

##############################################################################
# Experiment configuration
###############################################################################
BATCH_SIZE = 32
num_cpu_workers = 8

###############################################################################
# Model initialization
################################################################################
yolo_v10x = Yolo(model="yolov10x.pt")  # 61.4 Mb
yolo_11x = Yolo(model="yolo11x.pt")  # 109 Mb
rtdetr = RT_DETR("rtdetr-x.pt")  # 129 Mb

retinanet_R_101_FPN_3x_config = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"  # 228MB
retinanet_R_101_FPN_3x_weights = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
retinanet_R_101_FPN_3x = Detectron_obj(
    config_file=retinanet_R_101_FPN_3x_config,
    weights_file=retinanet_R_101_FPN_3x_weights,
)

faster_rcnn_R_50_FPN_3x_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"  # 167MB
faster_rcnn_R_50_FPN_3x_weights = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
faster_rcnn_R_50_FPN_3x = Detectron_obj(
    config_file=faster_rcnn_R_50_FPN_3x_config,
    weights_file=faster_rcnn_R_50_FPN_3x_weights,
)

################################################################################
# Aggregator Actor
################################################################################
@ray.remote
class Aggregator:
    def __init__(self):
        # Dictionary structure to store measurements for each combination
        self.aggregator_data = defaultdict(lambda: {
            "measurements_pen_list": [],
            "measurements_no_pen_list": []
        })

    def add_image_result(
        self,
        dataset: str,
        model: str,
        conf: float,
        measurements_pen: dict,
        measurements_no_pen: dict
    ):
        """
        Store the full measurement dictionaries for each image.
        """
        key = (dataset, model, conf)
        # Append measurements to the appropriate lists
        pen_size_mb = len(pickle.dumps(measurements_pen)) / (1024 * 1024)
        no_pen_size_mb = len(pickle.dumps(measurements_no_pen)) / (1024 * 1024)
        print(f"Pen dict size: {pen_size_mb:.2f} MB, No-pen dict size: {no_pen_size_mb:.2f} MB")
        self.aggregator_data[key]["measurements_pen_list"].append(measurements_pen)
        self.aggregator_data[key]["measurements_no_pen_list"].append(measurements_no_pen)

        # print("Current memory usage: {}%".format(psutil.virtual_memory().percent))
        print("Current number of items in aggregator: {}".format(len(self.aggregator_data)))

    def finalize(self):
        """
        Compute overall metrics for each (dataset, model, conf) combination.
        Returns structured output with averages of relevant metrics.
        """
        final_output = {}
        
        for (dataset, model, conf), data in self.aggregator_data.items():
            pen_measurements = data["measurements_pen_list"]
            no_pen_measurements = data["measurements_no_pen_list"]
            
            # Count TN cases
            tn_count_pen = sum(1 for m in pen_measurements if m.get("TN", 0) == 1)
            tn_count_no_pen = sum(1 for m in no_pen_measurements if m.get("TN", 0) == 1)
            
            # Calculate average mAP excluding TN cases
            non_tn_pen = [m for m in pen_measurements if m.get("TN", 0) != 1]
            non_tn_no_pen = [m for m in no_pen_measurements if m.get("TN", 0) != 1]
            
            avg_map_no_pen = sum((m["map"].item() if m != -1 else 0) for m in non_tn_no_pen) / len(no_pen_measurements)
            avg_map_pen = sum((m["map"].item() if m != -1 else 0) for m in non_tn_pen) / len(pen_measurements)
            avg_map_50_no_pen = sum((m["map_50"].item() if m != -1 else 0) for m in non_tn_no_pen) / len(no_pen_measurements)
            avg_map_75_no_pen = sum((m["map_75"].item() if m != -1 else 0) for m in non_tn_no_pen) / len(no_pen_measurements)
            
            # Store results
            key_str = f"{model}//{dataset}"
            if key_str not in final_output:
                final_output[key_str] = {}
                
            final_output[key_str][conf] = {
                "avg_mAP_no_penalty": avg_map_no_pen,
                "avg_mAP_penalty": avg_map_pen,
                "avg_map_50_no_penalty": avg_map_50_no_pen,
                "avg_map_75_no_penalty": avg_map_75_no_pen,
                "TN_no_penalty_total": tn_count_no_pen,
                "TN_penalty_total": tn_count_pen,
                "total_images": len(pen_measurements)
            }

        return final_output


################################################################################
# CPU post-processing worker
################################################################################
@ray.remote(num_cpus=1)
def cpu_metrics_worker(
    work_queue: Queue, conf_thresholds: List[float], aggregator
):
    """
    Continuously read from the queue, compute metrics (CPU-bound),
    and store per-image results in the aggregator.
    """
    while True:
        item = work_queue.get(block=True)
        print("Current number of items in queue: {}".format(work_queue.qsize()))

        # Sentinel: if item is None, no more data => we're done
        if item is None:
            break

        print("Computing metrics...")

        dataset = item["dataset"]
        model_name = item["model"]
        odrs_list = item["odrs"]
        gt_list = item["gt"]
        images = item["images"]

        for conf in conf_thresholds:
            # For each image in this batch
            for image, odrs, gt in zip(images, odrs_list, gt_list):
                # Filter predictions by confidence
                relevant_odrs = [o for o in odrs if o.score >= conf]

                # No-penalty metrics
                measurements_no_pen = (
                    ObjectDetectionUtils.compute_metrics_for_single_img(
                        relevant_odrs,
                        gt,
                        class_metrics=True,
                        extended_summary=True,
                        penalize_for_extra_predicitions=False,
                        image=image,
                    )
                )

                # Penalty metrics
                measurements_pen = ObjectDetectionUtils.compute_metrics_for_single_img(
                    relevant_odrs,
                    gt,
                    class_metrics=True,
                    extended_summary=True,
                    penalize_for_extra_predicitions=True,
                    image=image,
                )

                # Store in aggregator
                aggregator.add_image_result.remote(
                    dataset=dataset,
                    model=model_name,
                    conf=conf,
                    measurements_pen=measurements_pen,
                    measurements_no_pen=measurements_no_pen
                )

        del odrs_list
        del gt_list
        del images
        del item
        gc.collect()


################################################################################
# GPU-based function that does inference only
################################################################################
@ray.remote(num_gpus=1)
def metric_per_dataset(
    model: ObjectDetectionModelI,
    dataset_fn: Callable[[], "ImageDataset"],
    work_queue: Queue,
    batch_size: int = 16,
):
    """
    Loads images from the dataset, runs inference, pushes results onto 'work_queue'.
    """
    dataset = dataset_fn()
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )

    model.to(device=get_default_device())
    print(f"[GPU Task] Starting inference: {dataset}, with model {model}")
    start_time = time.time()

    for batch in tqdm(data_loader, desc="processing " + str(dataset)):
        images = [sample["image"] for sample in batch]
        x = torch.stack(images).to(get_default_device())
        y = [sample["labels"] for sample in batch]

        predictions = model.identify_for_image_batch(x)
        x = x.cpu()

        # Instead of computing CPU metrics here, push to queue
        work_queue.put(
            {
                "dataset": str(dataset),
                "model": str(model),
                "gt": y,
                "odrs": predictions,
                "images": images,
            }
        )

    end_time = time.time()
    print(
        f"[GPU Task] Finished inference: {dataset}, {model} in {end_time - start_time:.2f} seconds"
    )

    model.to(device="cpu")
    if "cuda" in str(get_default_device()):
        torch.cuda.empty_cache()

    del data_loader
    del dataset
    gc.collect()
    
    return True  


if __name__ == "__main__":
    ray.init()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", 
        "--dataset",
        type=str,
        default="nuimages",
        help="Dataset to evaluate (nuimages, waymo, bdd100k)",
        choices=["nuimages", "waymo", "bdd100k"]
    )

    args = parser.parse_args()
    dataset = args.dataset

    # 1) Prepare datasets
    datasets = []
    # NuImages
    if dataset == "nuimages":
        for split in ["mini", "train", "val"]:
            for size in ["all"]:
                datasets.append(
                    lambda split=split, size=size: NuImagesDataset(
                        split=split,
                        size=size,
                        transform=lambda i, l: yolo_nuscene_transform(
                            i, l, new_shape=(896, 1600)
                        ),
                    )
                )
    # Waymo
    elif dataset == "waymo":
        for split in ["training", "validation"]:
            datasets.append(
                lambda split=split: WaymoDataset(
                    split=split,
                    transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920)),
                )
            )
    # BDD100k
    elif dataset == "bdd100k":
        for split in ["train", "val"]:
            datasets.append(
                lambda split=split: Bdd100kDataset(
                    split=split,
                    transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
                    use_original_categories=False,
                    use_extended_annotations=False,
                )
            )

    models = [
        yolo_v10x,
        yolo_11x,
        rtdetr,
        retinanet_R_101_FPN_3x,
        faster_rcnn_R_50_FPN_3x,
    ]

    # 3) Confidence thresholds
    # confs = [float(c) for c in np.arange(0.05, 0.90, 0.05)]
    confs = [0.01, 0.5]
    # Optionally set a default threshold in each model:
    for m in models:
        m.set_threshold(confs[0])

   # We'll store final results here
    final_results = {}

    # For concurrency, we hold references to all GPU tasks + aggregator states
    active_pairs = []  # list of dicts with keys: aggregator, queue, gpu_task, cpu_workers, pair_label

    # Launch everything in parallel
    for dfn in datasets:
        for m in models:
            aggregator = Aggregator.remote()
            queue = Queue()
            # Start however many CPU workers you want
            cpu_workers = [
                cpu_metrics_worker.remote(queue, confs, aggregator)
                for _ in range(num_cpu_workers)
            ]
            # Launch GPU job
            gpu_task = metric_per_dataset.remote(m, dfn, queue, batch_size=BATCH_SIZE)

            label = f"{str(m)}-{str(dfn())}"
            print(f"Starting: {label}")
            active_pairs.append({
                "aggregator": aggregator,
                "queue": queue,
                "cpu_workers": cpu_workers,
                "gpu_task": gpu_task,
                "pair_label": label
            })

    # Now wait for GPU tasks as they finish in any order
    pending_gpu_tasks = [p["gpu_task"] for p in active_pairs]

    while pending_gpu_tasks:
        # Wait until at least 1 GPU task is done
        done, pending_gpu_tasks = ray.wait(pending_gpu_tasks, num_returns=1)
        completed_gpu_id = done[0]

        # Find the corresponding aggregator/cpu-workers in active_pairs
        pair_info = next(x for x in active_pairs if x["gpu_task"] == completed_gpu_id)

        label = pair_info["pair_label"]
        aggregator_actor = pair_info["aggregator"]
        queue_actor = pair_info["queue"]
        cpu_worker_ids = pair_info["cpu_workers"]

        # 1) GPU done => signal CPU workers to shut down
        for _ in range(len(cpu_worker_ids)):
            queue_actor.put(None)

        # 2) Wait for CPU workers to exit
        ray.get(cpu_worker_ids)

        # 3) Finalize aggregator => get result
        pair_result = ray.get(aggregator_actor.finalize.remote())

        # Store that result in local Python memory
        final_results[label] = pair_result

        # Drop references so Ray can GC
        active_pairs.remove(pair_info)
        del aggregator_actor
        del queue_actor
        del cpu_worker_ids
        gc.collect()
        print(f"Finished processing: {label}")
   

    # 5) Save results
    output_file_path = project_root_dir() / "evals" / "results.json"
    with open(output_file_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"Done. Results saved to {output_file_path}")



# yolo_v10x = Yolo(model="yolov10x.pt")  # 61.4 Mb
# yolo_11x = Yolo(model="yolo11x.pt")  # 109 Mb
# rtdetr = RT_DETR("rtdetr-x.pt")  # 129 Mb

# retinanet_R_101_FPN_3x_config = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"  # 228MB
# retinanet_R_101_FPN_3x_weights = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
# retinanet_R_101_FPN_3x = Detectron_obj(
#     config_file=retinanet_R_101_FPN_3x_config,
#     weights_file=retinanet_R_101_FPN_3x_weights,
# )

# faster_rcnn_R_50_FPN_3x_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"  # 167MB
# faster_rcnn_R_50_FPN_3x_weights = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# faster_rcnn_R_50_FPN_3x = Detectron_obj(
#     config_file=faster_rcnn_R_50_FPN_3x_config,
#     weights_file=faster_rcnn_R_50_FPN_3x_weights,
# )


# # @persistent_cache(str(project_root_dir() / "evals" / "eval_models_cache.pkl"))
# @ray.remote(num_gpus=1)
# def metric_per_dataset(
#     model: ObjectDetectionModelI,
#     dataset_fn: Callable[[], ImageDataset],
#     conf_thresholds: List[float],
# ):
#     dataset = dataset_fn()
#     data_loader = DataLoader(
#         dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         collate_fn=lambda x: x,
#     )

#     model.to(device=get_default_device())

#     no_pen_measurements = []
#     pen_measurements = []

#     for batch in tqdm(data_loader, desc="processing " + str(dataset)):
#         x = torch.stack([sample["image"] for sample in batch])
#         y = [sample["labels"] for sample in batch]

#         x = x.to(device=get_default_device())
#         prediction = model.identify_for_image_batch(x)
#         x = x.cpu()

#         # TODO: this loop should be split into another ray task since it's cpu bound
#         for idx, (odrs, gt) in enumerate(
#             zip(prediction, y)
#         ):  # odr = object detection result, gt = ground truth
#             for conf in conf_thresholds:
#                 relevant_odrs = list(filter(lambda x: x.score >= conf, odrs))

#                 measurements: dict = (
#                     ObjectDetectionUtils.compute_metrics_for_single_img(
#                         relevant_odrs,
#                         gt,
#                         class_metrics=True,
#                         extended_summary=True,
#                         image=x[idx],
#                         penalize_for_extra_predicitions=False,
#                     )
#                 )

#                 measurements_with_penalty: dict = (
#                     ObjectDetectionUtils.compute_metrics_for_single_img(
#                         relevant_odrs,
#                         gt,
#                         class_metrics=True,
#                         extended_summary=True,
#                         image=x[idx],
#                         penalize_for_extra_predicitions=True,
#                     )
#                 )

#                 no_pen_measurements.append(measurements)
#                 pen_measurements.append(measurements_with_penalty)

#     model.to(device="cpu")
#     del data_loader

#     # Count TN cases
#     tn_count_pen = sum(1 for m in pen_measurements if m.get("TN", 0) == 1)
#     tn_count_no_pen = sum(1 for m in no_pen_measurements if m.get("TN", 0) == 1)
    
#     # Calculate average mAP excluding TN cases
#     non_tn_pen = [m for m in pen_measurements if m.get("TN", 0) != 1]
#     non_tn_no_pen = [m for m in no_pen_measurements if m.get("TN", 0) != 1]
    
#     avg_map_no_pen = sum((m["map"].item() if m != -1 else 0) for m in non_tn_no_pen) / len(no_pen_measurements)
#     avg_map_pen = sum((m["map"].item() if m != -1 else 0) for m in non_tn_pen) / len(pen_measurements)
#     avg_map_50_no_pen = sum((m["map_50"].item() if m != -1 else 0) for m in non_tn_no_pen) / len(no_pen_measurements)
#     avg_map_75_no_pen = sum((m["map_75"].item() if m != -1 else 0) for m in non_tn_no_pen) / len(no_pen_measurements)
                
#     output = {
#         "avg_mAP_no_penalty": avg_map_no_pen,
#         "avg_mAP_penalty": avg_map_pen,
#         "avg_map_50_no_penalty": avg_map_50_no_pen,
#         "avg_map_75_no_penalty": avg_map_75_no_pen,
#         "TN_no_penalty_total": tn_count_no_pen,
#         "TN_penalty_total": tn_count_pen,
#         "total_images": len(pen_measurements),
#         "model": str(model),
#         "dataset": str(dataset),
#         "confs": conf_thresholds,
#     }
#     return output
    


# if __name__ == "__main__":
#     ray.init()

#     datasets = []
#     # NuImages
#     for split in ["mini", "train", "val"]:
#         for size in ["all"]:
#             datasets.append(
#                 lambda: NuImagesDataset(
#                     split=split,
#                     size=size,
#                     transform=lambda i, l: yolo_nuscene_transform(
#                         i, l, new_shape=(896, 1600)
#                     ),
#                 )
#             )
#     # Waymo
#     for split in ["training", "validation"]:
#         datasets.append(
#             lambda: WaymoDataset(
#                 split=split,
#                 transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920)),
#             )
#         )
#     # BDD100k
#     for split in ["train", "val"]:
#         datasets.append(
#             lambda: Bdd100kDataset(
#                 split=split,
#                 transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
#                 use_original_categories=False,
#                 use_extended_annotations=False,
#             )
#         )

#     models = [
#         yolo_v10x,
#         yolo_11x,
#         rtdetr,
#         retinanet_R_101_FPN_3x,
#         faster_rcnn_R_50_FPN_3x,
#     ]
#     confs = [float(c) for c in np.arange(0.05, 0.90, 0.05)]

#     for model in models:
#         model.set_threshold(confs[0])

#     BATCH_SIZE = 32

#     tasks = []
#     for d in datasets:
#         for model in models:
#             task_id = metric_per_dataset.remote(model, d, confs)
#             tasks.append(task_id)

#     results = ray.get(tasks)
#     output_file = {}
#     for result in results:
#         k = result["model"] + "//" + result["dataset"]
#         output_file[k] = result

#     print(output_file)

#     output_file_path = project_root_dir() / "evals" / "results.json"
#     with open(output_file_path, "w") as f:
#         json.dump(output_file, f, indent=4)
