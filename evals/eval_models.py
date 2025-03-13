import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import time
from collections import defaultdict
from typing import Callable, List

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
    persistent_cache,
    project_root_dir,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        # Dictionary of dicts:
        # aggregator_data[(dataset, model, conf)] = {
        #   "sum_mAP_no_penalty": float,
        #   "count_mAP_no_penalty": int,
        #   "sum_mAP_penalty": float,
        #   "count_mAP_penalty": int,
        #   "TN_no_penalty": int,
        #   "TN_penalty": int,
        # }
        self.aggregator_data = defaultdict(
            lambda: {
                "sum_mAP_no_penalty": 0.0,
                "count_mAP_no_penalty": 0,
                "sum_mAP_penalty": 0.0,
                "count_mAP_penalty": 0,
                "TN_no_penalty": 0,
                "TN_penalty": 0,
            }
        )

    def add_image_result(
        self,
        dataset: str,
        model: str,
        conf: float,
        is_TN_no_penalty: bool,
        is_TN_penalty: bool,
        mAP_no_penalty: float,
        mAP_penalty: float,
    ):
        """
        Store partial sums/counts for a single image at a particular confidence threshold.
        """
        key = (dataset, model, conf)
        data = self.aggregator_data[key]

        if is_TN_no_penalty:
            # If the image was "TN" => no ground-truth and no predictions
            data["TN_no_penalty"] += 1
        else:
            # If not TN, add the mAP to sum and increment count
            data["sum_mAP_no_penalty"] += mAP_no_penalty
            data["count_mAP_no_penalty"] += 1

        if is_TN_penalty:
            data["TN_penalty"] += 1
        else:
            data["sum_mAP_penalty"] += mAP_penalty
            data["count_mAP_penalty"] += 1

    def finalize(self):
        """
        Compute overall average for each (dataset, model, conf).
        Returns a dict of form:
          final_output["model//dataset"]["conf"]["avg_mAP_no_penalty"] = ...
        """
        final_output = {}
        # self.aggregator_data: keys => (dataset, model, conf)
        for (dataset, model, conf), stats in self.aggregator_data.items():
            # Unpack partial sums
            sum_mAP_no_pen = stats["sum_mAP_no_penalty"]
            cnt_mAP_no_pen = stats["count_mAP_no_penalty"]
            sum_mAP_pen = stats["sum_mAP_penalty"]
            cnt_mAP_pen = stats["count_mAP_penalty"]
            tn_no_pen = stats["TN_no_penalty"]
            tn_pen = stats["TN_penalty"]

            # Compute final averages
            if cnt_mAP_no_pen > 0:
                avg_mAP_no_pen = sum_mAP_no_pen / cnt_mAP_no_pen
            else:
                avg_mAP_no_pen = None  # or 0.0

            if cnt_mAP_pen > 0:
                avg_mAP_pen = sum_mAP_pen / cnt_mAP_pen
            else:
                avg_mAP_pen = None

            # Store in final_output
            key_str = f"{model}//{dataset}"
            if key_str not in final_output:
                final_output[key_str] = {}
            final_output[key_str][conf] = {
                "avg_mAP_no_penalty": avg_mAP_no_pen,
                "avg_mAP_penalty": avg_mAP_pen,
                "TN_no_penalty_total": tn_no_pen,
                "TN_penalty_total": tn_pen,
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

        # Sentinel: if item is None, no more data => we're done
        if item is None:
            break

        dataset = item["dataset"]
        model_name = item["model"]
        odrs_list = item["odrs"]
        gt_list = item["gt"]

        for conf in conf_thresholds:
            # For each image in this batch
            for odrs, gt in zip(odrs_list, gt_list):
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
                    )
                )
                is_TN_no_penalty = measurements_no_pen["TN"] == 1

                # If not TN, retrieve mAP
                if not is_TN_no_penalty:
                    mAP_no_pen = measurements_no_pen["map"].item()
                else:
                    mAP_no_pen = 0.0  # won't matter if it's TN

                # Penalty metrics
                measurements_pen = ObjectDetectionUtils.compute_metrics_for_single_img(
                    relevant_odrs,
                    gt,
                    class_metrics=True,
                    extended_summary=True,
                    penalize_for_extra_predicitions=True,
                )
                is_TN_penalty = measurements_pen["TN"] == 1

                if not is_TN_penalty:
                    mAP_pen = measurements_pen["map"].item()
                else:
                    mAP_pen = 0.0

                # Store in aggregator
                aggregator.add_image_result.remote(
                    dataset=dataset,
                    model=model_name,
                    conf=conf,
                    is_TN_no_penalty=is_TN_no_penalty,
                    is_TN_penalty=is_TN_penalty,
                    mAP_no_penalty=mAP_no_pen,
                    mAP_penalty=mAP_pen,
                )


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
        x = torch.stack([sample["image"] for sample in batch]).to("cuda")
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
            }
        )

    model.to(device="cpu")
    end_time = time.time()
    print(
        f"[GPU Task] Finished inference: {dataset}, {model} in {end_time - start_time:.2f} seconds"
    )
    return True


if __name__ == "__main__":
    ray.init()

    # 1) Prepare datasets
    datasets = []
    # NuImages
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
    for split in ["training", "validation"]:
        datasets.append(
            lambda split=split: WaymoDataset(
                split=split,
                transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920)),
            )
        )
    # BDD100k
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
    confs = [float(c) for c in np.arange(0.05, 0.90, 0.05)]
    # Optionally set a default threshold in each model:
    for m in models:
        m.set_threshold(confs[0])

    BATCH_SIZE = 32

    # 4) Create a global queue
    work_queue = Queue()

    # 5) Create aggregator
    aggregator = Aggregator.remote()

    # 6) Launch GPU tasks (parallel) for each (model, dataset)
    gpu_tasks = []
    for dataset_fn in datasets:
        for model in models:
            gpu_task_id = metric_per_dataset.remote(model, dataset_fn, work_queue, BATCH_SIZE)
            gpu_tasks.append(gpu_task_id)

    # 7) Launch a certain number of CPU workers
    num_cpu_workers = 16
    cpu_workers = []
    for _ in range(num_cpu_workers):
        worker_id = cpu_metrics_worker.remote(work_queue, confs, aggregator)
        cpu_workers.append(worker_id)

    # 8) Wait for all GPU tasks to complete
    ray.get(gpu_tasks)
    print("All GPU tasks are done. Signaling CPU workers to shut down...")

    # 9) Signal CPU workers to shut down by sending 'None' sentinel
    for _ in range(num_cpu_workers):
        work_queue.put(None)

    # 10) Wait for all CPU workers to exit
    ray.get(cpu_workers)

    # 11) Gather final results from aggregator
    final_results = ray.get(aggregator.finalize.remote())

    # 12) Write out results to JSON
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


# @persistent_cache(str(project_root_dir() / "evals" / "eval_models_cache.pkl"))
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

#     mAP_at_conf = defaultdict(list)
#     mAPs_with_penalty_at_conf = defaultdict(list)
#     TN_count = defaultdict(int)
#     TN_count_with_penalty = defaultdict(int)

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
#                 full_image_result = dict()
#                 full_image_result["image"] = x[idx]
#                 full_image_result["measurements"] = measurements
#                 full_image_result["predictions"] = relevant_odrs
#                 full_image_result["labels"] = gt

#                 if measurements["TN"] == 1:
#                     print("Both ground truth and predictions are empty. Ignore")
#                     TN_count[conf] += 1
#                 else:
#                     mAP = measurements["map"]
#                     mAP_at_conf[conf].append(mAP.item())

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
#                 full_result_with_penalty = dict()
#                 full_result_with_penalty["image"] = x[idx]
#                 full_result_with_penalty["measurements"] = measurements_with_penalty
#                 full_result_with_penalty["predictions"] = relevant_odrs
#                 full_result_with_penalty["labels"] = gt

#                 if measurements_with_penalty["TN"] == 1:
#                     print("Both ground truth and predictions are empty. Ignore")
#                     TN_count_with_penalty[conf] += 1
#                 else:
#                     mAP = measurements_with_penalty["map"]
#                     mAPs_with_penalty_at_conf[conf].append(mAP.item())

#     mAPs = {
#         conf: sum(mAP_at_conf[conf]) / len(mAP_at_conf[conf]) for conf in mAP_at_conf
#     }
#     mAPs_with_penalty = {
#         conf: sum(mAPs_with_penalty_at_conf[conf])
#         / len(mAPs_with_penalty_at_conf[conf])
#         for conf in mAPs_with_penalty_at_conf
#     }

#     model.to(device="cpu")

#     return {
#         "dataset": str(dataset),
#         "model": str(model),
#         "confidence_thresholds": conf_thresholds,
#         "average_mAP": mAPs,
#         "average_mAP_with_penalty": mAPs_with_penalty,
#         "TNs": TN_count,
#         "TN_with_penaltys": TN_count_with_penalty,
#     }


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
#     # confs = [0.2, 0.5, 0.7]

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
