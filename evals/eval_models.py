import argparse
import gc
import json
import logging
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
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

allowed_gpus = [0, 1, 2, 3, 5, 6]
num_cpu_workers = 8
BATCH_SIZE = 32
logger = logging.getLogger("ray")

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


@ray.remote(num_gpus=1)
def producer(
    model: ObjectDetectionModelI,
    model_name: str,
    dataset_fn: Callable[[], ImageDataset],
    work_queue: Queue,
    batch_size: int,
):
    dataset = dataset_fn()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )

    print(f"[GPU Task] Starting inference: {dataset}, with model {model_name}")
    start_time = time.time()

    for batch in tqdm(dataloader, desc="Processing " + str(dataset)):
        images = [sample["image"] for sample in batch]
        x = torch.stack(images).to(get_default_device())
        y = [sample["labels"] for sample in batch]

        predictions = model.identify_for_image_batch(x)
        work_queue.put(
            {
                "dataset": str(dataset),
                "model": model_name,
                "gt": y,
                "odrs": predictions,
                "images": images,
            }
        )
        x = x.cpu()

    end_time = time.time()
    print(
        f"[GPU Task] Finished inference: {dataset}, {model_name} in {end_time - start_time:.2f} seconds"
    )
    logger.info(
        f"[GPU Task] Finished inference: {dataset}, {model_name} in {end_time - start_time:.2f} seconds"
    )
    for _ in range(num_cpu_workers):
        work_queue.put(None)  # Signal completion

    torch.cuda.empty_cache()
    del dataset
    del dataloader
    gc.collect()


@ray.remote(num_cpus=1)
def consumer(
    work_queue,
    results_queue,
    confs: List[float],
):
    results = dict()
    while True:
        item = work_queue.get()
        if item is None:
            break

        print(
            f"[CPU Task] Processing batch item from: {item['dataset']}, {item['model']}"
        )
        dataset = item["dataset"]
        model_name = item["model"]
        gt_list = item["gt"]
        odrs_list = item["odrs"]
        images = item["images"]

        for conf in confs:
            for image, odrs, gt in zip(images, odrs_list, gt_list):
                key = (dataset, model_name, conf)
                relevant_odrs = [o for o in odrs if o.score >= conf]

                metrics_no_pen = ObjectDetectionUtils.compute_metrics_for_single_img(
                    relevant_odrs,
                    gt,
                    class_metrics=True,
                    extended_summary=True,
                    penalize_for_extra_predicitions=False,
                    image=image,
                )
                metrics_pen = ObjectDetectionUtils.compute_metrics_for_single_img(
                    relevant_odrs,
                    gt,
                    class_metrics=True,
                    extended_summary=True,
                    penalize_for_extra_predicitions=True,
                    image=image,
                )

                if key not in results:
                    results[key] = {
                        "metrics_no_pen": [],
                        "metrics_pen": [],
                    }
                results[key]["metrics_no_pen"].append(metrics_no_pen)
                results[key]["metrics_pen"].append(metrics_pen)

        del images
        del gt_list
        del odrs_list
        del item
        gc.collect()

    print(f"[CPU Task] Finished processing")
    results_queue.put(None)  # Signal completion
    results_queue.put(results)


def aggregate_results(results_queue, num_workers):
    aggregator = defaultdict(lambda: {"metrics_no_pen": [], "metrics_pen": []})

    finished_workers = 0
    while finished_workers < num_workers:
        print(
            f"Wating for results: {finished_workers}/{num_workers}. Results queue remaining size: {results_queue.qsize()}"
        )
        result = results_queue.get()
        if result is None:
            finished_workers = finished_workers + 1
            continue

        for key, metrics in result.items():
            aggregator[key]["metrics_no_pen"].extend(metrics["metrics_no_pen"])
            aggregator[key]["metrics_pen"].extend(metrics["metrics_pen"])

    return aggregator


def finalize_aggregator(aggregator):
    final_output = dict()
    for (dataset, model, conf), metrics in tqdm(aggregator.items(), desc="Finalizing"):
        print(f"Finalizing: {dataset}, {model}, {conf}")
        pen_metrics = metrics["metrics_pen"]
        no_pen_metrics = metrics["metrics_no_pen"]

        TN_count_pen = sum([m["TN"] for m in pen_metrics])
        TN_count = sum([m["TN"] for m in no_pen_metrics])

        avg_map_pen = sum([m["map"] for m in pen_metrics]) / len(pen_metrics)
        avg_map_50_pen = sum([m["map_50"] for m in pen_metrics]) / len(pen_metrics)
        avg_map_75_pen = sum([m["map_75"] for m in pen_metrics]) / len(pen_metrics)

        avg_map = sum([m["map"] for m in no_pen_metrics]) / len(no_pen_metrics)
        avg_map_50 = sum([m["map_50"] for m in no_pen_metrics]) / len(no_pen_metrics)
        avg_map_75 = sum([m["map_75"] for m in no_pen_metrics]) / len(no_pen_metrics)

        key = f"{model}-{dataset}"
        if key not in final_output:
            final_output[key] = dict()

        final_output[key][conf] = {
            "TN_count_pen": TN_count_pen,
            "avg_map_pen": avg_map_pen.item(),
            "avg_map_50_pen": avg_map_50_pen.item(),
            "avg_map_75_pen": avg_map_75_pen.item(),
            "TN_count": TN_count,
            "avg_map": avg_map.item(),
            "avg_map_50": avg_map_50.item(),
            "avg_map_75": avg_map_75.item(),
            "total_samples": len(pen_metrics),
        }

    from pprint import pprint

    print("Finalizing results: ", final_output)
    pprint(f"Finalizing done: {final_output}")
    return final_output


def is_gpu_available(id: int = 0, p: float = 0.8) -> bool:
    """
    Check if a GPU on the specified device ID has at least p% memory available.
    """
    try:
        gpu_info = torch.cuda.get_device_properties(id)
        gpu_memory_available = gpu_info.total_memory * p
        gpu_memory_used = torch.cuda.memory_allocated(id)
        return (gpu_memory_available - gpu_memory_used) > 0
    except Exception as e:
        print(f"Error checking GPU {id}: {e}")
        return False


if __name__ == "__main__":
    ray.init()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="nuimages",
        help="Dataset to evaluate (nuimages, waymo, bdd100k)",
        choices=["nuimages", "waymo", "bdd100k"],
        required=True,
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
                    transform=lambda i, l: yolo_bdd_transform(
                        i, l, new_shape=(768, 1280)
                    ),
                    use_original_categories=False,
                    use_extended_annotations=False,
                )
            )

    confs = [float(c) for c in np.arange(0.05, 0.90, 0.05)]

    # 2) Prepare models
    models = [
        # (yolo_v10x, "yolo_v10x"),
        # (yolo_11x, "yolo_11x"),
        # (rtdetr, "rtdetr"),
        # (retinanet_R_101_FPN_3x, "retinanet_R_101_FPN_3x"),
        (faster_rcnn_R_50_FPN_3x, "faster_rcnn_R_50_FPN_3x"),
    ]

    for dfn in datasets:
        active_pairs = []
        for model, model_name in models:
            work_queue = Queue(num_cpu_workers * 3)
            results_queue = Queue()

            cpu_workers = [
                consumer.remote(work_queue, results_queue, confs)
                for _ in range(num_cpu_workers)
            ]
            gpu_workers = [
                producer.remote(
                    model,
                    model_name,
                    dfn,
                    work_queue,
                    BATCH_SIZE,
                )
            ]

            label = f"{model_name}-{str(dfn())}"
            active_pairs.append(
                {
                    "pair_label": label,
                    "cpu_workers": cpu_workers,
                    "gpu_workers": gpu_workers,
                    "work_queue": work_queue,
                    "results_queue": results_queue,
                }
            )

        all_gpu_workers = [
            gpu_worker for p in active_pairs for gpu_worker in p["gpu_workers"]
        ]
        ray.get(all_gpu_workers)

        all_cpu_workers = [
            cpu_worker for p in active_pairs for cpu_worker in p["cpu_workers"]
        ]
        ray.get(all_cpu_workers)
        for pair in active_pairs:
            result = aggregate_results(pair["results_queue"], num_cpu_workers)
            final_output = finalize_aggregator(result)
            with open(f"{pair['pair_label']}.json", "w") as f:
                json.dump(final_output, f, indent=2)
