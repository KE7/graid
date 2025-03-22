import argparse
import gc
import json
import logging
import time
import tracemalloc
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
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

allowed_gpus = [0, 1, 2, 3, 5, 6]
num_cpu_workers = 1 # must be one 
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
    seed: int = 7,
):
    dataset = dataset_fn()

    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(dataset, generator=gen)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        generator=gen,
        sampler=sampler,
    )

    print(f"[GPU Task] Starting inference: {dataset}")
    start_time = time.time()
    model.to("cuda:0")
    # # print all available gpus
    # print(f"Available GPUs: {torch.cuda.device_count()}")
    # for i, (model, model_name) in enumerate(models):
    #     model.to("cuda:" + str(i))

    for i, batch in enumerate(
        tqdm(dataloader, desc="Processing " + str(dataset), total=100)
    ):
        print(f"[GPU Task] Processing batch {i}")
        # if i >= 100:
        #     print(f"[GPU Task] Finished processing {i} batches")
        #     break
        images = [sample["image"] for sample in batch]
        # x = torch.stack(images).to(get_default_device())
        x = torch.stack(images)
        y = [sample["labels"] for sample in batch]

        # for i, (model, model_name) in enumerate(models):
        #     x = x.to("cuda:" + str(i))
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
        print(f"[GPU Task] Sent batch item to CPU worker: {model_name}")
        print(f"Current work queue size: {work_queue.qsize()}")

    end_time = time.time()
    print(
        f"[GPU Task] Finished inference: {dataset}, {model_name} in {end_time - start_time:.2f} seconds"
    )
    logger.info(
        f"[GPU Task] Finished inference: {dataset}, {model_name} in {end_time - start_time:.2f} seconds"
    )
    # for work_queue in work_queues:
    for _ in range(num_cpu_workers * 3):
        work_queue.put(None)  # Signal completion

    torch.cuda.empty_cache()
    del dataset
    del dataloader
    gc.collect()


@ray.remote(num_cpus=1)
def consumer(
    work_queue: Queue,
    results_queue: Queue,
    confs: List[float],
    assigned_dataset: str,
    assigned_model: str,
):
    tracemalloc.start()
    # metric = MeanAveragePrecision(
    #     class_metrics=True,
    #     extended_summary=True,
    #     box_format="xyxy",
    #     iou_type="bbox",
    # )
    metrics_with_pen = dict()
    metrics_no_pen = dict()
    for conf in confs:
        metrics_no_pen[conf] = MeanAveragePrecision(
            class_metrics=True,
            extended_summary=True,
            box_format="xyxy",
            iou_type="bbox",
        )
        metrics_with_pen[conf] = MeanAveragePrecision(
            class_metrics=True,
            extended_summary=True,
            box_format="xyxy",
            iou_type="bbox",
        )
    true_negs = defaultdict(int)
    while True:
        print(
            f"[CPU Task] Waiting for batch item. Work queue remaining size: {work_queue.qsize()}"
        )
        item = work_queue.get()
        print(
            f"[CPU Task] Received batch item. Work queue remaining size: {work_queue.qsize()}"
        )
        if item is None:
            break

        print(
            f"[CPU Task] Processing batch item from: {item['dataset']}, {item['model']}"
        )
        dataset = item["dataset"]
        model_name = item["model"]
        assert dataset == assigned_dataset, f"Dataset mismatch: {dataset} != {assigned_dataset}"
        assert model_name == assigned_model, f"Model mismatch: {model_name} != {assigned_model}"
        gt_list = item["gt"]
        odrs_list = item["odrs"]
        images = item["images"]

        for conf in confs:
            for image, odrs, gt in zip(images, odrs_list, gt_list):
                key = (dataset, model_name, conf)
                index = len(odrs)
                for i, o in enumerate(odrs):
                    if o.score < conf:
                        index = i
                        break
                relevant_odrs = odrs[:index]

                tn_no_pen = ObjectDetectionUtils.compute_metrics_for_single_img(
                    relevant_odrs,
                    gt,
                    metric=metrics_no_pen[conf],
                    class_metrics=True,
                    extended_summary=True,
                    penalize_for_extra_predicitions=False,
                    image=image,
                )
                ObjectDetectionUtils.compute_metrics_for_single_img(
                    relevant_odrs,
                    gt,
                    metric=metrics_with_pen[conf],
                    class_metrics=True,
                    extended_summary=True,
                    penalize_for_extra_predicitions=True,
                    image=image,
                )

                true_negs[conf] += tn_no_pen["TN"]
                # if key not in results:
                #     results[key] = {
                #         "metrics_no_pen": [],
                #         "metrics_pen": [],
                #     }
                # results[key]["metrics_no_pen"].append(metrics_no_pen)
                # results[key]["metrics_pen"].append(metrics_pen)

        del images
        del gt_list
        del odrs_list
        del item
        # gc.collect()

    snapshot = tracemalloc.take_snapshot()
    # import sys
    # for k, data in results.items():
    #     no_pen_list_size = sys.getsizeof(data["metrics_no_pen"])
    #     pen_list_size = sys.getsizeof(data["metrics_pen"])
    #     print(f"{k}: metrics_no_pen size={no_pen_list_size/1024/1024:.2f} MB, metrics_pen size={pen_list_size/1024/1024:.2f} MB")

    # here is where we would call metric.compute
    
    # results_queue.put(
    #     results
    # )
    no_pen_scores = defaultdict(dict)
    pen_scores = defaultdict(dict)
    for conf in confs:
        no_pen_scores[conf] = metrics_no_pen[conf].compute()
        no_pen_scores[conf]["TN"] = true_negs[conf]     
        pen_scores[conf] = metrics_with_pen[conf].compute()
        pen_scores[conf]["TN"] = true_negs[conf]
        metrics_no_pen[conf].reset()
        metrics_with_pen[conf].reset()
        del metrics_no_pen[conf]
        del metrics_with_pen[conf]

    print(f"[CPU Task] Finished processing")

    top_stats = snapshot.statistics('lineno')
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

    final_results = dict()
    final_results[(assigned_dataset, assigned_model)] = {
        "metrics_no_pen": no_pen_scores,
        "metrics_pen": pen_scores,
    }
    results_queue.put(final_results)
    results_queue.put(None)  # Signal completion


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

    # from pprint import pprint
    # print("Finalizing results: ", final_output)
    # pprint(f"Finalizing done: {final_output}")
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


def main():
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
        for split in ["val"]:
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
        for split in ["validation"]:
            datasets.append(
                lambda split=split: WaymoDataset(
                    split=split,
                    transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920)),
                )
            )
    # BDD100k
    elif dataset == "bdd100k":
        for split in ["val"]:
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
        (yolo_v10x, "yolo_v10x"),
        (yolo_11x, "yolo_11x"),
        (rtdetr, "rtdetr"),
        (retinanet_R_101_FPN_3x, "retinanet_R_101_FPN_3x"),
        (faster_rcnn_R_50_FPN_3x, "faster_rcnn_R_50_FPN_3x"),
    ]

    # final_output = dict()
    work_queues = dict()
    gpu_workers = []

    for dfn in datasets:
        active_pairs = []

        # work_queues = [Queue(num_cpu_workers * 3) for _ in range(len(models))]
        # work_queue = Queue(num_cpu_workers * 3 * len(models))
        results_queue = Queue()

        for model, model_name in models:
            # work_queue = Queue(num_cpu_workers*3)
            # results_queue = Queue()
            current_work_queue = Queue(num_cpu_workers * 10)
            work_queues[(model_name, str(dfn()))] = current_work_queue

            cpu_workers = [
                consumer.remote(current_work_queue, results_queue, confs, str(dfn()), model_name)
                for _ in range(num_cpu_workers)
            ]

            label = f"{model_name}-{str(dfn())}"
            active_pairs.append(
                {
                    "pair_label": label,
                    "cpu_workers": cpu_workers,
                    # "gpu_workers": gpu_workers,
                    # "work_queue": work_queue,
                    "results_queue": results_queue,
                }
            )

            gpu_workers.append(
                producer.remote(
                    model,
                    model_name,
                    dfn,
                    current_work_queue,
                    BATCH_SIZE,
                )
            )

        # gpu_workers = [
        #     producer.remote(
        #         models,
        #         dfn,
        #         work_queue,
        #         BATCH_SIZE,
        #     )
        # ]

        ray.get(gpu_workers)
        # all_gpu_workers = [gpu_worker for p in active_pairs for gpu_worker in p["gpu_workers"]]
        # ray.get(all_gpu_workers)
        gc.collect()

        all_cpu_workers = [
            cpu_worker for p in active_pairs for cpu_worker in p["cpu_workers"]
        ]
        ray.get(all_cpu_workers)
        gc.collect()
        for pair in active_pairs:
            # result = aggregate_results(pair["results_queue"], num_cpu_workers)
            # final_output = finalize_aggregator(result)
            result = pair["results_queue"].get()
            with open(f"{pair['pair_label']}.json", "w") as f:
                json.dump(result, f, indent=2)

        del dfn
        gc.collect()

    # for k, v in final_output.items():
    #     with open(f"{k}.json", "w") as f:
    #         json.dump(v, f, indent=2)


main()
if __name__ == "__main__":
    main()
