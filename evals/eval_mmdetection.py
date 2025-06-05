import argparse
import gc
import json
import logging
import time
from collections import defaultdict
from typing import List

import numpy as np
import ray
import torch
from ray.util.queue import Queue
from graid.data.ImageLoader import (
    Bdd100kDataset,
    ImageDataset,
    NuImagesDataset,
    WaymoDataset,
)
from graid.interfaces.ObjectDetectionI import (
    ObjectDetectionModelI,
    ObjectDetectionUtils,
)
from graid.models.Detectron import Detectron_obj
from graid.models.MMDetection import MMdetection_obj
from graid.models.Ultralytics import RT_DETR, Yolo
from graid.utilities.common import (
    project_root_dir,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

num_cpu_workers = 1  # must be one
BATCH_SIZE = 32
logger = logging.getLogger("ray")


MMDETECTION_PATH = project_root_dir() / "install" / "mmdetection"

# GDINO_config = str(
#     MMDETECTION_PATH
#     / "configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg.py"
# )
GDINO_config = str(
    MMDETECTION_PATH / "configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py"
)

# GDINO_checkpoint = str(
#     "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
# )

GDINO_checkpoint = str(
    "https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth"
)
# GDINO = MMdetection_obj(GDINO_config, GDINO_checkpoint) # 1.41 GB

Co_DETR_config = str(
    MMDETECTION_PATH
    / "projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py"
)
Co_DETR_checkpoint = str(
    "https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth"
)
# Co_DETR = MMdetection_obj(Co_DETR_config, Co_DETR_checkpoint) # 902 MB


@ray.remote(num_gpus=1)
def producer(
    model: ObjectDetectionModelI,
    model_name: str,
    dataset: ImageDataset,
    work_queue: Queue,
    batch_size: int,
    seed: int = 7,
):
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

    if model_name == "GDINO":
        # MMDetection models are not serializable, so we can't pass them in Ray
        model = MMdetection_obj(GDINO_config, GDINO_checkpoint)
    elif model_name == "Co_DETR":
        model = MMdetection_obj(Co_DETR_config, Co_DETR_checkpoint)

    print(f"[GPU Task] Starting inference: {dataset}")
    start_time = time.time()
    model.to("cuda:0")

    for i, batch in enumerate(tqdm(dataloader, desc="Processing " + str(dataset))):
        print(f"[GPU Task] Processing batch {i}")
        images = [sample["image"] for sample in batch]
        x = torch.stack(images)
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
        print(f"[GPU Task] Sent batch item to CPU worker: {model_name}")
        print(f"Current work queue size: {work_queue.qsize()}")

    end_time = time.time()
    print(
        f"[GPU Task] Finished inference: {dataset}, {model_name} in {end_time - start_time:.2f} seconds"
    )
    logger.info(
        f"[GPU Task] Finished inference: {dataset}, {model_name} in {end_time - start_time:.2f} seconds"
    )
    model.to("cpu")  # Move model back to CPU to free GPU memory
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
            f"[CPU Task] Received batch workable-item ({item != None}). Work queue remaining size: {work_queue.qsize()}"
        )
        if item is None:
            break

        print(
            f"[CPU Task] Processing batch item from: {item['dataset']}, {item['model']}"
        )
        dataset = item["dataset"]
        model_name = item["model"]
        assert (
            dataset == assigned_dataset
        ), f"Dataset mismatch: {dataset} != {assigned_dataset}"
        assert (
            model_name == assigned_model
        ), f"Model mismatch: {model_name} != {assigned_model}"
        gt_list = item["gt"]
        odrs_list = item["odrs"]
        images = item["images"]

        for conf in confs:
            for image, odrs, gt in zip(images, odrs_list, gt_list):
                # key = (dataset, model_name, conf)
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
                    extended_summary=False,
                    penalize_for_extra_predicitions=False,
                    image=image,
                )
                ObjectDetectionUtils.compute_metrics_for_single_img(
                    relevant_odrs,
                    gt,
                    metric=metrics_with_pen[conf],
                    class_metrics=True,
                    extended_summary=False,
                    penalize_for_extra_predicitions=True,
                    image=image,
                )

                true_negs[conf] += tn_no_pen["TN"]

        del images
        del gt_list
        del odrs_list
        del item
        # gc.collect()

    no_pen_scores = defaultdict(dict)
    pen_scores = defaultdict(dict)
    for conf in tqdm(confs, desc="Computing COCO metrics..."):
        no_pen_scores[conf] = metrics_no_pen[conf].compute()
        no_pen_scores[conf]["TN"] = true_negs[conf]
        pen_scores[conf] = metrics_with_pen[conf].compute()
        metrics_no_pen[conf].reset()
        metrics_with_pen[conf].reset()
        del metrics_no_pen[conf]
        del metrics_with_pen[conf]

    print(f"[CPU Task] Finished processing")

    final_results = dict()
    key = f"{assigned_model}-{assigned_dataset}"
    final_results[key] = {
        "metrics_no_pen": no_pen_scores,
        "metrics_pen": pen_scores,
    }
    results_queue.put(final_results)


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


def convert_for_json(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {
            (k if isinstance(k, str) else str(k)): convert_for_json(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


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
                    NuImagesDataset(
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
                WaymoDataset(
                    split=split,
                    transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920)),
                )
            )
    # BDD100k
    elif dataset == "bdd100k":
        for split in ["val"]:
            datasets.append(
                Bdd100kDataset(
                    split=split,
                    transform=lambda i, l: yolo_bdd_transform(
                        i, l, new_shape=(768, 1280)
                    ),
                    use_original_categories=False,
                    use_extended_annotations=False,
                )
            )

    confs = [float(c) for c in np.arange(0.10, 0.95, 0.10)]

    # 2) Prepare models
    models = [
        (None, "GDINO"),
        (None, "Co_DETR"),
        # (yolo_v10x, "yolo_v10x"),
        # (yolo_11x, "yolo_11x"),
        # (rtdetr, "rtdetr"),
        # (retinanet_R_101_FPN_3x, "retinanet_R_101_FPN_3x"),
        # (faster_rcnn_R_50_FPN_3x, "faster_rcnn_R_50_FPN_3x"),
    ]

    work_queues = dict()
    gpu_workers = []
    results_queue = Queue()

    for dataset in datasets:
        active_pairs = []
        for model, model_name in models:
            current_work_queue = Queue(num_cpu_workers * 10)
            work_queues[(model_name, str(dataset))] = current_work_queue

            cpu_workers = [
                consumer.remote(
                    current_work_queue, results_queue, confs, str(dataset), model_name
                )
                for _ in range(num_cpu_workers)
            ]

            label = f"{model_name}-{str(dataset)}"
            active_pairs.append(
                {
                    "pair_label": label,
                    "cpu_workers": cpu_workers,
                    # "gpu_workers": gpu_workers,
                    # "work_queue": work_queue,
                    # "results_queue": results_queue,
                }
            )

            gpu_workers.append(
                producer.remote(
                    model,
                    model_name,
                    dataset,
                    current_work_queue,
                    BATCH_SIZE,
                )
            )

        ray.get(gpu_workers)
        gc.collect()

        all_cpu_workers = [
            cpu_worker for p in active_pairs for cpu_worker in p["cpu_workers"]
        ]
        ray.get(all_cpu_workers)
        print("All CPU workers finished.")

        for pair in active_pairs:
            result = results_queue.get()
            with open(f"{pair['pair_label']}.json", "w") as f:
                json.dump(convert_for_json(result), f, indent=2)


if __name__ == "__main__":
    main()
