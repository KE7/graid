"""
Database Generation Module for Object Detection Models

This module provides functionality to generate databases using various object detection
models from different backend families (Detectron, MMDetection, RT_DETR, YOLO).

The module supports:
- Multiple backend families: Detectron, MMDetection, RT_DETR, YOLO
- Multiple datasets: BDD100K, nuImages, Waymo
- Command line interface and programmatic function calls
- Configurable confidence thresholds and device settings
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from graid.data.Datasets import ObjDectDatasetBuilder
from graid.models.Detectron import Detectron_obj
from graid.models.MMDetection import MMdetection_obj
from graid.models.Ultralytics import RT_DETR, Yolo
from graid.utilities.common import (
    get_default_device,
    project_root_dir,
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)

# Dataset transforms (restored to original format)


def bdd_transform(i, l):
    return yolo_bdd_transform(i, l, new_shape=(768, 1280))


def nuimage_transform(i, l):
    return yolo_nuscene_transform(i, l, new_shape=(896, 1600))


def waymo_transform(i, l):
    return yolo_waymo_transform(i, l, (1280, 1920))


DATASET_TRANSFORMS = {
    "bdd": bdd_transform,
    "nuimage": nuimage_transform,
    "waymo": waymo_transform,
}

# GRAID supports any model from the supported backends
# Users can provide custom configurations for detectron and mmdetection
# or use any available model file for ultralytics


def create_model(
    backend: str,
    model_name: str,
    device: Optional[Union[str, torch.device]] = None,
    threshold: float = 0.2,
    custom_config: Optional[Dict[str, str]] = None,
):
    """
    Create a model instance based on backend and model name.

    Args:
        backend: Backend family ('detectron', 'mmdetection', 'ultralytics')
        model_name: Model name or path for ultralytics, or custom model identifier
        device: Device to use for inference
        threshold: Confidence threshold for detections
        custom_config: Custom configuration dict with backend-specific keys:
            - detectron: {'config': path, 'weights': path}
            - mmdetection: {'config': path, 'checkpoint': path}
            - ultralytics: ignored (model_name is the model file)

    Returns:
        Model instance implementing ObjectDetectionModelI

    Raises:
        ValueError: If backend is not supported or required config is missing
    """
    if device is None:
        device = get_default_device()

    if backend == "detectron":
        if custom_config is None:
            raise ValueError(
                f"Detectron backend requires custom_config with 'config' and 'weights' keys. "
                f"Example: {{'config': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml', 'weights': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'}}"
            )

        if "config" not in custom_config or "weights" not in custom_config:
            raise ValueError(
                f"Detectron custom_config must contain 'config' and 'weights' keys. "
                f"Got: {list(custom_config.keys())}"
            )

        config_file = custom_config["config"]
        weights_file = custom_config["weights"]

        model = Detectron_obj(
            config_file=config_file,
            weights_file=weights_file,
            threshold=threshold,
            device=device,
        )

    elif backend == "mmdetection":
        if custom_config is None:
            raise ValueError(
                f"MMDetection backend requires custom_config with 'config' and 'checkpoint' keys. "
                f"Example: {{'config': 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', 'checkpoint': 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'}}"
            )

        if "config" not in custom_config or "checkpoint" not in custom_config:
            raise ValueError(
                f"MMDetection custom_config must contain 'config' and 'checkpoint' keys. "
                f"Got: {list(custom_config.keys())}"
            )

        config_path = custom_config["config"]
        checkpoint = custom_config["checkpoint"]

        # Check if it's a custom model (absolute path) or pre-configured (relative path)
        if not Path(config_path).is_absolute():
            # Pre-configured model - use mmdetection installation path
            mmdet_path = project_root_dir() / "install" / "mmdetection"
            config_path = str(mmdet_path / config_path)

        model = MMdetection_obj(config_path, checkpoint, device=device)
        model.set_threshold(threshold)

    elif backend == "ultralytics":
        # For ultralytics, model_name is the model file path/name
        model_file = model_name

        if "rtdetr" in model_name.lower():
            model = RT_DETR(model_file)
        else:
            model = Yolo(model_file)

        model.set_threshold(threshold)
        model.to(device)

    else:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends: 'detectron', 'mmdetection', 'ultralytics'"
        )

    return model


def generate_db(
    dataset_name: str,
    split: str,
    conf: float = 0.2,
    backend: Optional[str] = None,
    model_name: Optional[str] = None,
    batch_size: int = 1,
    device: Optional[Union[str, torch.device]] = None,
) -> str:
    """
    Generate a database for object detection results.

    Args:
        dataset_name: Name of dataset ('bdd', 'nuimage', 'waymo')
        split: Dataset split ('train', 'val')
        conf: Confidence threshold for detections
        backend: Backend family ('detectron', 'mmdetection', 'ultralytics')
        model_name: Specific model name within backend
        batch_size: Batch size for processing
        device: Device to use for inference

    Returns:
        Database name that was created

    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name not in DATASET_TRANSFORMS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if device is None:
        device = get_default_device()

    # Create model if backend and model_name are provided
    model = None
    if backend and model_name:
        model = create_model(backend, model_name, device, conf)
        db_name = f"{dataset_name}_{split}_{conf}_{backend}_{model_name}"
    else:
        db_name = f"{dataset_name}_{split}_gt"

    transform = DATASET_TRANSFORMS[dataset_name]

    db_builder = ObjDectDatasetBuilder(
        split=split, dataset=dataset_name, db_name=db_name, transform=transform
    )

    if not db_builder.is_built():
        db_builder.build(model=model, batch_size=batch_size, conf=conf, device=device)

    return db_name


def list_available_models() -> Dict[str, List[str]]:
    """
    List supported backends and example models.

    Returns:
        Dictionary mapping backend names to example models or usage info
    """
    return {
        "detectron": [
            "Custom models via config file - provide config and weights paths"
        ],
        "mmdetection": [
            "Custom models via config file - provide config and checkpoint paths"
        ],
        "ultralytics": [
            "yolov8x.pt",
            "yolov10x.pt",
            "yolo11x.pt",
            "rtdetr-x.pt",
            "Any YOLOv8/YOLOv10/YOLOv11/RT-DETR model file",
        ],
    }


def main():
    """Command line interface for database generation."""
    parser = argparse.ArgumentParser(
        description="Generate object detection databases with various models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate ground truth database
  python generate_db.py --dataset bdd --split val
  
  # Generate with YOLO model
  python generate_db.py --dataset bdd --split val --backend ultralytics --model yolov8x --conf 0.3
  
  # Generate with Detectron model
  python generate_db.py --dataset nuimage --split train --backend detectron --model faster_rcnn_R_50_FPN_3x
  
  # Generate with MMDetection model
  python generate_db.py --dataset waymo --split val --backend mmdetection --model co_detr --conf 0.25
  
  # List available models
  python generate_db.py --list-models
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_TRANSFORMS.keys()),
        help="Dataset to use",
    )

    parser.add_argument(
        "--split", type=str, choices=["train", "val"], help="Dataset split to use"
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["detectron", "mmdetection", "ultralytics"],
        help="Model backend to use",
    )

    parser.add_argument(
        "--model", type=str, help="Specific model name within the backend"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold for detections (default: 0.2)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (e.g., 'cuda:0', 'cpu'). Auto-detected if not specified.",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models by backend",
    )

    args = parser.parse_args()

    if args.list_models:
        models = list_available_models()
        print("Available models by backend:")
        for backend, model_list in models.items():
            print(f"\n{backend.upper()}:")
            for model in model_list:
                print(f"  - {model}")
        return

    if not args.dataset or not args.split:
        parser.error("--dataset and --split are required unless using --list-models")

    if args.backend and not args.model:
        parser.error("--model is required when --backend is specified")

    if args.model and not args.backend:
        parser.error("--backend is required when --model is specified")

    # Note: Model validation is now done at runtime by trying to load the model
    # Users can provide any model name/path for their chosen backend

    try:
        db_name = generate_db(
            dataset_name=args.dataset,
            split=args.split,
            conf=args.conf,
            backend=args.backend,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(f"Successfully generated database: {db_name}")

    except Exception as e:
        print(f"Error generating database: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
