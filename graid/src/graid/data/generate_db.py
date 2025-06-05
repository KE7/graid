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
bdd_transform = lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280))
nuimage_transform = lambda i, l: yolo_nuscene_transform(i, l, new_shape=(896, 1600))
waymo_transform = lambda i, l: yolo_waymo_transform(i, l, (1280, 1920))

DATASET_TRANSFORMS = {
    "bdd": bdd_transform,
    "nuimage": nuimage_transform,
    "waymo": waymo_transform,
}

# Model configurations for different backends
MODEL_CONFIGS = {
    "detectron": {
        "retinanet_R_101_FPN_3x": {
            "config": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
            "weights": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
        },
        "faster_rcnn_R_50_FPN_3x": {
            "config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            "weights": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        },
    },
    "mmdetection": {
        "co_detr": {
            "config": "projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py",
            "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth",
        },
        "dino": {
            "config": "configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py",
            "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth",
        },
    },
    "ultralytics": {
        "yolov8x": "yolov8x.pt",
        "yolov10x": "yolov10x.pt", 
        "yolo11x": "yolo11x.pt",
        "rtdetr-x": "rtdetr-x.pt",
    },
}


def create_model(
    backend: str,
    model_name: str,
    device: Optional[Union[str, torch.device]] = None,
    threshold: float = 0.2,
):
    """
    Create a model instance based on backend and model name.
    
    Args:
        backend: Backend family ('detectron', 'mmdetection', 'ultralytics')
        model_name: Specific model name within the backend
        device: Device to use for inference
        threshold: Confidence threshold for detections
        
    Returns:
        Model instance implementing ObjectDetectionModelI
        
    Raises:
        ValueError: If backend or model_name is not supported
    """
    if device is None:
        device = get_default_device()
    
    if backend == "detectron":
        if model_name not in MODEL_CONFIGS["detectron"]:
            raise ValueError(f"Unsupported Detectron model: {model_name}")
        
        config_info = MODEL_CONFIGS["detectron"][model_name]
        
        # Handle both pre-configured and custom models
        if "config" in config_info and "weights" in config_info:
            # Custom model format
            config_file = config_info["config"]
            weights_file = config_info["weights"]
        else:
            # Pre-configured model format (backward compatibility)
            config_file = config_info["config"]
            weights_file = config_info["weights"]
        
        model = Detectron_obj(
            config_file=config_file,
            weights_file=weights_file,
            threshold=threshold,
            device=device,
        )
        
    elif backend == "mmdetection":
        if model_name not in MODEL_CONFIGS["mmdetection"]:
            raise ValueError(f"Unsupported MMDetection model: {model_name}")
        
        config_info = MODEL_CONFIGS["mmdetection"][model_name]
        
        # Handle both pre-configured and custom models
        if "config" in config_info and "checkpoint" in config_info:
            # Check if it's a custom model (absolute path) or pre-configured (relative path)
            config_path = config_info["config"]
            if not Path(config_path).is_absolute():
                # Pre-configured model - use mmdetection installation path
                mmdet_path = project_root_dir() / "install" / "mmdetection"
                config_path = str(mmdet_path / config_path)
            
            checkpoint = config_info["checkpoint"]
        else:
            raise ValueError(f"Invalid MMDetection model configuration for {model_name}")
        
        model = MMdetection_obj(config_path, checkpoint, device=device)
        model.set_threshold(threshold)
        
    elif backend == "ultralytics":
        if model_name not in MODEL_CONFIGS["ultralytics"]:
            raise ValueError(f"Unsupported Ultralytics model: {model_name}")
        
        model_file = MODEL_CONFIGS["ultralytics"][model_name]
        
        if "rtdetr" in model_name:
            model = RT_DETR(model_file)
        else:
            model = Yolo(model_file)
            
        model.set_threshold(threshold)
        model.to(device)
        
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
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
        split=split,
        dataset=dataset_name,
        db_name=db_name,
        transform=transform
    )
    
    if not db_builder.is_built():
        db_builder.build(
            model=model,
            batch_size=batch_size,
            conf=conf,
            device=device
        )
    
    return db_name


def list_available_models() -> Dict[str, List[str]]:
    """
    List all available models by backend.
    
    Returns:
        Dictionary mapping backend names to lists of available models
    """
    return {
        backend: list(models.keys()) 
        for backend, models in MODEL_CONFIGS.items()
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
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_TRANSFORMS.keys()),
        help="Dataset to use"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        help="Dataset split to use"
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model backend to use"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model name within the backend"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold for detections (default: 0.2)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (e.g., 'cuda:0', 'cpu'). Auto-detected if not specified."
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models by backend"
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
    
    # Validate model exists in backend
    if args.backend and args.model:
        if args.model not in MODEL_CONFIGS[args.backend]:
            parser.error(f"Model '{args.model}' not available for backend '{args.backend}'")
    
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
