# Database Generation Module

This module provides functionality to generate databases using various object detection models from different backend families. The generated databases can be used for evaluation, comparison, and analysis of object detection models across different datasets.

## Features

- **Multiple Backend Support**: Detectron2, MMDetection, and Ultralytics (YOLO/RT-DETR)
- **Dataset Support**: BDD100K, nuImages, and Waymo
- **Flexible Interface**: Both command-line and programmatic API
- **Configurable Parameters**: Confidence thresholds, batch sizes, and device selection
- **Ground Truth Generation**: Support for generating ground truth databases

## Supported Models

### Detectron2 Backend
- `retinanet_R_101_FPN_3x`: RetinaNet with ResNet-101 backbone
- `faster_rcnn_R_50_FPN_3x`: Faster R-CNN with ResNet-50 backbone

### MMDetection Backend
- `co_detr`: CO-DETR with Swin-L backbone
- `dino`: DINO with Swin-L backbone

### Ultralytics Backend
- `yolov8x`: YOLOv8 Extra Large
- `yolov10x`: YOLOv10 Extra Large
- `yolo11x`: YOLO11 Extra Large
- `rtdetr-x`: RT-DETR Extra Large

## Command Line Usage

### Basic Usage

Generate a ground truth database:
```bash
python generate_db.py --dataset bdd --split val
```

Generate with a specific model:
```bash
python generate_db.py --dataset bdd --split val --backend ultralytics --model yolov8x --conf 0.3
```

List available models:
```bash
python generate_db.py --list-models
```

### Command Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--dataset` | str | Dataset to use (bdd, nuimage, waymo) |
| `--split` | str | Dataset split (train, val) |
| `--backend` | str | Model backend (detectron, mmdetection, ultralytics) |
| `--model` | str | Specific model within backend |
| `--conf` | float | Confidence threshold (default: 0.2) |
| `--batch-size` | int | Batch size (default: 1) |
| `--device` | str | Device (cuda:N, cpu) |
| `--list-models` | flag | List all available models |

## Programmatic Usage

```python
from scenic_reasoning.data.generate_db import generate_db, list_available_models

# Generate ground truth database
db_name = generate_db("bdd", "val")

# Generate with YOLO model
db_name = generate_db(
    dataset_name="bdd",
    split="val", 
    backend="ultralytics",
    model_name="yolov8x",
    conf=0.3
)

# List available models
models = list_available_models()
```

## Function Reference

### generate_db()
Generate a database for object detection results.

Parameters:
- dataset_name (str): Name of dataset ('bdd', 'nuimage', 'waymo')
- split (str): Dataset split ('train', 'val')
- conf (float, optional): Confidence threshold (default: 0.2)
- backend (str, optional): Backend family
- model_name (str, optional): Specific model name
- batch_size (int, optional): Batch size (default: 1)
- device (str/torch.device, optional): Device to use

Returns: Database name (str)

### create_model()
Create a model instance based on backend and model name.

### list_available_models()
List all available models by backend.

Returns: Dict mapping backend names to model lists

## Examples

### Command Line Examples

```bash
# Ground truth
python generate_db.py --dataset bdd --split val

# YOLO model
python generate_db.py --dataset bdd --split val --backend ultralytics --model yolov8x --conf 0.3

# Detectron model
python generate_db.py --dataset nuimage --split train --backend detectron --model faster_rcnn_R_50_FPN_3x

# MMDetection model
python generate_db.py --dataset waymo --split val --backend mmdetection --model co_detr --conf 0.25
```

### Python Examples

```python
# Multiple models
models_to_test = [
    ("ultralytics", "yolov8x"),
    ("detectron", "faster_rcnn_R_50_FPN_3x"),
    ("mmdetection", "co_detr"),
]

for backend, model_name in models_to_test:
    db_name = generate_db(
        dataset_name="bdd",
        split="val",
        backend=backend,
        model_name=model_name,
        conf=0.25
    )
    print(f"Generated {db_name}")
```

## Output

Generated databases follow this naming convention:
- Ground truth: `{dataset}_{split}_gt`
- Model predictions: `{dataset}_{split}_{conf}_{backend}_{model_name}`

Examples:
- `bdd_val_gt`
- `bdd_val_0.2_ultralytics_yolov8x`
- `nuimage_train_0.25_mmdetection_co_detr` 