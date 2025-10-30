## GRAID: <u>G</u>enerating <u>R</u>easoning questions from <u>A</u>nalysis of <u>I</u>mages via <u>D</u>iscriminative artificial intelligence

[Design Doc](https://docs.google.com/document/d/1zgb1odK3zfwLg2zKts2eC1uQcQfUd6q_kKeMzd1q-m4/edit?tab=t.0)

## 🚀 Quick Start

### Installation
0. Install uv (optional if you already have it): `curl -LsSf https://astral.sh/uv/install.sh | sh` (or see [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/))
1. Create a virtual environment: `uv venv`
2. Activate it: `source .venv/bin/activate` (or use direnv with the provided .envrc)
3. Install dependencies: `uv sync`
4. Install all backends: `uv run install_all`

### 🤗 HuggingFace Dataset Generation

**Generate high-quality VQA datasets for modern ML workflows:**
```bash
# Interactive mode with step-by-step guidance
graid generate-dataset
```

**Key Features:**
- **🎯 Object Filtering**: Smart allowable sets for focused object detection
- **🔬 Multi-Model Ensemble**: Weighted Boxes Fusion (WBF) for improved accuracy  
- **⚙️ Flexible Configuration**: JSON configs for reproducible experiments
- **🌐 HuggingFace Hub Integration**: Direct upload to share datasets
- **🖼️ PIL Image Support**: Ready for modern vision-language models
- **📊 Rich Metadata**: Comprehensive dataset documentation

**Quick Examples:**
```bash
# Generate with specific object types (autonomous driving focus)
uv run graid generate-dataset --allowable-set "person,car,truck,bicycle,traffic light"

# Multi-model ensemble for enhanced accuracy
uv run graid generate-dataset --config examples/wbf_ensemble.json

# Upload directly to HuggingFace Hub
uv run graid generate-dataset --upload-to-hub --hub-repo-id "your-org/dataset-name"

# List all valid COCO objects
uv run graid generate-dataset --list-objects
```

### 🎛️ Configuration-Driven Workflows

**Create reusable configurations for systematic experiments:**

**Basic Configuration:**
```json
{
  "dataset_name": "bdd",
  "split": "val", 
  "models": [
    {
      "backend": "detectron",
      "model_name": "faster_rcnn_R_50_FPN_3x",
      "confidence_threshold": 0.7
    },
    {
      "backend": "mmdetection", 
      "model_name": "co_detr",
      "confidence_threshold": 0.6
    }
  ],
  "use_wbf": true,
  "wbf_config": {
    "iou_threshold": 0.6,
    "model_weights": [1.0, 1.2]
  },
  "allowable_set": ["person", "car", "truck", "bus", "motorcycle", "bicycle"],
  "confidence_threshold": 0.5,
  "batch_size": 4
}
```

**Advanced Configuration with Custom Questions and Transforms:**
```json
{
  "dataset_name": "bdd",
  "split": "val",
  "models": [
    {
      "backend": "ultralytics",
      "model_name": "yolov8x.pt",
      "confidence_threshold": 0.6
    }
  ],
  "use_wbf": false,
  "allowable_set": ["person", "car", "bicycle", "motorcycle", "traffic light"],
  "confidence_threshold": 0.5,
  "batch_size": 2,
  
  "questions": [
    {
      "name": "HowMany",
      "params": {}
    },
    {
      "name": "Quadrants", 
      "params": {
        "N": 3,
        "M": 3
      }
    },
    {
      "name": "WidthVsHeight",
      "params": {
        "threshold": 0.4
      }
    },
    {
      "name": "LargestAppearance",
      "params": {
        "threshold": 0.35
      }
    },
    {
      "name": "MostClusteredObjects",
      "params": {
        "threshold": 80
      }
    }
  ],
  
  "transforms": {
    "type": "yolo_bdd",
    "new_shape": [640, 640]
  },
  
  "save_path": "./datasets/custom_bdd_vqa",
  "upload_to_hub": true,
  "hub_repo_id": "your-org/bdd-reasoning-dataset",
  "hub_private": false
}
```

**Custom Model Configuration:**
```json
{
  "dataset_name": "custom",
  "split": "train",
  "models": [
    {
      "backend": "detectron",
      "model_name": "custom_retinanet",
      "custom_config": {
        "config": "path/to/config.yaml", 
        "weights": "path/to/model.pth"
      }
    },
    {
      "backend": "ultralytics",
      "model_name": "custom_yolo",
      "custom_config": {
        "model_path": "path/to/custom_yolo.pt"
      }
    }
  ],
  "transforms": {
    "type": "yolo_bdd",
    "new_shape": [832, 832]
  },
  "questions": [
    {
      "name": "IsObjectCentered",
      "params": {}
    },
    {
      "name": "LeftOf", 
      "params": {}
    },
    {
      "name": "RightOf",
      "params": {}
    }
  ]
}
```

### 📦 Custom Dataset Support

**Bring Your Own Data**: GRAID supports any PyTorch-compatible dataset:

```python
from graid.data.generate_dataset import generate_dataset
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """Your custom dataset implementation"""
    def __getitem__(self, idx):
        # Return: (image_tensor, optional_annotations, metadata)
        # Annotations are only needed for mAP/mAR evaluation
        # For VQA generation, only images are required
        pass

# Generate HuggingFace dataset from your data
dataset = generate_dataset(
    dataset_name="custom",
    split="train",
    models=your_models,
    allowable_set=["person", "vehicle"], 
    save_path="./datasets/custom_vqa"
)
```

**Key Point**: Custom datasets only require images for VQA generation. Annotations are optional and only needed if you want to evaluate model performance with mAP/mAR metrics.

## 🔧 Advanced Features

### **Multi-Model Ensemble with WBF**
Combine predictions from multiple models using Weighted Boxes Fusion for enhanced detection accuracy:
- Improved precision through model consensus
- Configurable fusion parameters and model weights
- Supports mixed backends (Detectron2 + MMDetection + Ultralytics)

### **Intelligent Object Filtering**
Focus datasets on specific object categories:
- **Common presets**: Autonomous driving, indoor scenes, animals
- **Interactive selection**: Visual picker from 80 COCO categories
- **Manual specification**: Comma-separated object lists
- **Validation**: Automatic checking against COCO standard

### **Production-Ready Outputs**
Generated datasets include:
- **PIL Images**: Direct compatibility with vision-language models
- **Rich Annotations**: Bounding boxes, confidence scores, object classes
- **Structured QA Pairs**: Question templates with precise answers
- **Comprehensive Metadata**: Model info, generation parameters, statistics

## 📊 Supported Models & Datasets

### Backends

|                       | Detectron2  | MMDetection | Ultralytics |
|-----------------------|-------------|-------------|-------------|
| Object Detection      | ✅           | ✅          | ✅          |
| Instance Segmentation | ✅           | ✅          | ✅          |
| WBF Ensemble         | ✅           | ✅          | ✅          |

### Built-in Datasets

|                       | BDD100K     | NuImages    | Waymo       |
|-----------------------|-------------|-------------|-------------|
| Object Detection      | ✅           | ✅          | ✅          |
| Instance Segmentation | ✅           | ✅          | ✅          |
| HuggingFace Export    | ✅           | ✅          | ✅          |

### Example Models

**Detectron2:** `faster_rcnn_R_50_FPN_3x`, `retinanet_R_101_FPN_3x`  
**MMDetection:** `co_detr`, `dino`, `rtmdet`  
**Ultralytics:** `yolov8x`, `yolov10x`, `yolo11x`, `rtdetr-x`

## 🎯 Research Applications

This framework enables systematic evaluation of:
- **Vision-Language Models**: Generate targeted VQA benchmarks
- **Object Detection Methods**: Compare model performance on specific object types  
- **Reasoning Capabilities**: Create challenging spatial and counting questions
- **Domain Adaptation**: Generate domain-specific evaluation sets
- **Ensemble Methods**: Evaluate fusion strategies across detection models

## 📈 Quality Assurance

Generated datasets undergo comprehensive validation:
- **Model Verification**: Automatic testing of model loading and inference
- **Annotation Quality**: Confidence score filtering and duplicate removal
- **Metadata Integrity**: Complete provenance tracking for reproducibility
- **Format Compliance**: COCO-standard annotations with HuggingFace compatibility

## 🔍 Example commands

**Interactive CLI**: User-friendly prompts for dataset and model selection
```bash
uv run graid generate
```

**Available Commands:**
```bash
uv run graid --help              # Show help
uv run graid list-models         # List available models  
uv run graid list-questions      # List available question types with parameters
uv run graid info                # Show project information
uv run graid generate-dataset    # Modern HuggingFace generation

# Interactive features
uv run graid generate-dataset --interactive-questions  # Select questions interactively
uv run graid generate-dataset --list-questions         # Show available questions
```

## 📄 License

GRAID is open source software licensed under the [Apache License 2.0](LICENSE). This applies to both the GRAID framework code and any datasets generated using GRAID.

**Important**: When using GRAID with source datasets (BDD100K, Waymo, nuImages, etc.), you must also comply with the original source dataset license terms.
