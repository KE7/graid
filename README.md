# GRAID: Generating Reasoning questions from Analysis of Images via Discriminative artificial intelligence

[Design Doc](https://docs.google.com/document/d/1zgb1odK3zfwLg2zKts2eC1uQcQfUd6q_kKeMzd1q-m4/edit?tab=t.0)

## 🚀 Quick Start

### Installation
1. Create a virtual environment: `uv venv`
2. Activate it: `source .venv/bin/activate` (or use direnv with the provided .envrc)
3. Install dependencies: `uv sync`
4. Install all backends: `uv run install_all`

### Using GRAID CLI

**Interactive Mode (Recommended):**
```bash
# Using conda environment
/work/ke/miniconda3/envs/scenic_reason/bin/python scenic_reasoning/src/scenic_reasoning/graid_cli.py generate

# Using uv (after installation)
uv run graid generate
```

**Non-Interactive Mode:**
```bash
# Generate ground truth database
uv run graid generate --dataset bdd --split val --interactive false

# Use pre-configured model
uv run graid generate --dataset nuimage --split train --backend ultralytics --model yolov8x --conf 0.3 --interactive false
```

**Available Commands:**
```bash
uv run graid --help          # Show help
uv run graid list-models     # List available models
uv run graid info           # Show project information
```

## Status

### Backends

|                       | Ultralytics | Detectron | MMDetection |
|-----------------------|-------------|-----------|-------------|
| Object Detection      | ✅           | ✅        | ✅          |
| Instance Segmentation | ✅           | ✅        | ✅          |

### Datasets

|                       | BDD100K     | Waymo     | NuImages    |
|-----------------------|-------------|-----------|-------------|
| Object Detection      | ✅           | ✅        | ✅          |
| Instance Segmentation | ✅           | ✅        | ✅          |

## 🧠 Supported Models

**Detectron2:** `retinanet_R_101_FPN_3x`, `faster_rcnn_R_50_FPN_3x`  
**MMDetection:** `co_detr`, `dino`  
**Ultralytics:** `yolov8x`, `yolov10x`, `yolo11x`, `rtdetr-x`

## ✨ GRAID Features

- **Interactive CLI**: User-friendly prompts for dataset and model selection
- **Multiple Backends**: Support for Detectron2, MMDetection, and Ultralytics
- **Custom Models**: Bring your own model configurations
- **Ground Truth Support**: Generate databases using original annotations
- **Batch Processing**: Support for non-interactive scripted usage

## 📁 Project Structure

The project has been renamed from `scenic-reasoning` to **GRAID**. Key components:

- **Package**: `scenic_reasoning/src/graid/` (new GRAID package)
- **CLI**: `scenic_reasoning/src/scenic_reasoning/graid_cli.py`
- **Original**: `scenic_reasoning/src/scenic_reasoning/` (backward compatibility)

## 📊 Databases

Generated databases are saved in:
```
data/databases_ablations/{dataset}_{split}_{conf}_{backend}_{model}.sqlite
```

✅ **Ready to use!**
