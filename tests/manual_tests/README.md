# Manual Tests

This directory contains **integration and model tests** that require human intervention to run properly.

## ⚠️ Human Intervention Required

These tests typically require:
- **Real model weights** downloaded and available
- **Large datasets** (BDD, NuImages, Waymo) properly configured
- **GPU resources** for model inference
- **Manual setup** of paths, credentials, or environment variables
- **Human verification** of outputs and results

## Test Files

- **`test_detectron_obj.py`** - Tests Detectron2 object detection models
- **`test_dino_obj.py`** - Tests DINO object detection models  
- **`test_mmdetection_obj.py`** - Tests MMDetection object detection models
- **`test_mmdetection_seg.py`** - Tests MMDetection segmentation models
- **`test_ultralytics_obj.py`** - Tests Ultralytics (YOLO) object detection models
- **`test_ultralytics_seg.py`** - Tests Ultralytics segmentation models
- **`test_questions.py`** - Tests question-answering functionality

## Running Manual Tests

```bash
# These tests should be run individually with proper setup
# Example:
python -m pytest tests/manual_tests/test_detectron_obj.py -v -s

# Or run directly:
python tests/manual_tests/test_detectron_obj.py
```

## Before Running

1. Ensure all required model weights are downloaded
2. Verify dataset paths are correctly configured
3. Check that GPU/CUDA is available if required
4. Review test parameters and adjust as needed
5. Be prepared to manually verify outputs

## Purpose

These tests validate:
- End-to-end model functionality
- Real dataset integration
- Performance characteristics  
- Model accuracy and outputs

They complement the automated unit tests by testing the full system with real data and models. 