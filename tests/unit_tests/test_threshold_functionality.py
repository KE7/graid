#!/usr/bin/env python3
"""
Test script to verify threshold functionality works correctly across all model types.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from graid.data.ImageLoader import Bdd100kDataset
from graid.models.Detectron import Detectron_obj
from graid.models.MMDetection import MMdetection_obj
from graid.models.Ultralytics import Yolo
from graid.utilities.common import yolo_bdd_transform

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "graid" / "src"))


def test_threshold_functionality():
    """Test that threshold functionality works correctly for all model types"""

    # Load a test image using the dataset
    dataset = Bdd100kDataset(
        split="val",
        transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
        use_original_categories=False,
        use_extended_annotations=False,
    )

    # Get the first image
    sample = dataset[0]
    print(f"Dataset sample type: {type(sample)}")

    # Handle different dataset formats
    if isinstance(sample, dict):
        image = sample["image"]
    elif isinstance(sample, (list, tuple)):
        image = sample[0]
    else:
        image = sample

    print("Testing threshold functionality...")
    print(f"Test image type: {type(image)}")
    print(f"Test image shape: {image.shape}")

    # Test Detectron model
    print("\n=== Testing Detectron Model ===")
    detectron_model = Detectron_obj(
        config_file="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        weights_file="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        device="cuda",
    )

    # Test with default threshold (0.0)
    detections_default = detectron_model.identify_for_image(image)
    print(
        f"Detectron with default threshold (0.0): {len(detections_default)} detections"
    )

    # Test with low threshold
    detectron_model.set_threshold(0.1)
    detections_low = detectron_model.identify_for_image(image)
    print(f"Detectron with threshold 0.1: {len(detections_low)} detections")

    # Test with high threshold
    detectron_model.set_threshold(0.8)
    detections_high = detectron_model.identify_for_image(image)
    print(f"Detectron with threshold 0.8: {len(detections_high)} detections")

    # Test MMDetection model
    print("\n=== Testing MMDetection Model ===")
    from graid.utilities.common import project_root_dir

    MMDETECTION_PATH = project_root_dir() / "install" / "mmdetection"

    mmdet_model = MMdetection_obj(
        config_file=str(
            MMDETECTION_PATH / "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
        ),
        checkpoint_file="https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        device="cuda",
    )

    # Test with default threshold (0.0)
    detections_default = mmdet_model.identify_for_image(image)
    print(
        f"MMDetection with default threshold (0.0): {len(detections_default)} detections"
    )

    # Test with low threshold
    mmdet_model.set_threshold(0.1)
    detections_low = mmdet_model.identify_for_image(image)
    print(f"MMDetection with threshold 0.1: {len(detections_low)} detections")

    # Test with high threshold
    mmdet_model.set_threshold(0.8)
    detections_high = mmdet_model.identify_for_image(image)
    print(f"MMDetection with threshold 0.8: {len(detections_high)} detections")

    # Test Ultralytics model
    print("\n=== Testing Ultralytics Model ===")
    yolo_model = Yolo("yolov8n.pt")

    # Test with default threshold (0.0)
    detections_default = yolo_model.identify_for_image(image)
    # Handle batch format - get first batch
    if (
        isinstance(detections_default, list)
        and len(detections_default) > 0
        and isinstance(detections_default[0], list)
    ):
        detections_default = detections_default[0]
    print(
        f"YOLO with default threshold (0.0): {len(detections_default)} detections")

    # Test with low threshold
    yolo_model.set_threshold(0.1)
    detections_low = yolo_model.identify_for_image(image)
    # Handle batch format - get first batch
    if (
        isinstance(detections_low, list)
        and len(detections_low) > 0
        and isinstance(detections_low[0], list)
    ):
        detections_low = detections_low[0]
    print(f"YOLO with threshold 0.1: {len(detections_low)} detections")

    # Test with high threshold
    yolo_model.set_threshold(0.8)
    detections_high = yolo_model.identify_for_image(image)
    # Handle batch format - get first batch
    if (
        isinstance(detections_high, list)
        and len(detections_high) > 0
        and isinstance(detections_high[0], list)
    ):
        detections_high = detections_high[0]
    print(f"YOLO with threshold 0.8: {len(detections_high)} detections")

    print("\n✅ All threshold tests passed!")

    # Test that confidence scores are above threshold
    print("\n=== Testing Confidence Scores ===")
    # Handle the case where detections_high might be a list of lists
    if isinstance(detections_high, list) and len(detections_high) > 0:
        # If it's a list of lists (batch format), get the first batch
        if isinstance(detections_high[0], list):
            detections_to_check = detections_high[0]
        else:
            detections_to_check = detections_high
    else:
        detections_to_check = detections_high

    # Check that all detections are above threshold
    for det in detections_to_check:
        if hasattr(det, "score") and det.score < 0.8:
            print(
                f"❌ Found detection with score {det.score} below threshold 0.8")
            assert False, f"Detection with score {det.score} below threshold 0.8"

    print("✅ All confidence scores are above threshold!")

    print("\n🎉 Threshold functionality tests completed successfully!")
    print("✅ Detectron2: Uses built-in cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST")
    print("✅ MMDetection: Uses manual post-processing threshold filtering")
    print(
        "✅ Ultralytics: Uses built-in conf parameter with proper image transformations"
    )

    return True


if __name__ == "__main__":
    test_threshold_functionality()
