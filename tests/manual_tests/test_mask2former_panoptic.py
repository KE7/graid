#!/usr/bin/env python3
"""
Manual test for Mask2Former PANOPTIC segmentation with visual output.
This test shows both instance segmentation (things) and semantic segmentation (stuff)
like road, sky, sidewalk, etc.
"""

from graid.utilities.common import yolo_bdd_transform
from graid.models.MMDetection import MMdetection_seg
from graid.data.ImageLoader import Bdd100kDataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import islice
from pathlib import Path

import sys
sys.path.append('/work/ke/research/scenic-reasoning/graid/src')


# COCO panoptic class names (things + stuff)
COCO_PANOPTIC_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    # Stuff classes start here
    'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood',
    'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
    'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
    'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged',
    'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged',
    'rock-merged', 'wall-other-merged', 'rug-merged'
]


def get_panoptic_colors():
    """Generate colors for panoptic segmentation visualization."""
    np.random.seed(42)  # For reproducible colors
    colors = []
    for i in range(len(COCO_PANOPTIC_CLASSES)):
        colors.append(tuple(np.random.randint(0, 255, 3).tolist()))
    return colors


def draw_panoptic_segmentation(image, panoptic_seg, alpha=0.6):
    """
    Draw panoptic segmentation on image.

    Args:
        image: Original image as numpy array (H, W, C)
        panoptic_seg: Panoptic segmentation tensor (H, W)
        alpha: Transparency factor for overlay

    Returns:
        Image with panoptic segmentation overlay
    """
    # Convert to RGB if needed
    if image.shape[-1] == 3:
        display_image = image.copy()
    else:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get colors for each class
    colors = get_panoptic_colors()

    # Create overlay
    overlay = display_image.copy()

    # Get unique segment IDs
    unique_ids = torch.unique(panoptic_seg)

    for segment_id in unique_ids:
        if segment_id == 0:  # Skip background
            continue

        # Create mask for this segment
        mask = (panoptic_seg == segment_id).cpu().numpy()

        # Decode segment ID to get class
        # In COCO panoptic format: segment_id = class_id * 1000 + instance_id
        class_id = segment_id // 1000
        instance_id = segment_id % 1000

        if class_id < len(COCO_PANOPTIC_CLASSES):
            class_name = COCO_PANOPTIC_CLASSES[class_id]
            color = colors[class_id]

            # Apply colored mask
            overlay[mask] = color

            # Add label for larger segments
            if mask.sum() > 1000:  # Only label segments with > 1000 pixels
                # Find centroid for label placement
                mask_indices = np.where(mask)
                if len(mask_indices[0]) > 0:
                    centroid_y = int(np.mean(mask_indices[0]))
                    centroid_x = int(np.mean(mask_indices[1]))

                    # Create label text
                    if instance_id > 0:
                        label_text = f"{class_name}_{instance_id}"
                    else:
                        label_text = class_name

                    # Draw text background
                    text_size = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(overlay,
                                  (centroid_x - text_size[0]//2 - 2,
                                   centroid_y - text_size[1]//2 - 2),
                                  (centroid_x + text_size[0]//2 + 2,
                                   centroid_y + text_size[1]//2 + 2),
                                  (0, 0, 0), -1)

                    # Draw text
                    cv2.putText(overlay, label_text,
                                (centroid_x - text_size[0]//2,
                                 centroid_y + text_size[1]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Blend original image with overlay
    result_image = cv2.addWeighted(display_image, 1-alpha, overlay, alpha, 0)

    return result_image


def analyze_panoptic_results(panoptic_seg):
    """Analyze panoptic segmentation results."""
    unique_ids = torch.unique(panoptic_seg)

    print(f"Total unique segments: {len(unique_ids)}")

    # Count things vs stuff
    things_count = 0
    stuff_count = 0
    class_counts = {}

    for segment_id in unique_ids:
        if segment_id == 0:  # Skip background
            continue

        class_id = segment_id // 1000
        instance_id = segment_id % 1000

        if class_id < len(COCO_PANOPTIC_CLASSES):
            class_name = COCO_PANOPTIC_CLASSES[class_id]

            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1

            # Things have instance IDs > 0, stuff classes have instance_id = 0
            if instance_id > 0:
                things_count += 1
            else:
                stuff_count += 1

    print(f"Things (instances): {things_count}")
    print(f"Stuff (semantic): {stuff_count}")
    print(f"Classes detected: {list(class_counts.keys())}")

    return class_counts


def main():
    """Run manual panoptic segmentation visualization test."""

    print("=== Manual Mask2Former PANOPTIC Segmentation Test ===")

    # Configuration
    NUM_IMAGES = 3
    SAVE_RESULTS = True
    SHOW_PLOTS = True

    # Initialize dataset
    print("Loading BDD100K dataset...")
    dataset = Bdd100kDataset(
        split="val",
        transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
        use_original_categories=False,
        use_extended_annotations=False,
    )

    # Initialize model
    print("Loading Mask2Former panoptic model...")
    config_file = "/work/ke/research/scenic-reasoning/install/mmdetection/configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.py"
    checkpoint_file = "https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic_20220407_104949-82f8d28d.pth"

    model = MMdetection_seg(config_file, checkpoint_file)
    print("Model loaded successfully!")

    # Create output directory
    output_dir = Path("panoptic_results")
    if SAVE_RESULTS:
        output_dir.mkdir(exist_ok=True)
        print(f"Results will be saved to: {output_dir}")

    # Process images
    print(f"\nProcessing {NUM_IMAGES} images...")

    for i in range(NUM_IMAGES):
        print(f"\n--- Image {i+1}/{NUM_IMAGES} ---")

        # Get sample from dataset
        sample = dataset[i]
        if isinstance(sample, dict) and 'image' in sample:
            image_tensor = sample['image']
            image_name = f"image_{i+1}"
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            image_tensor = sample[0]
            image_name = f"image_{i+1}"
        else:
            image_tensor = sample
            image_name = f"image_{i+1}"

        # Convert tensor to numpy for visualization
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.ndimension() == 3:
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image_tensor

        # Ensure values are in [0, 255] range
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        print(f"Image: {image_name}")
        print(f"Shape: {image_np.shape}")

        # Run inference to get panoptic results
        print("Running panoptic segmentation...")
        from mmdet.apis import inference_detector
        raw_results = inference_detector(model._model, image_np)

        if hasattr(raw_results, 'pred_panoptic_seg') and raw_results.pred_panoptic_seg is not None:
            panoptic_seg = raw_results.pred_panoptic_seg

            if hasattr(panoptic_seg, 'sem_seg'):
                sem_seg = panoptic_seg.sem_seg[0]  # Remove batch dimension
                print(f"Semantic segmentation shape: {sem_seg.shape}")

                # Analyze results
                class_counts = analyze_panoptic_results(sem_seg)

                # Create visualization
                print("Creating panoptic visualization...")
                vis_image = draw_panoptic_segmentation(image_np, sem_seg)

                # Display results
                if SHOW_PLOTS:
                    plt.figure(figsize=(20, 10))

                    # Original image
                    plt.subplot(1, 2, 1)
                    plt.imshow(image_np)
                    plt.title(f"Original Image: {image_name}")
                    plt.axis('off')

                    # Panoptic segmentation results
                    plt.subplot(1, 2, 2)
                    plt.imshow(vis_image)
                    plt.title(
                        f"Panoptic Segmentation\n({len(class_counts)} classes detected)")
                    plt.axis('off')

                    plt.tight_layout()

                    if SAVE_RESULTS:
                        save_path = output_dir / \
                            f"panoptic_{i+1}_{image_name}.png"
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        print(f"Saved: {save_path}")

                    plt.show()
            else:
                print("No semantic segmentation data found")
        else:
            print("No panoptic segmentation results found")

    print(f"\n=== Test Complete ===")
    if SAVE_RESULTS:
        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
