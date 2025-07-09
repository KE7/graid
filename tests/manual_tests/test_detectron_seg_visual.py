"""
Manual test for Detectron2 instance segmentation with visual output.
This test loads images, runs segmentation, and displays results with overlaid masks
showing class names and confidence scores.
"""

from itertools import islice
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from graid.data.ImageLoader import Bdd100kDataset
from graid.models.Detectron import Detectron_seg
from graid.utilities.common import get_default_device


def draw_masks_with_labels(image, results, alpha=0.5):
    """
    Draw segmentation masks on image with class labels and confidence scores.

    Args:
        image: Original image as numpy array (H, W, C)
        results: List of InstanceSegmentationResultI objects
        alpha: Transparency factor for mask overlay

    Returns:
        Image with overlaid masks and labels
    """
    # Convert to RGB if needed
    if image.shape[-1] == 3:
        display_image = image.copy()
    else:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Colors for different instances (using distinct colors)
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (255, 192, 203),  # Pink
        (128, 128, 0),  # Olive
    ]

    # Create overlay for masks
    overlay = display_image.copy()

    for i, result in enumerate(results):
        # Get mask as numpy array
        mask = result.as_tensor()[0].cpu().numpy().astype(bool)

        # Get color for this instance
        color = colors[i % len(colors)]

        # Apply colored mask
        overlay[mask] = color

        # Find bounding box for label placement
        mask_indices = np.where(mask)
        if len(mask_indices[0]) > 0:
            y_min, y_max = mask_indices[0].min(), mask_indices[0].max()
            x_min, x_max = mask_indices[1].min(), mask_indices[1].max()

            # Create label text
            label_text = f"{result.label}: {result.score:.1%}"

            # Calculate text position (top-left of bounding box)
            text_x = max(0, x_min)
            text_y = max(20, y_min)

            # Draw text background
            text_size = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                overlay,
                (text_x, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 10, text_y + 5),
                (0, 0, 0),
                -1,
            )

            # Draw text
            cv2.putText(
                overlay,
                label_text,
                (text_x + 5, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    # Blend original image with overlay
    result_image = cv2.addWeighted(display_image, 1 - alpha, overlay, alpha, 0)

    return result_image


def main():
    """Run manual segmentation visualization test."""

    print("=== Manual Detectron2 Segmentation Visualization Test ===")

    # Configuration
    NUM_IMAGES = 3
    SAVE_RESULTS = True
    SHOW_PLOTS = True

    # Initialize dataset
    print("Loading BDD100K dataset...")
    bdd = Bdd100kDataset(
        split="val", use_original_categories=False, use_extended_annotations=False)

    # Initialize model with standard Detectron2 Mask R-CNN
    print("Loading Detectron2 segmentation model...")
    model = Detectron_seg(
        config_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        weights_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        threshold=0.3,  # Lower threshold to see more detections
        device=get_default_device(),
    )

    print(f"Model loaded on device: {get_default_device()}")

    # Create output directory
    output_dir = Path("segmentation_results")
    if SAVE_RESULTS:
        output_dir.mkdir(exist_ok=True)
        print(f"Results will be saved to: {output_dir}")

    # Process images
    print(f"\nProcessing {NUM_IMAGES} images...")

    for i, data in enumerate(islice(bdd, NUM_IMAGES)):
        print(f"\n--- Image {i+1}/{NUM_IMAGES} ---")

        # Extract image and metadata
        image_tensor = data["image"]  # Shape: [3, H, W]
        image_name = data["name"]

        # Convert tensor to numpy for visualization (CHW -> HWC)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

        # Ensure values are in [0, 255] range
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        print(f"Image: {image_name}")
        print(f"Shape: {image_np.shape}")

        # Run segmentation
        print("Running segmentation...")
        results = model.identify_for_image(image_tensor)

        print(f"Found {len(results)} instances:")
        for j, result in enumerate(results):
            print(f"  {j+1}. {result.label}: {result.score:.1%}")

        # Create visualization
        if len(results) > 0:
            print("Creating visualization...")
            vis_image = draw_masks_with_labels(image_np, results)

            # Display results
            if SHOW_PLOTS:
                plt.figure(figsize=(15, 10))

                # Original image
                plt.subplot(1, 2, 1)
                plt.imshow(image_np)
                plt.title(f"Original Image: {image_name}")
                plt.axis("off")

                # Segmentation results
                plt.subplot(1, 2, 2)
                plt.imshow(vis_image)
                plt.title(f"Segmentation Results ({len(results)} instances)")
                plt.axis("off")

                plt.tight_layout()

                if SAVE_RESULTS:
                    save_path = output_dir / \
                        f"segmentation_{i+1}_{image_name}.png"
                    plt.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(f"Saved: {save_path}")

                plt.show()
        else:
            print("No instances detected in this image.")

    print(f"\n=== Test Complete ===")
    if SAVE_RESULTS:
        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
