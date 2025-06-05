import os

import cv2
import numpy as np
import supervision as sv
import torch
from graid.utilities.common import get_default_device
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from supervision import Detections

CHECKPOINT_PATH = "../evals/sam_vit_h_4b8939.pth"
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

DEVICE = get_default_device()
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

IMAGE_PATH = "../demo/demo2.jpg"
MIN_AREA_PERCENTAGE = 0.005
MAX_AREA_PERCENTAGE = 0.05


# load image
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# segment image
sam_result = mask_generator.generate(image_rgb)
detections = sv.Detections.from_sam(sam_result=sam_result)

# filter masks
height, width, channels = image_bgr.shape
image_area = height * width

min_area_mask = (detections.area / image_area) > MIN_AREA_PERCENTAGE
max_area_mask = (detections.area / image_area) < MAX_AREA_PERCENTAGE
detections = detections[min_area_mask & max_area_mask]


def Find_Center(mask: np.ndarray) -> tuple[int, int]:
    """
    Finds the location in `mask` (binary) that is farthest from any boundary point.

    Steps:
      1. Convert the mask to 8-bit single channel if needed.
      2. Run a distance transform on that mask.
      3. Find the (x, y) location that maximizes distance from the boundary.
      4. Return that (x, y) coordinate.
    """
    # Ensure mask is single-channel, 8-bit
    mask_8u = mask.astype(np.uint8)

    # Distance transform (L2 norm)
    dist = cv2.distanceTransform(mask_8u, distanceType=cv2.DIST_L2, maskSize=3)

    # Find the global maximum in distance map
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    return max_loc


def Mark_Allocation(masks: list[np.ndarray]) -> list[tuple[int, int]]:
    """
    Sorts all region masks by ascending area, then for each mask:
      - Excludes overlap with previously processed masks
      - Finds the 'center' by distance transform
    Returns a list of (x, y) centers in that sorted order.
    """

    # 1) Sort all masks by ascending area
    #    (Compute area by summing pixels in each mask.)
    areas = [mask.sum() for mask in masks]
    sort_indices = np.argsort(areas)  # ascending
    sorted_masks = [masks[i] for i in sort_indices]

    # Prepare an "excluded region" mask to carve out overlaps
    h, w = sorted_masks[0][0].shape
    excluded = np.zeros((h, w), dtype=np.uint8)

    centers = []
    # 2) For each mask (smallest area → largest)
    for i, mask_ in enumerate(sorted_masks):
        # Convert to 8-bit for bitwise ops if necessary
        mask_8u = mask_[0].astype(np.uint8)

        # Exclude overlapping area with previously processed masks
        # final_mask = mask & NOT(excluded)
        final_mask = cv2.bitwise_and(mask_8u, cv2.bitwise_not(excluded))

        # Find center point via distance transform
        center_xy = Find_Center(final_mask)
        centers.append(center_xy)

        # Update the excluded mask (mark these pixels as used)
        excluded = cv2.bitwise_or(excluded, final_mask)

    # `centers` is in sorted‐by‐area order, matching `sorted_masks`
    return centers


all_masks = [detections[i].mask for i in range(len(detections))]

# 5) Run the Mark_Allocation algorithm to get center points in sorted order
centers = Mark_Allocation(all_masks)

# 6) We need to reorder the Detections as well to match the sorted area order
sorted_idx = np.argsort(detections.area)
sorted_detections = detections[sorted_idx]


# Create annotators
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.3)
# Annotate on a copy of the original image
annotated_image = image_bgr.copy()

# 8a) Draw the sorted masks
annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)

print(centers, len(centers))
for idx, (x, y) in enumerate(centers, start=1):
    # Draw black circular padding
    cv2.circle(annotated_image, (x, y), 11, (0, 0, 0), -1)  # Black circle

    # Put white number text on top
    cv2.putText(
        annotated_image,
        str(idx),
        (x - 6, y + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

# Finally, save the results
cv2.imwrite("annotated_image.jpg", annotated_image)
print("done!")
