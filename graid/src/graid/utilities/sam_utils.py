"""
SAM (Segment Anything Model) utilities for object mask refinement.
"""
from enum import Enum
from typing import Optional, Tuple, List

import torch
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

from graid.interfaces.ObjectDetectionI import ObjectDetectionResultI
from graid.utilities.common import get_default_device, project_root_dir

_sam_model = None


def get_sam_model(**kwargs) -> torch.nn.Module:
    """
    Get a SAM model instance, loading it if necessary.
    
    This function ensures that the SAM model is loaded only once.
    
    Args:
        model_path: Path to SAM checkpoint (default: checkpoints/sam_vit_h_4b8939.pth)
        device: Device to use for inference
        model_type: SAM model type (default: vit_h)
        
    Returns:
        The loaded SAM model.
    """
    global _sam_model
    if _sam_model is None:
        model_path = kwargs.get(
            "model_path", project_root_dir() / "checkpoints" / "sam_vit_h_4b8939.pth"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at {model_path}. "
                "Please download the SAM checkpoint following the project's setup instructions."
            )

        device = kwargs.get("device", get_default_device())
        model_type = kwargs.get("model_type", "vit_h")

        _sam_model = sam_model_registry[model_type](checkpoint=str(model_path))
        _sam_model.to(device=device)

    return _sam_model


class SAMMaskReturnType(Enum):
    LARGEST_AREA = "largest_area"
    HIGHEST_CONFIDENCE = "highest_confidence"


class SAMPredictor:
    """
    Wrapper around SAM for getting refined object masks from bounding boxes.
    """

    def __init__(self, **kwargs):
        """
        Initialize SAM predictor.
        
        Args:
            model_path: Path to SAM checkpoint (default: checkpoints/sam_vit_h_4b8939.pth)
            device: Device to use for inference
            model_type: SAM model type (default: vit_h)
        """
        sam = get_sam_model(**kwargs)
        self.predictor = SamPredictor(sam)

    def get_mask_from_bbox(
        self,
        image: Image.Image,
        detection: ObjectDetectionResultI,
        return_type: SAMMaskReturnType = SAMMaskReturnType.LARGEST_AREA,
    ) -> Optional[torch.Tensor]:
        """
        Get refined mask for an object using its bounding box as prompt.
        
        Args:
            image: PIL Image
            detection: ObjectDetectionResultI containing the bounding box
            return_type: Method to select the best mask from multiple predictions
            
        Returns:
            Binary mask as torch.Tensor of shape (H, W), or None if no valid mask
        """
        # Convert PIL Image to numpy array (RGB)
        image_array = np.array(image)

        # Set the image for SAM predictor
        self.predictor.set_image(image_array)

        # Get bounding box in XYXY format
        bbox_xyxy = detection.as_xyxy().squeeze().cpu().numpy()  # Shape: (4,)

        # SAM expects bbox in [x_min, y_min, x_max, y_max] format
        input_box = bbox_xyxy

        # Predict masks using the bounding box as prompt
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],  # Add batch dimension
            multimask_output=True,
        )

        if len(masks) == 0:
            return None

        if return_type == SAMMaskReturnType.LARGEST_AREA:
            mask_areas = np.sum(masks, axis=(1, 2))
            best_idx = np.argmax(mask_areas)
            best_mask = masks[best_idx]
        elif return_type == SAMMaskReturnType.HIGHEST_CONFIDENCE:
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
        else:
            raise ValueError(f"Invalid return_type: {return_type}")

        # Convert to torch tensor
        return torch.from_numpy(best_mask).bool()

    def get_masks_from_detections(
        self,
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
        return_type: SAMMaskReturnType = SAMMaskReturnType.LARGEST_AREA,
    ) -> List[Tuple[ObjectDetectionResultI, Optional[torch.Tensor]]]:
        """
        Get refined masks for multiple detections.
        
        Args:
            image: PIL Image
            detections: List of ObjectDetectionResultI objects
            return_type: Method to select the best mask
            
        Returns:
            List of (detection, mask) tuples where mask can be None if prediction failed
        """
        results = []

        # Set image once for all predictions
        image_array = np.array(image)
        self.predictor.set_image(image_array)

        for detection in detections:
            mask = self.get_mask_from_bbox(
                image, detection, return_type=return_type
            )
            results.append((detection, mask))

        return results

    def to(self, device: torch.device) -> "SAMPredictor":
        """Move model to specified device."""
        self.device = device
        self.predictor.model.to(device)
        return self


class SAMMaskGenerator:
    """
    Wrapper around SAM's SamAutomaticMaskGenerator.
    """

    def __init__(self, **kwargs):
        """
        Initialize SAM mask generator.
        
        Args:
            **kwargs: Arguments for SamAutomaticMaskGenerator and get_sam_model.
        """
        sam = get_sam_model(**kwargs)
        self.mask_generator = SamAutomaticMaskGenerator(sam, **kwargs)

    def generate(self, image: np.ndarray) -> List[dict]:
        """
        Generate masks for the entire image.
        
        Args:
            image: Image as a numpy array in RGB format.
            
        Returns:
            A list of masks, where each mask is a dictionary containing segmentation data.
        """
        return self.mask_generator.generate(image)


def extract_average_depth_from_mask(
    depth_map: torch.Tensor, mask: torch.Tensor
) -> Optional[float]:
    """
    Extract average depth value from pixels covered by the mask.
    
    Args:
        depth_map: Depth map tensor of shape (H, W) with depth values in meters
        mask: Binary mask tensor of shape (H, W)
        
    Returns:
        Average depth in meters, or None if mask is empty
    """
    if mask.sum() == 0:
        return None

    # Apply mask to depth map and compute average
    masked_depths = depth_map[mask]
    return float(masked_depths.mean().item())


def compare_object_depths(
    depth_map: torch.Tensor,
    detection1: ObjectDetectionResultI,
    mask1: torch.Tensor,
    detection2: ObjectDetectionResultI,
    mask2: torch.Tensor,
    margin_ratio: float = 0.1,
) -> Tuple[Optional[str], float, float]:
    """
    Compare relative depths of two objects using their masks.
    
    Args:
        depth_map: Depth map tensor (H, W) with values in meters
        detection1: First object detection
        mask1: Mask for first object
        detection2: Second object detection  
        mask2: Mask for second object
        margin_ratio: Required margin for reliable comparison
        
    Returns:
        Tuple of:
        - Comparison result: "object1_front", "object2_front", or None if too close
        - Average depth of object1
        - Average depth of object2
    """
    avg_depth1 = extract_average_depth_from_mask(depth_map, mask1)
    avg_depth2 = extract_average_depth_from_mask(depth_map, mask2)

    if avg_depth1 is None or avg_depth2 is None:
        return None, avg_depth1 or 0.0, avg_depth2 or 0.0

    # Smaller depth values mean closer to camera (in front)
    depth_diff = abs(avg_depth1 - avg_depth2)
    min_depth = min(avg_depth1, avg_depth2)

    # Check if difference is significant relative to distance
    if depth_diff < margin_ratio * min_depth:
        return None, avg_depth1, avg_depth2

    if avg_depth1 < avg_depth2:
        return "object1_front", avg_depth1, avg_depth2
    else:
        return "object2_front", avg_depth1, avg_depth2 