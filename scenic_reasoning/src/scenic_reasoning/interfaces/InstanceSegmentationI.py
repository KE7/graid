from enum import Enum
from typing import Tuple, Union, List
import torch
from detectron2.structures import BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures.masks import polygons_to_bitmask
class Mask_Format(Enum):
    # not sure what this needs to be here
    BITMASK = 0
    POLYGON = 1
class InstanceSegmentationResultI:
    def __init__(
        self,
        score: float,
        cls: int,
        label: str,
        instance_id: int,
        mask: Union[torch.Tensor, BitMasks],
        image_hw: Tuple[int, int],
        mask_format: Mask_Format = Mask_Format.BITMASK,
    ):
        """
        Initialize InstanceSegmentationResultI.
        Args:
            score (float): Detection confidence.
            cls (int): Class ID.
            label (str): Class label.
            instance_id (int): Unique ID for each instance of the same class.
            mask (Union[torch.Tensor, BitMasks]): Segmentation mask data.
            image_hw (Tuple[int, int]): Image size (height, width).
            mask_format (Mask_Format, optional): Format of the mask. Defaults to Mask_Format.BITMASK.
        """
        self._score = score
        self._class = cls
        self._label = label
        self._instance_id = instance_id
        self._image_hw = image_hw
        if isinstance(mask, BitMasks):
            self._bitmask = mask
        else:
            # Initialize mask based on format
            if mask_format == Mask_Format.BITMASK:
                self._bitmask = BitMasks(mask.unsqueeze(0))
            elif mask_format == Mask_Format.POLYGON:
                self._bitmask = BitMasks(polygons_to_bitmask(mask, image_hw[0], image_hw[1]))
            else:
                raise NotImplementedError(f"{mask_format} not supported for initializing InstanceSegmentationResultI")
    @property
    def score(self):
        return self._score
    @property
    def cls(self):
        return self._class
    @property
    def label(self):
        return self._label
    @property
    def instance_id(self):
        return self._instance_id
    @property
    def bitmask(self):
        return self._bitmask
    def get_area(self) -> float:
        return self._bitmask.area().item()
    def as_tensor(self) -> torch.Tensor:
        return self._bitmask.tensor
    def intersection(self, other: 'InstanceSegmentationResultI') -> float:
        """
        Calculates the intersection area between this mask and another mask.
        Args:
            other (InstanceSegmentationResultI): Another segmentation result.
        Returns:
            float: Intersection area.
        """
        intersection = (self._bitmask.tensor & other.bitmask.tensor).float().sum().item()
        return intersection
    def union(self, other: 'InstanceSegmentationResultI') -> float:
        """
        Calculates the union area between this mask and another mask.
        Args:
            other (InstanceSegmentationResultI): Another segmentation result.
        Returns:
            float: Union area.
        """
        union = (self._bitmask.tensor | other.bitmask.tensor).float().sum().item()
        return union
    def iou(self, other: 'InstanceSegmentationResultI') -> float:
        """
        Calculates the Intersection over Union (IoU) between this mask and another mask.
        Args:
            other (InstanceSegmentationResultI): Another segmentation result.
        Returns:
            float: IoU score.
        """
        intersection = self.intersection(other)
        union = self.union(other)
        return intersection / union if union > 0 else 0
class InstanceSegmentationUtils:
    @staticmethod
    def pairwise_iou(
        instances1: List[InstanceSegmentationResultI],
        instances2: List[InstanceSegmentationResultI],
    ) -> torch.Tensor:
        """
        Calculates pairwise IoU between two lists of instance masks.
        Args:
            instances1 (List[InstanceSegmentationResultI]): First list of instances.
            instances2 (List[InstanceSegmentationResultI]): Second list of instances.
        Returns:
            torch.Tensor: Pairwise IoU matrix.
        """
        iou_matrix = torch.zeros((len(instances1), len(instances2)), dtype=torch.float)
        for i, inst1 in enumerate(instances1):
            for j, inst2 in enumerate(instances2):
                iou_matrix[i, j] = inst1.iou(inst2)
        return iou_matrix
    @staticmethod
    def pairwise_intersection_area(
        instances1: List[InstanceSegmentationResultI],
        instances2: List[InstanceSegmentationResultI],
    ) -> torch.Tensor:
        """
        Calculates pairwise intersection area between two lists of instance masks.
        Args:
            instances1 (List[InstanceSegmentationResultI]): First list of instances.
            instances2 (List[InstanceSegmentationResultI]): Second list of instances.
        Returns:
            torch.Tensor: Pairwise intersection area matrix.
        """
        intersection_matrix = torch.zeros((len(instances1), len(instances2)), dtype=torch.float)
        for i, inst1 in enumerate(instances1):
            for j, inst2 in enumerate(instances2):
                intersection_matrix[i, j] = inst1.intersection(inst2)
        return intersection_matrix
    @staticmethod
    def pairwise_union_area(
        instances1: List[InstanceSegmentationResultI],
        instances2: List[InstanceSegmentationResultI],
    ) -> torch.Tensor:
        """
        Calculates pairwise union area between two lists of instance masks.
        Args:
            instances1 (List[InstanceSegmentationResultI]): First list of instances.
            instances2 (List[InstanceSegmentationResultI]): Second list of instances.
        Returns:
            torch.Tensor: Pairwise union area matrix.
        """
        union_matrix = torch.zeros((len(instances1), len(instances2)), dtype=torch.float)
        for i, inst1 in enumerate(instances1):
            for j, inst2 in enumerate(instances2):
                union_matrix[i, j] = inst1.union(inst2)
        return union_matrix