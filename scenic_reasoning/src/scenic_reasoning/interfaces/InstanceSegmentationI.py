from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.structures import BitMasks
from detectron2.structures.boxes import pairwise_intersection, pairwise_iou
from detectron2.structures.masks import polygons_to_bitmask
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class Mask_Format(Enum):
    BITMASK = 0
    POLYGON = 1
    RLE = 2


class InstanceSegmentationResultI:
    def __init__(
        self,
        score: float,
        cls: int,
        label: str,
        instance_id: int,
        image_hw: Tuple[int, int],
        mask: Union[torch.Tensor, BitMasks],
        mask_format: Mask_Format = Mask_Format.BITMASK,
        attributes: Optional[List[List[Dict]]] = None,
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
                self._bitmask = BitMasks(mask)
            elif mask_format == Mask_Format.POLYGON:
                self._bitmask = BitMasks(
                    polygons_to_bitmask(mask, image_hw[0], image_hw[1])
                )
            elif mask_format == Mask_Format.RLE:
                height, width = self._image_hw
                # TODO: implement this if needed. NuImage doens't need this yet.

            else:
                raise NotImplementedError(
                    f"{mask_format} not supported for initializing InstanceSegmentationResultI"
                )

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

    def intersection(self, other: "InstanceSegmentationResultI") -> torch.Tensor:
        """
        Calculates the intersection area between this mask and another mask.
        Args:
            other (InstanceSegmentationResultI): Another segmentation result.
        Returns:
            float: Intersection area.
        """
        return pairwise_intersection(self._bitmask, other.bitmask)

    def union(self, other: "InstanceSegmentationResultI") -> torch.Tensor:
        """
        Calculates the union area between this mask and another mask.
        Args:
            other (InstanceSegmentationResultI): Another segmentation result.
        Returns:
            float: Union area.
        """
        union = (self._bitmask.tensor | other.bitmask.tensor).float()
        return union

    def iou(self, other: "InstanceSegmentationResultI") -> torch.Tensor:
        """
        Calculates the Intersection over Union (IoU) between this mask and another mask.
        Args:
            other (InstanceSegmentationResultI): Another segmentation result.
        Returns:
            float: IoU score.
        """
        return pairwise_iou(self._bitmask, other.bitmask)


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
        intersection_matrix = torch.zeros(
            (len(instances1), len(instances2)), dtype=torch.float
        )
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
        union_matrix = torch.zeros(
            (len(instances1), len(instances2)), dtype=torch.float
        )
        for i, inst1 in enumerate(instances1):
            for j, inst2 in enumerate(instances2):
                union_matrix[i, j] = inst1.union(inst2)
        return union_matrix

    @staticmethod
    def compute_metrics_for_single_img(
        ground_truth: List[InstanceSegmentationResultI],
        predictions: List[InstanceSegmentationResultI],
        class_metrics: bool = False,
        extended_summary: bool = False,
        debug: bool = False,
        image: Optional[Image.Image] = None,
    ) -> Dict[str, float]:

        masks = []
        scores = []
        class_ids = []
        instance_ids = []

        for truth in ground_truth:
            masks.append(truth._bitmask.tensor)
            scores.append(truth._score)  # score is a float or tensor
            class_ids.append(truth._class)
            instance_ids.append(truth._instance_id)

        scores = (
            (
                torch.tensor(scores)
                if isinstance(scores[0], float)
                else torch.cat(scores)
            )
            if scores
            else torch.tensor([])
        )
        class_ids = (
            (
                torch.tensor(class_ids)
                if isinstance(class_ids[0], int)
                else torch.cat(class_ids)
            )
            if class_ids
            else torch.tensor([])
        )

        masks = torch.cat(masks) if masks else torch.tensor([])

        targets = [dict(masks=masks, scores=scores, labels=class_ids)]

        pred_masks = []
        pred_scores = []
        pred_class_ids = []
        pred_instance_ids = []

        for pred in predictions:
            pred_masks.append(pred._bitmask.tensor)
            pred_scores.append(pred._score)  # score is a float or tensor
            pred_class_ids.append(int(torch.tensor(pred._class)))
            pred_instance_ids.append(pred._instance_id)

        pred_scores = torch.tensor(pred_scores)
        pred_class_ids = torch.tensor(pred_class_ids)

        pred_masks = torch.cat(pred_masks) if pred_masks else torch.tensor([])

        preds = [dict(masks=pred_masks, scores=pred_scores, labels=pred_class_ids)]

        metric = MeanAveragePrecision(
            iou_type="segm",
            class_metrics=class_metrics,
            extended_summary=extended_summary,
        )

        metric.update(preds, targets)

        return metric.compute()


class InstanceSegmentationModelI(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def identify_for_image(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
    ) -> List[List[Optional[InstanceSegmentationResultI]]]:
        pass

    @abstractmethod
    def identify_for_image_as_tensor(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs,
    ) -> List[Optional[InstanceSegmentationResultI]]:
        pass

    @abstractmethod
    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[Optional[InstanceSegmentationResultI]]]:
        pass

    @abstractmethod
    def to(self, device: Union[str, torch.device]):
        pass
