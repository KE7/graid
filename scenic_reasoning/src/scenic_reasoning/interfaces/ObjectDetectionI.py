from abc import ABC
from enum import Enum
from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
import torch
from detectron2.structures.boxes import Boxes as Detectron2Boxes
from detectron2.structures.boxes import (
    pairwise_intersection,
    pairwise_iou,
    pairwise_point_box_distance,
)
from PIL import Image
from ultralytics.engine.results import Boxes as UltralyticsBoxes


class BBox_Format(Enum):
    XYWH = 0
    XYWHN = 1
    XYXY = 2
    XYXYN = 3
    Detectron2Box = 4
    UltralyticsBox = 5


# TODO: Tensor-size this class since there can be a lot of boxes
class ObjectDetectionResultI:
    def __init__(
        self,
        score: float,
        cls: int,
        label: str,
        bbox: Union[torch.Tensor, Union[Detectron2Boxes, UltralyticsBoxes]],
        image_hw: Tuple[int, int],
        bbox_format: BBox_Format = BBox_Format.XYXY,
        attributes: Dict = None,
    ):
        """
        Initialize ObjectDetectionResultI.
        Note: Requires double the amount of memory to store a result
        (one for UltralyticsBoxes and one for Detectron2Boxes),
        but has methods to convert between different formats.

        Args:
            score (float): detection confidence
            cls (int): class id
            label (str): class label
            bbox (Union[torch.Tensor, Union[Detectron2Boxes, UltralyticsBoxes]]): bounding box data
            image_hw (Tuple[int, int]): image size
            bbox_format (BBox_Format, optional): format of the bounding box. Defaults to BBox_Format.XYXY.
        """
        self._scores = score
        self._class = cls
        self._label = label
        self._attributes = attributes

        if isinstance(bbox, UltralyticsBoxes):
            self._ultra_boxes = bbox
            self._detectron2_boxes = Detectron2Boxes(bbox.xyxy)
        else:
            x1, y1, x2, y2 = None, None, None, None

            if bbox_format == BBox_Format.XYWH:
                x1, y1 = bbox[0], bbox[1]
                x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            elif bbox_format == BBox_Format.XYXY:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                raise NotImplementedError(
                    f"{bbox_format} not supported for initializing DetectionResult"
                )

            box = torch.tensor([x1, y1, x2, y2, score, cls])

            self._ultra_boxes = UltralyticsBoxes(boxes=box, orig_shape=image_hw)
            self._detectron2_boxes = Detectron2Boxes(self._ultra_boxes.xyxy)

    def as_xyxy(self):
        return self._ultra_boxes.xyxy

    def as_xyxyn(self):
        return self._ultra_boxes.xyxyn

    def as_xywh(self):
        return self._ultra_boxes.xywh

    def as_xywhn(self):
        return self._ultra_boxes.xywhn

    @property
    def scores(self):
        return self._scores

    @property
    def label(self):
        return self._label

    @property
    def cls(self):
        return self._class

    @property
    def as_ultra_box(self):
        return self._ultra_boxes

    @property
    def as_detectron_box(self):
        return self.as_detectron_box

    # the rest of this code is adapted from
    # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/structures/boxes.html

    def inside_box(
        self, box_size: Tuple[int, int], boundary_threshold: int = 0
    ) -> torch.Tensor:
        return self._detectron2_boxes.inside_box(box_size, boundary_threshold)

    def get_center(self) -> torch.Tensor:
        return self._detectron2_boxes.get_centers()

    def get_area(self) -> float:
        return self._detectron2_boxes.area()[0]


class ObjectDetectionUtils:
    @staticmethod
    def pairwise_iou(boxes1: ObjectDetectionResultI, boxes2: ObjectDetectionResultI):
        return pairwise_iou(boxes1._detectron2_boxes, boxes2._detectron2_boxes)

    @staticmethod
    def pairwise_intersection_area(
        boxes1: ObjectDetectionResultI, boxes2: ObjectDetectionResultI
    ):
        return pairwise_intersection(boxes1._detectron2_boxes, boxes2._detectron2_boxes)

    @staticmethod
    def pairwise_point_box_distance(
        points: torch.Tensor, boxes: ObjectDetectionResultI
    ):
        return pairwise_point_box_distance(points, boxes._detectron2_boxes)

    @staticmethod
    def compute_metrics(
        ground_truth: List[ObjectDetectionResultI],
        predictions: List[ObjectDetectionResultI],
        iou_threshold: float = 0.5,
        debug: bool = False,
        image: Image.Image = None,
    ) -> Dict[str, float]:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        if len(ground_truth) == 0 or len(predictions) == 0:
            return {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": 0,
                "recall": 0,
                "f1": 0,
            }

        gt_boxes = Detectron2Boxes(
            torch.stack(
                [gt._detectron2_boxes.tensor for gt in ground_truth], dim=0
            ).squeeze(1)
        )
        pred_boxes = Detectron2Boxes(
            torch.stack(
                [pred._detectron2_boxes.tensor for pred in predictions], dim=0
            ).squeeze(1)
        )

        # Given two lists of boxes of size N and M, compute the IoU
        # (intersection over union) between **all** N x M pairs of boxes.
        ious = pairwise_iou(gt_boxes, pred_boxes)  # Shape: (N, M)

        # Find best matching predicted box for each ground truth box
        max_ious_per_gt = torch.max(ious, axis=1)  # Shape: (N,)

        # Find best matching ground truth box for each predicted box
        max_ious_per_pred = torch.max(ious, axis=0)  # Shape: (M,)

        # True positives: ground truth boxes matched with predictions above threshold
        true_positives = torch.sum(max_ious_per_gt.values >= iou_threshold)

        # False positives are predicted boxes that were not matched
        false_positives = torch.sum(max_ious_per_pred.values < iou_threshold)

        # False negatives are ground truth boxes that were not matched
        false_negatives = torch.sum(max_ious_per_gt.values < iou_threshold)

        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        # Calculate AP per class
        ground_truth_classes = set([gt.label.lower() for gt in ground_truth])
        predictions_classes = set([pred.label.lower() for pred in predictions])
        classes = ground_truth_classes.union(predictions_classes)
        ap_per_class = {}
        for cls in classes:
            cls_gt = [gt for gt in ground_truth if gt.label.lower() == cls]
            cls_pred = [pred for pred in predictions if pred.label.lower() == cls]

            if len(cls_gt) == 0 or len(cls_pred) == 0:
                ap_per_class[cls] = 0
                continue

            cls_gt_boxes = Detectron2Boxes(
                torch.stack(
                    [gt._detectron2_boxes.tensor for gt in cls_gt], dim=0
                ).squeeze(1)
            )
            cls_pred_boxes = Detectron2Boxes(
                torch.stack(
                    [pred._detectron2_boxes.tensor for pred in cls_pred], dim=0
                ).squeeze(1)
            )

            ious = pairwise_iou(cls_gt_boxes, cls_pred_boxes)
            max_ious_per_gt = torch.max(ious, axis=1)
            max_ious_per_pred = torch.max(ious, axis=0)

            true_positives = torch.sum(max_ious_per_gt.values >= iou_threshold).item()
            false_positives = torch.sum(max_ious_per_pred.values < iou_threshold).item()
            false_negatives = torch.sum(max_ious_per_gt.values < iou_threshold).item()

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)

            denominator = precision + recall
            if denominator == 0:
                ap_per_class[cls] = 0
            else:
                ap_per_class[cls] = 2 * (precision * recall) / denominator

        # Calculate mAP
        mAP = np.mean(list(ap_per_class.values()))

        f1_denominator = precision + recall
        if f1_denominator == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / f1_denominator

        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap_per_class": ap_per_class,
            "mAP": mAP,
        }


class ObjectDetectionModelI(ABC):
    def __init__(self):
        pass

    def identify_for_image(
        self, image: Union[Image.Image, torch.Tensor, List[torch.Tensor]]
    ) -> List[ObjectDetectionResultI]:
        pass

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[ObjectDetectionResultI]]:
        pass
