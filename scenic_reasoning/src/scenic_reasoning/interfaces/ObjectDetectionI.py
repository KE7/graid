from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from pathlib import Path
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
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics.engine.results import Boxes as UltralyticsBoxes


class BBox_Format(Enum):
    XYWH = 0
    XYWHN = 1
    XYXY = 2
    XYXYN = 3
    Detectron2Box = 4
    UltralyticsBox = 5


class ObjectDetectionResultI:
    def __init__(
        self,
        score: Union[float, torch.Tensor],
        cls: Union[int, torch.Tensor],
        label: Union[str, torch.Tensor],
        bbox: Union[
            Union[torch.Tensor, List[float]], Union[Detectron2Boxes, UltralyticsBoxes]
        ],
        image_hw: Tuple[int, int],
        bbox_format: BBox_Format = BBox_Format.XYXY,
        attributes: Dict = None,
    ):
        """
        Initialize ObjectDetectionResultI. If you are creating multiple
            bounding boxes, via a tensor, only XYXY format is supported.
        Note: Requires double the amount of memory to store a result
        (one for UltralyticsBoxes and one for Detectron2Boxes),
        but has methods to convert between different formats.

        Args:
            score (Union[float, torch.Tensor]):
                detection confidence. If a tensor, should have shape (# of boxes,)
            cls (Union[int, torch.Tensor]):
                class id. If a tensor, should have shape (# of boxes,)
            label (Union[str, torch.Tensor]):
                class label. If a tensor, should have shape (# of boxes,)
            bbox (Union[torch.Tensor, Union[Detectron2Boxes, UltralyticsBoxes]]):
                bounding box data
            image_hw (Tuple[int, int]): image size
            bbox_format (BBox_Format, optional):
                format of the bounding box. Defaults to BBox_Format.XYXY.
        """
        self._score = score
        self._class = cls
        self._label = label
        self._attributes = attributes

        if isinstance(bbox, UltralyticsBoxes):
            self._ultra_boxes = bbox
            self._detectron2_boxes = Detectron2Boxes(bbox.xyxy)
        elif isinstance(bbox, torch.Tensor):
            # should have shape (# of boxes, 6) where each row is:
            #   (x1, y1, x2, y2, score, cls, <optional: track_id>)

            assert (
                bbox_format == BBox_Format.XYXY
            ), "Only XYXY format supported for tensor input"

            if (
                bbox.shape[1] == 4
                and isinstance(score, torch.Tensor)
                and isinstance(cls, torch.Tensor)
            ):
                bbox = torch.cat([bbox, score.unsqueeze(1), cls.unsqueeze(1)], dim=1)
            elif bbox.shape[1] == 4:
                raise ValueError(
                    f"Tried to initialize DetectionResult with {bbox.shape[0]} many "
                    "bounding boxes but only a single score and class provided."
                )
            elif box.shape[1] == 6 or bbox.shape[1] == 7:
                self._score = bbox[:, 4]
                self._class = bbox[:, 5]
            elif bbox.shape[1] != 6 and bbox.shape[1] != 7:
                raise ValueError(
                    f"{bbox.shape[1]} not supported for initializing DetectionResult"
                    " (should be 6 or 7)"
                )
            # all x1 < x2 and y1 < y2
            if torch.any(bbox[:, 0] > bbox[:, 2]) or torch.any(bbox[:, 1] > bbox[:, 3]):
                raise ValueError(
                    f"Bounding box coordinates are not in the correct format. "
                    "All x1 < x2 and y1 < y2 but found "
                    f"{bbox[:, 0]} > {bbox[:, 2]} and {bbox[:, 1]} > {bbox[:, 3]}"
                )
            self._ultra_boxes = UltralyticsBoxes(boxes=bbox, orig_shape=image_hw)
        elif isinstance(bbox, List):
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

    def as_xyxy(self) -> torch.Tensor:
        return self._ultra_boxes.xyxy

    def as_xyxyn(self) -> torch.Tensor:
        return self._ultra_boxes.xyxyn

    def as_xywh(self) -> torch.Tensor:
        return self._ultra_boxes.xywh

    def as_xywhn(self) -> torch.Tensor:
        return self._ultra_boxes.xywhn

    def _check_self_consistency(self):
        # if the scores are a float, then the class and label should be too
        # and the bbox should have shape (4,)
        if isinstance(self._score, float):
            assert isinstance(
                self._class, int
            ), "Single instance detection result does not have a single int class"
            assert isinstance(
                self._label, str
            ), "Single instance detection result does not have a single string label"
            assert self._ultra_boxes.shape == (
                1,
                6,
            ), "Single instance detection result does not have a single bounding box"
        elif isinstance(self._score, torch.Tensor):
            assert isinstance(
                self._class, torch.Tensor
            ), "Tensor detection result does not have a tensor class"
            assert isinstance(
                self._label, torch.Tensor
            ), "Tensor detection result does not have a tensor label"
            assert self._ultra_boxes.xyxy.shape == (
                self._score.shape[0],
                6,
            ), "Tensor detection result does not have a tensor bounding box in the correct Utralytics format"
            assert (
                self._class.shape == self._score.shape == self._label.shape
            ), "Tensor detection result does not have matching sizes for scores, classes, and labels"
            assert (
                self._ultra_boxes.xyxy.shape[0] == self._score.shape[0]
            ), "Tensor detection result does not have the same number of bounding boxes as scores, classes, and labels"

    @property
    def score(self) -> Union[float, torch.Tensor]:
        self._check_self_consistency()
        return self._score

    @property
    def label(self) -> Union[str, torch.Tensor]:
        self._check_self_consistency()
        return self._label

    @property
    def cls(self) -> Union[int, torch.Tensor]:
        self._check_self_consistency()
        return self._class

    @property
    def as_ultra_box(self) -> UltralyticsBoxes:
        self._check_self_consistency()
        return self._ultra_boxes

    @property
    def as_detectron_box(self) -> Detectron2Boxes:
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
    def pairwise_iou(
        boxes1: ObjectDetectionResultI, boxes2: ObjectDetectionResultI
    ) -> torch.Tensor:
        return pairwise_iou(boxes1._detectron2_boxes, boxes2._detectron2_boxes)

    @staticmethod
    def pairwise_intersection_area(
        boxes1: ObjectDetectionResultI, boxes2: ObjectDetectionResultI
    ) -> torch.Tensor:
        return pairwise_intersection(boxes1._detectron2_boxes, boxes2._detectron2_boxes)

    @staticmethod
    def pairwise_point_box_distance(
        points: torch.Tensor, boxes: ObjectDetectionResultI
    ) -> torch.Tensor:
        return pairwise_point_box_distance(points, boxes._detectron2_boxes)

    @staticmethod
    def compute_metrics(
        ground_truth: List[ObjectDetectionResultI],
        predictions: List[ObjectDetectionResultI],
        iou_threshold: float = 0.5,
        debug: bool = False,
        image: Image.Image = None,
    ) -> Dict[str, float]:
        # # Initialize variables
        # ap_per_class = {}
        # all_detections = defaultdict(list)
        # all_annotations = defaultdict(list)

        # # Collect annotations and detections per class
        # for gt in ground_truth:
        #     all_annotations[gt.label.lower()].append(gt)

        # for pred in predictions:
        #     all_detections[pred.label.lower()].append(pred)

        # classes = set(all_annotations.keys()).union(set(all_detections.keys()))

        # # For overall metrics
        # total_true_positives = 0
        # total_false_positives = 0
        # total_false_negatives = 0

        # for cls in classes:
        #     cls_gt = all_annotations.get(cls, [])
        #     cls_pred = all_detections.get(cls, [])

        #     n_gt = len(cls_gt)
        #     n_pred = len(cls_pred)

        #     # No ground truth or predictions for this class
        #     if n_gt == 0 and n_pred == 0:
        #         ap_per_class[cls] = 0
        #         continue
        #     elif n_gt == 0:
        #         # All predictions are false positives
        #         total_false_positives += n_pred
        #         ap_per_class[cls] = 0
        #         continue
        #     elif n_pred == 0:
        #         # All ground truths are false negatives
        #         total_false_negatives += n_gt
        #         ap_per_class[cls] = 0
        #         continue

        #     # Extract tensors from your class
        #     gt_boxes_tensor = torch.cat([gt._detectron2_boxes.tensor for gt in cls_gt], dim=0)  # Shape: [n_gt, 4]
        #     pred_boxes_tensor = torch.cat([pred._detectron2_boxes.tensor for pred in cls_pred], dim=0)  # Shape: [n_pred, 4]
        #     pred_scores = torch.tensor([pred.scores for pred in cls_pred])  # Shape: [n_pred]

        #     # Sort predictions by confidence score
        #     sorted_indices = torch.argsort(pred_scores, descending=True)
        #     pred_boxes_tensor = pred_boxes_tensor[sorted_indices]
        #     pred_scores = pred_scores[sorted_indices]

        #     # Compute IoU matrix between all predictions and all ground truths
        #     iou_matrix = pairwise_iou(
        #         Detectron2Boxes(pred_boxes_tensor),
        #         Detectron2Boxes(gt_boxes_tensor)
        #     )  # Shape: [n_pred, n_gt]

        #     # Initialize matches (1: matched, 0: unmatched)
        #     matched_gt = torch.zeros(n_gt, dtype=torch.bool)

        #     # True positives and false positives
        #     TP = torch.zeros(n_pred)
        #     FP = torch.zeros(n_pred)

        #     # For each prediction, find the best matching ground truth
        #     for pred_idx in range(n_pred):
        #         # Get IoUs for this prediction with all ground truths
        #         ious = iou_matrix[pred_idx]  # Shape: [n_gt]

        #         # Set IoUs to zero for already matched ground truths
        #         ious[matched_gt] = 0

        #         # Find the maximum IoU and the corresponding ground truth index
        #         max_iou, max_gt_idx = torch.max(ious, dim=0)

        #         if max_iou >= iou_threshold:
        #             TP[pred_idx] = 1
        #             matched_gt[max_gt_idx] = True
        #         else:
        #             FP[pred_idx] = 1

        #     # Compute cumulative true positives and false positives
        #     cum_TP = torch.cumsum(TP, dim=0)
        #     cum_FP = torch.cumsum(FP, dim=0)

        #     # Compute precision and recall
        #     precision = cum_TP / (cum_TP + cum_FP + 1e-16)
        #     recall = cum_TP / (n_gt + 1e-16)

        #     # Append sentinel values
        #     mrec = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
        #     mpre = torch.cat((torch.tensor([0.0]), precision, torch.tensor([0.0])))

        #     # Compute the precision envelope
        #     for i in range(mpre.numel() - 2, -1, -1):
        #         mpre[i] = torch.max(mpre[i], mpre[i + 1])

        #     # Integrate area under curve
        #     idx = torch.where(mrec[1:] != mrec[:-1])[0]
        #     ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()

        #     ap_per_class[cls] = ap

        #     # Collect overall metrics
        #     total_true_positives += TP.sum().item()
        #     total_false_positives += FP.sum().item()
        #     total_false_negatives += n_gt - TP.sum().item()

        # # Compute mAP
        # mAP = np.mean(list(ap_per_class.values())) if len(ap_per_class) > 0 else 0

        # # Compute overall precision and recall
        # total_precision_denominator = total_true_positives + total_false_positives + 1e-16
        # total_recall_denominator = total_true_positives + total_false_negatives + 1e-16

        # precision = total_true_positives / total_precision_denominator
        # recall = total_true_positives / total_recall_denominator

        # f1 = 2 * (precision * recall) / (precision + recall + 1e-16)

        # return {
        #     "true_positives": int(total_true_positives),
        #     "false_positives": int(total_false_positives),
        #     "false_negatives": int(total_false_negatives),
        #     "precision": precision,
        #     "recall": recall,
        #     "f1": f1,
        #     "ap_per_class": ap_per_class,
        #     "mAP": mAP,
        # }
        boxes = []
        scores = []
        classes = []
        for truth in ground_truth:
            boxes.append(truth.as_xyxy())
            scores.append(truth.score)  # score is a float or tensor
            classes.append(truth.cls)

        # boxes = torch.stack(boxes) # shape: (num_boxes, 1, 4)
        boxes = torch.cat(boxes)  # shape: (num_boxes, 4)
        scores = (
            torch.tensor(scores) if isinstance(scores[0], float) else torch.cat(scores)
        )
        classes = (
            torch.tensor(classes) if isinstance(classes[0], int) else torch.cat(classes)
        )

        targets: List[Dict[str, torch.Tensor]] = [
            dict(boxes=boxes.squeeze(1), labels=classes, scores=scores)
        ]

        pred_boxes = []
        pred_scores = []
        pred_classes = []
        for pred in predictions:
            pred_boxes.append(pred.as_xyxy())
            pred_scores.append(pred.score)  # score is a float or tensor
            pred_classes.append(pred.cls)

        pred_boxes = torch.cat(pred_boxes)
        pred_scores = (
            torch.tensor(pred_scores)
            if isinstance(pred_scores[0], float)
            else torch.cat(pred_scores)
        )
        pred_classes = (
            torch.tensor(pred_classes)
            if isinstance(pred_classes[0], int)
            else torch.cat(pred_classes)
        )

        preds: List[Dict[str, torch.Tensor]] = [
            dict(boxes=pred_boxes, labels=pred_classes, scores=pred_scores)
        ]

        metric = MeanAveragePrecision(
            class_metrics=True,
            extended_summary=True,
        )

        metric.update(targets, preds)

        return metric.compute()


class ObjectDetectionModelI(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def identify_for_image(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
    ) -> List[ObjectDetectionResultI]:
        pass

    @abstractmethod
    def identify_for_image_as_tensor(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs,
    ) -> List[ObjectDetectionResultI]:
        pass

    @abstractmethod
    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[ObjectDetectionResultI]]:
        pass
