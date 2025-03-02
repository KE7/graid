from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import random
import cv2
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
        attributes: Optional[List[Dict]] = None,
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
            elif bbox.shape[1] == 6 or bbox.shape[1] == 7:
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

    def flatten(self) -> List["ObjectDetectionResultI"]:
        """
        If the current detection result is a single instance, it returns itself
        Otherwise, it returns a list of detection results based on the n boxes
        that were in the original detection result.
        """
        if isinstance(self._score, float):
            return [self]
        elif isinstance(self._score, torch.Tensor):
            return [
                ObjectDetectionResultI(
                    score=self._score[i],
                    cls=self._class[i],
                    label=self._label[i],
                    bbox=self._ultra_boxes.xyxy[i],
                    image_hw=self._ultra_boxes.orig_shape,
                )
                for i in range(self._score.shape[0])
            ]
        else:
            raise NotImplementedError(
                f"{type(self._score)} not supported for flattening DetectionResult"
            )

    def flatten_to_boxes(
        self, bbox_format: BBox_Format = BBox_Format.XYXY
    ) -> List[Tuple[str, int, float, torch.Tensor]]:
        """
        Flattens the bounding boxes (which can be shape (N, 4)
        into a list of tuples of the form (label, class, score, box).
        Each tuple corresponds to a single bounding box.
        """
        if bbox_format == BBox_Format.XYXY:
            boxes = self.as_xyxy()
        elif bbox_format == BBox_Format.XYWH:
            boxes = self.as_xywh()
        elif bbox_format == BBox_Format.XYWHN:
            boxes = self.as_xywhn()
        elif bbox_format == BBox_Format.XYXYN:
            boxes = self.as_xyxyn()
        else:
            raise NotImplementedError(
                f"{bbox_format} not supported for flattening DetectionResult"
            )

        # Flatten the boxes
        flattened_boxes = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            label = (
                self.label[i].item()
                if isinstance(self.label, torch.Tensor)
                else self.label
            )
            cls = self.cls[i].item() if isinstance(self.cls, torch.Tensor) else self.cls
            score = (
                self.score[i].item()
                if isinstance(self.score, torch.Tensor)
                else self.score
            )
            flattened_boxes.append((label, cls, score, box))

        return flattened_boxes

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

    def get_area(self) -> torch.Tensor:
        return self._detectron2_boxes.area()


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
    def compute_metrics_for_single_img(
        ground_truth: List[ObjectDetectionResultI],
        predictions: List[ObjectDetectionResultI],
        class_metrics: bool = False,
        extended_summary: bool = False,
        debug: bool = False,
        image: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        
        gt_classes_set = set([truth.cls for truth in ground_truth])
        pred_classes_set = set([pred.cls for pred in predictions])
        intersection_classes = gt_classes_set.intersection(pred_classes_set)


        pred_boxes = []
        pred_scores = []
        pred_classes = []
        remove_indices_pred = []

        for i, pred in enumerate(predictions):
            if pred.cls not in intersection_classes:
                remove_indices_pred.append(i)
                continue
            pred_boxes.append(pred.as_xyxy())
            pred_scores.append(pred.score)  # score is a float or tensor
            pred_classes.append(pred.cls)

        for i in sorted(remove_indices_pred, reverse=True):
            predictions.pop(i)

        pred_boxes = torch.cat(pred_boxes) if pred_boxes else torch.Tensor([])
        pred_scores = (
            (
                torch.tensor(pred_scores)
                if isinstance(pred_scores[0], float)
                else torch.cat(pred_scores)
            )
            if pred_scores
            else torch.Tensor([])
        )
        pred_classes = (
            (
                torch.tensor(pred_classes)
                if isinstance(pred_classes[0], int)
                else torch.cat(pred_classes)
            )
            if pred_classes
            else torch.Tensor([])
        )

        preds: List[Dict[str, torch.Tensor]] = [
            dict(boxes=pred_boxes, labels=pred_classes, scores=pred_scores)
        ]


        boxes = []
        scores = []
        classes = []
        remove_indices_gt = []
        for i, truth in enumerate(ground_truth):
            if truth.cls not in intersection_classes:
                remove_indices_gt.append(i)
                continue
            boxes.append(truth.as_xyxy())
            scores.append(truth.score)
            classes.append(truth.cls)

        for i in sorted(remove_indices_gt, reverse=True):
            ground_truth.pop(i)

        num_fake_boxes = max(0, len(pred_boxes) - len(boxes))
        image_size = (image.shape[1], image.shape[0])
        fake_bbox_size = 20
        for i in range(num_fake_boxes):
            x1 = i * fake_bbox_size
            y1 = fake_bbox_size
            x2 = x1 + fake_bbox_size
            y2 = y1 + fake_bbox_size

            fake_bbox = [x1, y1, x2, y2]
            fake_score = 1.0
            fake_class = -1
            fake_label = "fake"


            fake_detection = ObjectDetectionResultI(
                score=fake_score,
                cls=fake_class,
                label=fake_label,
                bbox=fake_bbox,
                image_hw=image_size
            )
            ground_truth.append(fake_detection)

            boxes.append(fake_detection.as_xyxy())
            scores.append(fake_detection.score) 
            classes.append(fake_detection.cls)

        boxes = torch.cat(boxes) if boxes else torch.Tensor([])  # shape: (num_boxes, 4)
        scores = (
            torch.tensor(scores) if isinstance(scores[0], float) else torch.cat(scores)
        ) if scores else torch.Tensor([])
        classes = (
            torch.tensor(classes) if isinstance(classes[0], int) else torch.cat(classes)
        ) if classes else torch.Tensor([])


        targets: List[Dict[str, torch.Tensor]] = [
            dict(boxes=boxes, labels=classes, scores=scores)
        ]
        

        metric = MeanAveragePrecision(
            class_metrics=class_metrics,
            extended_summary=extended_summary,
            box_format='xyxy',
            iou_thresholds=[0.25],
            iou_type='bbox',
            backend='faster_coco_eval',
            max_detection_thresholds=[10, 20, 100]
        )

        metric.update(target=targets, preds=preds)

        
        return metric.compute()

    def show_image_with_detections(image : Image.Image, detections: List[ObjectDetectionResultI]) -> None:
        # Convert PIL image to a NumPy array in OpenCV's BGR format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Make a copy to draw bounding boxes on
        cv_image_with_boxes = cv_image.copy()

        # Draw bounding boxes and labels on the copy
        for detection in detections:
            bbox = detection.as_xyxy()

            # Handle cases where bbox might have multiple boxes
            if bbox.shape[0] > 1:
                for i, box in enumerate(bbox):
                    x1, y1, x2, y2 = map(int, box)
                    score = detection.score[i].item()
                    label = str(detection.label[i].item())

                    # Choose a color and draw rectangle
                    if score > 0.8:
                        color = (0, 255, 0)
                    elif score > 0.5:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)
                    cv2.rectangle(cv_image_with_boxes, (x1, y1), (x2, y2), color, 2)

                    # Put label text above the box
                    cv2.putText(
                        cv_image_with_boxes,
                        f"{label}: {score:.2f}",
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )
            else:
                x1, y1, x2, y2 = map(int, bbox[0])
                score = detection.score
                label = str(detection.label)

                # Pick a color based on the score
                if score > 0.8:
                    color = (0, 255, 0)  # green
                elif score > 0.5:
                    color = (0, 255, 255)  # yellow
                else:
                    color = (0, 0, 255)  # red

                # Draw bounding box
                cv2.rectangle(cv_image_with_boxes, (x1, y1), (x2, y2), color, 2)
                # Put label text above the box
                cv2.putText(
                    cv_image_with_boxes,
                    f"{label}: {score:.2f}",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        # Flag to track whether we show boxes or not
        show_boxes = True

        while True:
            # Display the appropriate image
            if show_boxes:
                cv2.imshow("Detections", cv_image_with_boxes)
            else:
                cv2.imshow("Detections", cv_image)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                # Close window
                break
            elif key == 32:  # space
                # Toggle showing bounding boxes
                show_boxes = not show_boxes

        cv2.destroyAllWindows()

    def show_image_with_detections_and_gt(
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
        ground_truth: List[ObjectDetectionResultI],
    ) -> None:
        # gt will be drawn in green
        # detections will be colored based on score (orange, yellow, red)
        # Convert PIL image to a NumPy array in OpenCV's BGR format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Make a copy to draw bounding boxes on
        cv_image_with_boxes = cv_image.copy()
        cv_image_with_gt = cv_image.copy()
        cv_image_with_preds = cv_image.copy()
        # Draw bounding boxes and labels on the copy
        for detection in detections:
            bbox = detection.as_xyxy()
            if bbox.shape[0] > 1:
                for i, box in enumerate(bbox):
                    x1, y1, x2, y2 = map(int, box)
                    score = detection.score[i].item()
                    label = str(detection.label[i].item())
                    # Choose a color and draw rectangle
                    if score > 0.8:
                        # orange
                        color = (0, 165, 255)
                    elif score > 0.5:
                        # yellow
                        color = (0, 255, 255)
                    else:
                        # red
                        color = (0, 0, 255)
                    
                    # Draw bounding box
                    cv2.rectangle(cv_image_with_boxes, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(cv_image_with_preds, (x1, y1), (x2, y2), color, 2)
                    # Put label text above the box
                    cv2.putText(
                        cv_image_with_boxes,
                        f"{label}: {score:.2f}",
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        cv_image_with_preds,
                        f"{label}: {score:.2f}",
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
            else:
                x1, y1, x2, y2 = map(int, bbox[0])
                score = detection.score
                label = str(detection.label)
                # Pick a color based on the score
                if score > 0.8:
                    # orange
                    color = (0, 165, 255)
                elif score > 0.5:
                    # yellow
                    color = (0, 255, 255)
                else:
                    # red
                    color = (0, 0, 255)

                # Draw bounding box
                cv2.rectangle(cv_image_with_boxes, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(cv_image_with_preds, (x1, y1), (x2, y2), color, 2)
                # Put label text above the box
                cv2.putText(
                    cv_image_with_boxes,
                    f"{label}: {score:.2f}",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    cv_image_with_preds,
                    f"{label}: {score:.2f}",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
        # Draw ground truth boxes in green
        for truth in ground_truth:
            bbox = truth.as_xyxy()
            if bbox.shape[0] > 1:
                for i, box in enumerate(bbox):
                    x1, y1, x2, y2 = map(int, box)
                    label = str(truth.label[i].item())
                    # Draw bounding box
                    cv2.rectangle(cv_image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(cv_image_with_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put label text above the box
                    cv2.putText(
                        cv_image_with_boxes,
                        f"{label}",
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        cv_image_with_gt,
                        f"{label}",
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
            else:
                x1, y1, x2, y2 = map(int, bbox[0])
                label = str(truth.label)
                # Draw bounding box
                cv2.rectangle(cv_image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(cv_image_with_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label text above the box
                cv2.putText(
                    cv_image_with_boxes,
                    f"{label}",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    cv_image_with_gt,
                    f"{label}",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        # Flag to track whether we show boxes or not
        show_boxes = True
        img_to_show = cv_image_with_boxes
        while True:
            # Display the appropriate image
            cv2.imshow("Detections", img_to_show)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                # Close window
                break
            elif key == 32:
                # Toggle showing bounding boxes
                show_boxes = not show_boxes
                if show_boxes:
                    img_to_show = cv_image_with_boxes
                else:
                    img_to_show = cv_image
            elif key == ord("g"):
                img_to_show = cv_image_with_gt
            elif key == ord("p"):
                img_to_show = cv_image_with_preds
        
        cv2.destroyAllWindows()


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
    ) -> List[List[Optional[ObjectDetectionResultI]]]:
        pass

    @abstractmethod
    def identify_for_image_as_tensor(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs,
    ) -> List[Optional[ObjectDetectionResultI]]:
        pass

    @abstractmethod
    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[Optional[ObjectDetectionResultI]]]:
        pass

    @abstractmethod
    def to(self, device: Union[str, torch.device]):
        pass
