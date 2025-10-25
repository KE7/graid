from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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


class BBox_Format(Enum):
    XYWH = 0
    XYWHN = 1
    XYXY = 2
    XYXYN = 3
    Detectron2Box = 4


class ObjectDetectionResultI:
    def __init__(
        self,
        score: Union[float, torch.Tensor],
        cls: Union[int, torch.Tensor],
        label: Union[str, torch.Tensor],
        bbox: Union[Union[torch.Tensor, List[float]], Detectron2Boxes],
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
        self._image_hw = image_hw

        # Initialize _detectron2_boxes with proper type
        self._detectron2_boxes: Detectron2Boxes

        if isinstance(bbox, Detectron2Boxes):
            self._detectron2_boxes = bbox
        elif isinstance(bbox, torch.Tensor):
            # should have shape (# of boxes, 4) or (# of boxes, 6) where each row is:
            #   (x1, y1, x2, y2) or (x1, y1, x2, y2, score, cls)

            assert (
                bbox_format == BBox_Format.XYXY
            ), "Only XYXY format supported for tensor input"

            if (
                bbox.shape[1] == 4
                and isinstance(score, torch.Tensor)
                and isinstance(cls, torch.Tensor)
            ):
                # Use the provided score and cls tensors
                pass
            elif bbox.shape[1] == 4:
                # Check if we have single box with scalar score/cls (common for flatten)
                if (
                    bbox.shape[0] == 1
                    and isinstance(score, (int, float))
                    and isinstance(cls, (int, str))
                ):
                    # This is okay - single box with scalar values
                    pass
                elif not isinstance(score, torch.Tensor) or not isinstance(
                    cls, torch.Tensor
                ):
                    raise ValueError(
                        f"Tried to initialize DetectionResult with {bbox.shape[0]} many "
                        "bounding boxes but only a single score and class provided."
                    )
            elif bbox.shape[1] == 6 or bbox.shape[1] == 7:
                self._score = bbox[:, 4]
                self._class = bbox[:, 5]
            elif bbox.shape[1] != 4 and bbox.shape[1] != 6 and bbox.shape[1] != 7:
                raise ValueError(
                    f"{bbox.shape[1]} not supported for initializing DetectionResult"
                    " (should be 4, 6 or 7)"
                )

            # Extract just the bbox coordinates (first 4 columns)
            bbox_coords = bbox[:, :4] if bbox.shape[1] > 4 else bbox

            # all x1 < x2 and y1 < y2
            if torch.any(bbox_coords[:, 0] > bbox_coords[:, 2]) or torch.any(
                bbox_coords[:, 1] > bbox_coords[:, 3]
            ):
                raise ValueError(
                    f"Bounding box coordinates are not in the correct format. "
                    "All x1 < x2 and y1 < y2 but found some boxes with x1 > x2 or y1 > y2"
                )
            self._detectron2_boxes = Detectron2Boxes(bbox_coords)
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

            box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            self._detectron2_boxes = Detectron2Boxes(box)

    def as_xyxy(self) -> torch.Tensor:
        """Get bounding boxes in XYXY format (x1, y1, x2, y2)."""
        return self._detectron2_boxes.tensor

    def as_xyxyn(self) -> torch.Tensor:
        """Get bounding boxes in normalized XYXY format (x1, y1, x2, y2) divided by image dimensions."""
        boxes = self._detectron2_boxes.tensor
        img_h, img_w = self._image_hw
        normalized = boxes.clone()
        normalized[:, [0, 2]] /= img_w  # normalize x coordinates
        normalized[:, [1, 3]] /= img_h  # normalize y coordinates
        return normalized

    def as_xywh(self) -> torch.Tensor:
        """Get bounding boxes in XYWH format (x, y, width, height) where x,y is top-left corner."""
        boxes = self._detectron2_boxes.tensor
        x1, y1, x2, y2 = boxes.split((1, 1, 1, 1), dim=-1)
        # COCO format: x, y (top-left corner), width, height
        xywh = torch.cat([x1, y1, (x2 - x1), (y2 - y1)], dim=-1)
        return xywh

    def as_xywhn(self) -> torch.Tensor:
        """Get bounding boxes in normalized XYWH format (x, y, width, height) divided by image dimensions."""
        xywh = self.as_xywh()
        img_h, img_w = self._image_hw
        normalized = xywh.clone()
        normalized[:, [0, 2]] /= img_w  # normalize x coordinates and width
        normalized[:, [1, 3]] /= img_h  # normalize y coordinates and height
        return normalized

    def flatten(self) -> List["ObjectDetectionResultI"]:
        """
        If the current detection result is a single instance, it returns itself
        Otherwise, it returns a list of detection results based on the n boxes
        that were in the original detection result.
        """
        if isinstance(self._score, float):
            return [self]
        elif isinstance(self._score, torch.Tensor):
            boxes = self._detectron2_boxes.tensor
            return [
                ObjectDetectionResultI(
                    score=float(self._score[i]),
                    cls=(
                        int(self._class[i])
                        if isinstance(self._class, torch.Tensor)
                        else self._class
                    ),
                    label=(
                        str(self._label[i])
                        if isinstance(self._label, torch.Tensor)
                        else self._label
                    ),
                    bbox=self._detectron2_boxes.tensor[
                        i : i + 1
                    ],  # Extract tensor directly
                    image_hw=self._image_hw,
                    bbox_format=BBox_Format.XYXY,
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
        # and the bbox should have shape (1, 4)
        if isinstance(self._score, float):
            assert isinstance(
                self._class, int
            ), "Single instance detection result does not have a single int class"
            assert isinstance(
                self._label, str
            ), "Single instance detection result does not have a single string label"
            assert self._detectron2_boxes.tensor.shape == (
                1,
                4,
            ), "Single instance detection result does not have a single bounding box"
        elif isinstance(self._score, torch.Tensor):
            assert isinstance(
                self._class, torch.Tensor
            ), "Tensor detection result does not have a tensor class"
            assert isinstance(
                self._label, torch.Tensor
            ), "Tensor detection result does not have a tensor label"
            assert self._detectron2_boxes.tensor.shape == (
                self._score.shape[0],
                4,
            ), "Tensor detection result does not have a tensor bounding box in the correct format"
            assert (
                self._class.shape == self._score.shape == self._label.shape
            ), "Tensor detection result does not have matching sizes for scores, classes, and labels"
            assert (
                self._detectron2_boxes.tensor.shape[0] == self._score.shape[0]
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
    def as_detectron_box(self) -> Detectron2Boxes:
        self._check_self_consistency()
        return self._detectron2_boxes

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
        metric: Optional[MeanAveragePrecision] = None,
        class_metrics: bool = False,
        extended_summary: bool = False,
        debug: bool = False,
        image: Optional[torch.Tensor] = None,
        penalize_for_extra_predicitions: bool = False,
    ) -> Dict[Any, Any]:

        pred_boxes = []
        pred_scores = []
        pred_classes = []

        for i, pred in enumerate(predictions):
            pred_boxes.append(pred.as_xyxy())
            pred_scores.append(pred.score)
            pred_classes.append(pred.cls)

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
        for truth in ground_truth:
            boxes.append(truth.as_xyxy())
            scores.append(truth.score)
            classes.append(truth.cls)

        if penalize_for_extra_predicitions:
            # if there are more predictions than ground truth, add fake boxes
            # so that we will end up lowering the mAP
            num_ghost_boxes = max(0, len(pred_boxes) - len(boxes))
            if image is not None:
                image_size = (image.shape[1], image.shape[0])
            else:
                # Default fallback image size
                image_size = (640, 480)
            ghost_bbox_size = 20
            for i in range(num_ghost_boxes):
                x1 = i * ghost_bbox_size
                y1 = ghost_bbox_size
                x2 = x1 + ghost_bbox_size
                y2 = y1 + ghost_bbox_size

                ghost_bbox = [float(x1), float(y1), float(x2), float(y2)]
                fake_score = 1.0
                nonsignificant_class = -1
                nonexistent_label = "fake"

                fake_detection = ObjectDetectionResultI(
                    score=fake_score,
                    cls=nonsignificant_class,
                    label=nonexistent_label,
                    bbox=ghost_bbox,
                    image_hw=image_size,
                )
                ground_truth.append(fake_detection)

                boxes.append(fake_detection.as_xyxy())
                scores.append(fake_detection.score)
                classes.append(fake_detection.cls)

        boxes = torch.cat(boxes) if boxes else torch.Tensor([])  # shape: (num_boxes, 4)
        scores = (
            (
                torch.tensor(scores)
                if isinstance(scores[0], float)
                else torch.cat(scores)
            )
            if scores
            else torch.Tensor([])
        )
        classes = (
            (
                torch.tensor(classes)
                if isinstance(classes[0], int)
                else torch.cat(classes)
            )
            if classes
            else torch.Tensor([])
        )

        targets: List[Dict[str, torch.Tensor]] = [
            dict(boxes=boxes, labels=classes, scores=scores)
        ]

        # TODO: This should be pulled out and the caller should pass it in
        # so that we can avoid the memory leak issue:
        # https://github.com/Lightning-AI/torchmetrics/issues/1949
        need_to_delete_metric = False
        if metric is None:
            metric = MeanAveragePrecision(
                class_metrics=class_metrics,
                extended_summary=extended_summary,
                box_format="xyxy",
                iou_type="bbox",
            )
            need_to_delete_metric = True

        # We only need to call update
        metric.update(target=targets, preds=preds)

        # once we are at the end is when we call compute
        # if need_to_delete_metric:
        score = metric.compute()
        score["TN"] = 0
        for p, t in zip(preds, targets):
            if p["boxes"].shape == torch.Size([0]) and t["boxes"].shape == torch.Size(
                [0]
            ):
                score["TN"] += 1

        # ret = max(0, score['map'].item())
        ret = score["map"].item()
        print("!!!!!!!!!!!!!!!!!!", ret)

        metric.reset()
        del metric
        del score

        # print(score['map'], "!!!!!!!!!!!!!!!!")
        return ret

        # else:
        #     tn = 0
        #     for p, t in zip(preds, targets):
        #         if p["boxes"].shape == torch.Size([0]) and t[
        #             "boxes"
        #         ].shape == torch.Size([0]):
        #             tn += 1
        #     return {"TN": tn}

    @staticmethod
    def normalize_detections(
        detections: List[ObjectDetectionResultI],
        bbox_format: BBox_Format = BBox_Format.XYXY,
    ) -> Dict[str, Any]:
        """
        Normalize a mixed list of detection results into per-box, tensor-safe structures.

        Returns a dictionary with:
          - detections: List[ObjectDetectionResultI] (one per box)
          - labels: List[str]
          - bboxes_xyxy: torch.Tensor of shape (N, 4)
          - bbox_list: List[Dict[str, float]] [{'x1','y1','x2','y2'}]
          - counts: Dict[str, int] class → count
        """
        # Flatten into one detection per box
        flattened: List[ObjectDetectionResultI] = []
        for det in detections:
            flattened.extend(det.flatten())

        labels: List[str] = []
        boxes_xyxy: List[torch.Tensor] = []
        counts: Dict[str, int] = {}

        for det in flattened:
            # Label as string
            lbl = det.label
            lbl_str = (
                str(lbl)
                if isinstance(lbl, (str, int, float))
                else str(lbl.item()) if hasattr(lbl, "item") else str(lbl)
            )
            labels.append(lbl_str)
            counts[lbl_str] = counts.get(lbl_str, 0) + 1

            # Bbox in XYXY first 4 coords
            if bbox_format == BBox_Format.XYXY:
                xyxy = det.as_xyxy()
            elif bbox_format == BBox_Format.XYWH:
                xyxy = det.as_xywh()
            elif bbox_format == BBox_Format.XYWHN:
                xyxy = det.as_xywhn()
            elif bbox_format == BBox_Format.XYXYN:
                xyxy = det.as_xyxyn()
            else:
                # Default to xyxy
                xyxy = det.as_xyxy()

            # Ensure shape (4,) tensor for this single detection
            if xyxy.dim() == 2:
                # expected (1, 6) or (1, 4+) layout from UltralyticsBoxes
                coords = xyxy[0][:4]
            else:
                coords = xyxy[:4]
            boxes_xyxy.append(coords)

        # Stack xyxy to (N, 4)
        bxyxy = (
            torch.stack(boxes_xyxy)
            if boxes_xyxy
            else torch.empty((0, 4), dtype=torch.float32)
        )

        # Generate list-of-dicts format commonly used when writing out
        bbox_list: List[Dict[str, float]] = [
            {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])}
            for b in bxyxy
        ]

        return {
            "detections": flattened,
            "labels": labels,
            "bboxes_xyxy": bxyxy,
            "bbox_list": bbox_list,
            "counts": counts,
        }

    @staticmethod
    def build_question_context(
        image: Optional[Union[np.ndarray, torch.Tensor, Image.Image]],
        detections: List[ObjectDetectionResultI],
    ) -> Dict[str, Any]:
        """Precompute per-image features for questions to avoid recomputation.

        Returns a dictionary with:
          - detections, labels, bboxes_xyxy, bbox_list, counts (from normalize_detections)
          - centers: Tensor (N,2)
          - areas: Tensor (N,)
          - aspects: Tensor (N,) width/height
          - class_to_indices: Dict[str, List[int]]
        """
        norm = ObjectDetectionUtils.normalize_detections(detections)
        bxyxy: torch.Tensor = norm["bboxes_xyxy"]
        if bxyxy.numel() > 0:
            widths = (bxyxy[:, 2] - bxyxy[:, 0]).clamp(min=1.0)
            heights = (bxyxy[:, 3] - bxyxy[:, 1]).clamp(min=1.0)
            centers = torch.stack(
                [(bxyxy[:, 0] + bxyxy[:, 2]) / 2.0, (bxyxy[:, 1] + bxyxy[:, 3]) / 2.0],
                dim=1,
            )
            areas = widths * heights
            aspects = widths / heights
        else:
            centers = torch.empty((0, 2), dtype=torch.float32)
            areas = torch.empty((0,), dtype=torch.float32)
            aspects = torch.empty((0,), dtype=torch.float32)

        class_to_indices: Dict[str, List[int]] = {}
        for idx, lbl in enumerate(norm["labels"]):
            class_to_indices.setdefault(lbl, []).append(idx)

        ctx = {
            **norm,
            "centers": centers,
            "areas": areas,
            "aspects": aspects,
            "class_to_indices": class_to_indices,
            "image": image,
        }
        return ctx



    @staticmethod
    def show_image_with_detections(
        image: Image.Image, detections: List[ObjectDetectionResultI]
    ) -> None:
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
                    score = (
                        detection.score[i].item()
                        if isinstance(detection.score, torch.Tensor)
                        else detection.score
                    )
                    label = str(
                        detection.label[i]
                        if isinstance(detection.label, torch.Tensor)
                        else detection.label
                    )

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

    @staticmethod
    def show_image_with_detections_and_gt(
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
        ground_truth: List[ObjectDetectionResultI],
    ):
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
                    score = (
                        detection.score[i].item()
                        if isinstance(detection.score, torch.Tensor)
                        else detection.score
                    )
                    label = str(
                        detection.label[i]
                        if isinstance(detection.label, torch.Tensor)
                        else detection.label
                    )
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
                    label = str(
                        truth.label[i]
                        if isinstance(truth.label, torch.Tensor)
                        else truth.label
                    )
                    # Draw bounding box
                    cv2.rectangle(
                        cv_image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
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

        return img_to_show


class ObjectDetectionModelI(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def identify_for_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        debug: bool = False,
    ) -> List[List[ObjectDetectionResultI]]:
        pass

    @abstractmethod
    def identify_for_image_batch(
        self,
        image: Union[np.ndarray, torch.Tensor],
        debug: bool = False,
        **kwargs,
    ) -> List[List[ObjectDetectionResultI]]:
        pass

    @abstractmethod
    def identify_for_video(
        self,
        video: Union[
            Iterator[Union[np.ndarray, torch.Tensor]],
            List[Union[np.ndarray, torch.Tensor]],
        ],
        batch_size: int = 1,
    ) -> Iterator[List[Optional[ObjectDetectionResultI]]]:
        pass

    @abstractmethod
    def to(self, device: Union[str, torch.device]):
        pass
