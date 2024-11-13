from abc import ABC
from enum import Enum
from typing import Dict, Iterator, List, Tuple, Union

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
        self._labels = label
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
                    f"{bbox_format} not supported for initalizing DetectionResult"
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
    def labels(self):
        return self._labels

    @property
    def cls(self):
        return self._class

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


class ObjectDetectionModelI(ABC):
    def __init__(self):
        pass

    def identify_for_image(self, image : Union[Image.Image, torch.Tensor, List[torch.Tensor]]) -> List[ObjectDetectionResultI]:
        pass

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[ObjectDetectionResultI]]:
        pass