from typing import Iterator, List, Union
from itertools import islice

import torch
from PIL import Image
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from scenic_reasoning.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionModelI,
    ObjectDetectionResultI,
)

class Detectron2Model(ObjectDetectionModelI):
    def __init__(self, config_file: str, weights_file: str, threshold: float = 0.5):
        # Input Detectron2 config file and weights file
        cfg = get_cfg()
        cfg.MODEL.WEIGHTS = weights_file
        cfg.merge_from_file(config_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self._predictor = DefaultPredictor(cfg)
        self._metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def identify_for_image(self, image, **kwargs) -> List[ObjectDetectionResultI]:
        """
        Run object detection on an image or a batch of images.

        Args:
            image: either a PIL image or a tensor of shape (B, C, H, W)
                where B is the batch size, C is the channel size, H is the
                height, and W is the width.

        Returns:
            A list of list of ObjectDetectionResultI, where the outer list
            represents the batch of images, and the inner list represents the
            detections in a particular image.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        predictions = self._predictor(image)
    
        if len(predictions) == 0:
            return None
        
        instances = predictions["instances"]

        formatted_results = []
        for i in range(len(instances)):
            box = instances.pred_boxes[i]
            score = instances.scores[i].item()
            cls_id = int(instances.pred_classes[i].item())
            label = self._metadata.thing_classes[cls_id]

            odr = ObjectDetectionResultI(
                score=score,
                cls=cls_id,
                label=label,
                bbox=box,
                image_hw=image.shape[:2],
                bbox_format=BBox_Format.XYXY,
            )

            formatted_results.append(odr)

        return formatted_results

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[List[ObjectDetectionResultI]]]:
        def batch_iterator(iterable, n):
            iterator = iter(iterable)
            return iter(lambda: list(islice(iterator, n)), [])

        if isinstance(video, list):
            video_iterator = batch_iterator(video, batch_size)
        else:
            video_iterator = batch_iterator(video, batch_size)

        for batch in video_iterator:
            if not batch: # End of iterator
                break

            batch_results = []
            for image in batch:
                image = np.array(image)
                predictions = self._predictor(image)
                instances = predictions["instances"]

                per_frame_results = []
                for i in range(len(instances)):
                    box = instances.pred_boxes[i]
                    score = instances.scores[i].item()
                    cls_id = int(instances.pred_classes[i].item())
                    label = self._metadata.thing_classes[cls_id]

                    odr = ObjectDetectionResultI(
                        score=score,
                        cls=cls_id,
                        label=label,
                        bbox=box,
                        image_hw=image.shape[:2],
                        bbox_format=BBox_Format.XYXY,
                    )

                    per_frame_results.append(odr)

                batch_results.append(per_frame_results)

            yield batch_results