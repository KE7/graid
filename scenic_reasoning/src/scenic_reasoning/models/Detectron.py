from itertools import islice
from typing import Iterator, List, Union

import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from PIL import Image
from scenic_reasoning.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionModelI,
    ObjectDetectionResultI,
)
from scenic_reasoning.utilities.common import get_default_device


class Detectron2Model(ObjectDetectionModelI):
    def __init__(self, config_file: str, weights_file: str, threshold: float = 0.5):
        # Input Detectron2 config file and weights file
        cfg = get_cfg()
        cfg.MODEL.DEVICE = str(get_default_device())
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
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

        # TODO: Detectron2 predictor does not support batched inputs
        #  so either we loop through the batch or we do the preprocessing steps
        #  of the predictor ourselves and then call the model
        #  I prefer the latter approach. Preprocessing steps are in the predictor:
        #   - load the checkpoint
        #   - take the image in BGR format and apply conversion defined by cfg.INPUT.FORMAT
        #   - resize the image
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

    def identify_for_image_as_tensor(
        self, image, **kwargs
    ) -> List[ObjectDetectionResultI]:
        raise NotImplementedError

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
            if not batch:  # End of iterator
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

    def to(self, device: Union[str, torch.device]):
        raise RuntimeError(
            "Moving devices is an unsupported " "operation for the Detectron2 library"
        )
