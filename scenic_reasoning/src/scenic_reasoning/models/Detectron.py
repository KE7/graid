from itertools import islice
import cv2
import torch
from pathlib import Path
from detectron2.utils.logger import setup_logger
from scenic_reasoning.utilities.common import get_default_device
from PIL import Image
from typing import Iterator, List, Optional, Union

setup_logger()
from typing import List, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import BitMasks
from detectron2.utils.visualizer import Visualizer
from scenic_reasoning.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionModelI,
    ObjectDetectionResultI,
)
from scenic_reasoning.interfaces.InstanceSegmentationI import (
    InstanceSegmentationResultI,
    Mask_Format,
    InstanceSegmentationModelI
)

class Detectron2(ObjectDetectionModelI):
    def __init__(self,
        config_file: str,
        weights_file: str,
        device: Optional[str] = None,
        threshold: float = 0.1,):
        cfg = get_cfg()
        if device is None:
            device = get_default_device()
        cfg.MODEL.DEVICE = device
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self._model = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    
    def identify_for_image(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
    ) -> List[List[Optional[ObjectDetectionResultI]]]:
        """
        Run object detection on a single image.

        Args:
            image: A PIL image or a tensor of shape (C, H, W).

        Returns:
            A list of list of ObjectDetectionResultI, where each inner list represents
            detections in a single image.
        """
        if isinstance(image, torch.Tensor):
            image = image.permute(
            1, 2, 0
            ).numpy()  # Convert from (C, H, W) to (H, W, C)

        if isinstance(image, Image.Image):
            image = np.array(Image)
        outputs = self._model(image)
        instances = outputs["instances"]
        labels = instances.pred_classes
        scores = instances.scores
        boxes = instances.pred_boxes.tensor

        results = []
        if len(boxes) > 0:
            for i in range(len(boxes)):
                attributes = {}

                object_detection_result = ObjectDetectionResultI(
                    score=float(scores[i]),
                    cls=int(labels[i]),
                    label=str(labels[i]),
                    bbox=boxes[i].tolist(),
                    image_hw=image.shape[1:],
                    bbox_format=BBox_Format.XYXY,
                    attributes=attributes,
                )
                results.append(object_detection_result)
        return [results]

    def identify_for_image_as_tensor(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs,
    ) -> List[Optional[ObjectDetectionResultI]]:
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
        if not isinstance(image, list) and not isinstance(image, tuple):
            return self._identify_for_image(image, debug, **kwargs)
        # input is a batch of images
        results = []
        for img in image:
            res = self.identify_for_image(img, debug, **kwargs)
            results.append(res)
        return results

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[Optional[ObjectDetectionResultI]]]:
        def batch_iterator(iterable, n):
            iterator = iter(iterable)
            return iter(lambda: list(islice(iterator, n)), [])

        # Convert video to iterator of batches
        if isinstance(video, list):
            video_iterator = batch_iterator(video, batch_size)
        else:
            video_iterator = batch_iterator(video, batch_size)

        for batch in video_iterator:
            batch_results = []
            for frame in batch:
                frame_results = self.identify_for_image(frame)
                batch_results.append(frame_results)
            yield batch_results

    def to(self, device: Union[str, torch.device]):
        pass

class Detectron2InstanceSegmentation(InstanceSegmentationModelI):
    def __init__(
        self,
        config_file: str,
        weights_file: str,
        device: Optional[str] = None,
        threshold: float = 0.1,
    ):
        cfg = get_cfg()
        if device is None:
            device = get_default_device()
        cfg.MODEL.DEVICE = device
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def identify_for_image(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
    ) -> List[List[Optional[InstanceSegmentationResultI]]]:
        if isinstance(image, Image.Image):
            image = np.array(image)

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy() # Convert from (C, H, W) to (H, W, C)

        outputs = self.predictor(image)
        instances = outputs["instances"]
        results = []

        for idx in range(len(instances)):
            mask = BitMasks(instances.pred_masks[idx].unsqueeze(0))
            result = InstanceSegmentationResultI(
                score=instances.scores[idx].item(),
                cls=instances.pred_classes[idx].item(),
                label=self.metadata.thing_classes[
                    instances.pred_classes[idx].item()
                ],
                instance_id=idx,
                mask=mask,
                image_hw=image.shape[:2],
                mask_format=Mask_Format.BITMASK,
            )
            results.append(result)

        return [results]

    def identify_for_image_as_tensor(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs,
    ) -> List[Optional[InstanceSegmentationResultI]]:
        if isinstance(image, Image.Image):
            image = np.array(image)

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # # Convert from (C, H, W) to (H, W, C)

        outputs = self.predictor(image)
        instances = outputs["instances"]
        results = []

        for idx in range(len(instances)):
            mask = BitMasks(instances.pred_masks[idx].unsqueeze(0))
            result = InstanceSegmentationResultI(
                score=instances.scores[idx].item(),
                cls=instances.pred_classes[idx].item(),
                label=self.metadata.thing_classes[
                    instances.pred_classes[idx].item()
                ],
                instance_id=idx,
                mask=mask,
                image_hw=image.shape[:2],
                mask_format=Mask_Format.BITMASK,
            )
            results.append(result)

        return results

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[List[InstanceSegmentationResultI]]]:
        def batch_iterator(iterable, n):
            iterator = iter(iterable)
            return iter(lambda: list(islice(iterator, n)), [])

        if isinstance(video, list):
            video_iterator = batch_iterator(video, batch_size)
        else:
            video_iterator = batch_iterator(video, batch_size)

        for batch in video_iterator:
            if not batch:
                break

            images = [np.array(img) for img in batch]
            batch_results = []

            for image in images:
                outputs = self.predictor(image)
                instances = outputs["instances"]
                frame_results = []

                for idx in range(len(instances)):
                    mask = BitMasks(instances.pred_masks[idx].unsqueeze(0))
                    result = InstanceSegmentationResultI(
                        score=instances.scores[idx].item(),
                        cls=instances.pred_classes[idx].item(),
                        label=self.metadata.thing_classes[
                            instances.pred_classes[idx].item()
                        ],
                        instance_id=idx,
                        mask=mask,
                        image_hw=image.shape[:2],
                        mask_format=Mask_Format.BITMASK,
                    )
                    frame_results.append(result)

                batch_results.append(frame_results)

            yield batch_results
    
    # method to see the segmentation on the image
    def visualize(self, im):  # image_tensor: torch.Tensor):
        # im = image_tensor.permute(1, 2, 0).numpy()
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], self.metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()
    
    def to(self, device: Union[str, torch.device]):
        pass
