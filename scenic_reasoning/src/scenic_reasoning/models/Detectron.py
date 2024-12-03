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
    def __init__(self, config_file: str, weights_file: str, threshold: float = 0.1):
        # Input Detectron2 config file and weights file
        cfg = get_cfg()
        cfg.MODEL.DEVICE = str(get_default_device())
        cfg.MODEL.DEVICE = "cpu"            # TODO: for debugging purposes, change to above later
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
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

        if isinstance(image, torch.Tensor):
            # Convert to HWC (Numpy format) if image is Pytorch tensor in CHW format
            if image.ndimension() == 4:  # Batched input (B, C, H, W)
                batch_results = []
                for img in image:
                    img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC 
                    batch_results.append(self._process_single_image(img_np))
                return batch_results
            
            elif image.ndimension() == 3:  # Single input (C, H, W)
                print(f"image should be CHW: {image.shape}")
                image = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC
                image = np.ascontiguousarray(image)  # Ensure the array is contiguous in memory

        # Single image input
        print(f"image should be HWC: {image.shape}")
        return self._process_single_image(image)
    
    def _process_single_image(self, image: np.ndarray) -> List[ObjectDetectionResultI]:
        print(f"Image to predict: {image.shape}")
        predictions = self._predictor(image)
        print(f"Predictions: {predictions}")

        if len(predictions) == 0:
            print("Predictions were empty and not found in this image.")
            return [[None]]
        
        if "instances" not in predictions or len(predictions["instances"]) == 0:
            print("No instances or predictions in this image.")
            return [[None]]

        instances = predictions["instances"]

        if not hasattr(instances, "pred_boxes") or len(instances.pred_boxes) == 0:
            print("Prediction boxes attribute missing or not found in instances.")
            return [[None]]

        formatted_results = []
        for i in range(len(instances)):
            box = instances.pred_boxes[i].tensor.cpu().numpy().tolist()[0]
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
        """
        Run object detection on a video represented as an iterator or list of images.

        Args:
            video: An iterator or list of PIL images.
            batch_size: Number of images to process at a time.

        Returns:
            An iterator of lists of lists of ObjectDetectionResultI, where the outer
            list represents the batches, the middle list represents frames, and the
            inner list represents detections within a frame.
        """
        def batch_iterator(iterable, n):
            iterator = iter(iterable)
            return iter(lambda: list(islice(iterator, n)), [])

        video_iterator = batch_iterator(video, batch_size)

        for batch in video_iterator:
            if not batch:  # End of iterator
                break

            batch_results = []
            for image in batch:
                if isinstance(image, Image.Image):
                    image = np.array(image)

                frame_results = self._process_single_image(image)
                batch_results.append(frame_results)

            yield batch_results

    def to(self, device: Union[str, torch.device]):
        raise RuntimeError(
            "Moving devices is an unsupported " "operation for the Detectron2 library"
        )
