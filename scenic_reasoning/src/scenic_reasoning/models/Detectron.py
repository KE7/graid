import cv2
import torch
from detectron2.utils.logger import setup_logger
from scenic_reasoning.utilities.common import get_default_device

setup_logger()
from itertools import islice
from typing import Iterator, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import BitMasks
from detectron2.utils.visualizer import Visualizer
from PIL import Image
from scenic_reasoning.interfaces.InstanceSegmentationI import (
    InstanceSegmentationResultI,
    Mask_Format,
)
from scenic_reasoning.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionModelI,
    ObjectDetectionResultI,
)
from scenic_reasoning.utilities.common import get_default_device


class Detectron_obj(ObjectDetectionModelI):
    def __init__(self, config_file: str, weights_file: str, threshold: float = 0.1):
        # Input Detectron2 config file and weights file
        cfg = get_cfg()
        cfg.MODEL.DEVICE = str(get_default_device())
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
        self.cfg = cfg
        self._predictor = DefaultPredictor(cfg)
        self._metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.model_name = config_file

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
                    # img_np = img.cpu().numpy()
                    batch_results.append(self._process_single_image(img_np))
                return batch_results

            elif image.ndimension() == 3:  # Single input (C, H, W)
                print(f"image should be CHW: {image.shape}")
                image = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC
                image = np.ascontiguousarray(
                    image
                )  # Ensure the array is contiguous in memory

        # Single image input
        print(f"image should be HWC: {image.shape}")
        return self._process_single_image(image)

    def _process_single_image(self, image: np.ndarray) -> List[ObjectDetectionResultI]:

        predictions = self._predictor(image)

        if len(predictions) == 0:
            print("Predictions were empty and not found in this image.")
            return []

        if "instances" not in predictions or len(predictions["instances"]) == 0:
            print("No instances or predictions in this image.")
            return []

        instances = predictions["instances"]

        if not hasattr(instances, "pred_boxes") or len(instances.pred_boxes) == 0:
            print("Prediction boxes attribute missing or not found in instances.")
            return []

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
        self, batched_images, **kwargs
    ) -> List[ObjectDetectionResultI]:
        assert (
            batched_images.ndimension() == 4
        ), "Input tensor must be of shape (B, C, H, W) in RGB format"
        batched_images = batched_images[:, [2, 1, 0], ...]  # Convert RGB to BGR
        list_of_images = []
        for i in range(batched_images.shape[0]):
            image = batched_images[i]
            image = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC
            image = self.aug.get_transform(image).apply_image(image)
            image = torch.as_tensor(
                image.astype("float32").transpose(2, 0, 1)
            )  # Convert back to CHW
            image.to(self.cfg.MODEL.DEVICE)
            height, width = image.shape[1:]
            list_of_images.append({"image": image, "height": height, "width": width})

        predictions = self.model(list_of_images)

        formatted_results = []
        for i in range(len(predictions)):
            img_result = []
            for j in range(len(predictions[i]["instances"])):
                box = (
                    predictions[i]["instances"][j]
                    .pred_boxes.tensor.cpu()
                    .numpy()
                    .tolist()[0]
                )
                score = predictions[i]["instances"][j].scores.item()
                cls_id = int(predictions[i]["instances"][j].pred_classes.item())
                label = self._metadata.thing_classes[cls_id]

                odr = ObjectDetectionResultI(
                    score=score,
                    cls=cls_id,
                    label=label,
                    bbox=box,
                    image_hw=(height, width),
                    bbox_format=BBox_Format.XYXY,
                )

                img_result.append(odr)

            formatted_results.append(img_result)

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
        pass

    def set_threshold(self, threshold: float):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    def __str__(self):
        return self.model_name.split("/")[-1].split(".")[0]


class Detectron2InstanceSegmentation:
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

    # new method for running the model for prediction
    def predict(self, image_tensor: torch.Tensor):
        image = image_tensor.permute(
            1, 2, 0
        ).numpy()  # Convert from (C, H, W) to (H, W, C)
        outputs = self.predictor(image)
        instances = outputs["instances"]
        results = []

        for idx in range(len(instances)):
            results.append(
                {
                    "score": instances.scores[idx].item(),
                    "cls": instances.pred_classes[idx].item(),
                    "bbox": instances.pred_boxes[idx].tensor.numpy()[
                        0
                    ],  # Convert bbox to array
                }
            )

        return results

    # method to see the segmentation on the image
    def visualize(self, im):  # image_tensor: torch.Tensor):
        # im = image_tensor.permute(1, 2, 0).numpy()
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], self.metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()

    def identify_for_tensor(
        self, image_tensor: torch.Tensor
    ) -> List[InstanceSegmentationResultI]:
        # Convert PyTorch tensor (C, H, W) to OpenCV format (H, W, C)
        image = image_tensor.permute(1, 2, 0).numpy()
        height, width = image.shape[:2]
        outputs = self.predictor(image)
        instances = outputs["instances"]
        results = []

        for idx in range(len(instances)):
            mask = BitMasks(instances.pred_masks[idx].unsqueeze(0))
            result = InstanceSegmentationResultI(
                score=instances.scores[idx].item(),
                cls=instances.pred_classes[idx].item(),
                label=self.metadata.thing_classes[instances.pred_classes[idx].item()],
                instance_id=idx,
                mask=mask,
                image_hw=(height, width),
                mask_format=Mask_Format.BITMASK,
            )
            results.append(result)

        return results
