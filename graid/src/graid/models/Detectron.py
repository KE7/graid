from itertools import islice
from pathlib import Path
from typing import Iterator, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import BitMasks
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from PIL import Image

from graid.interfaces.InstanceSegmentationI import (
    InstanceSegmentationModelI,
    InstanceSegmentationResultI,
    Mask_Format,
)
from graid.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionModelI,
    ObjectDetectionResultI,
)
from graid.utilities.common import (
    convert_batch_to_numpy,
    convert_image_to_numpy,
    get_default_device,
)

setup_logger()


class DetectronBase:
    """Base class for Detectron2 models with shared functionality."""

    def __init__(
        self,
        config_file: str,
        weights_file: str,
        threshold: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        # Input Detectron2 config file and weights file
        cfg = get_cfg()
        cfg.MODEL.DEVICE = str(get_default_device()) if device is None else str(device)
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
        self.cfg = cfg
        self._predictor = DefaultPredictor(cfg)
        self._metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.model_name = config_file
        self.threshold = threshold  # Store threshold for reference

    def to(self, device: Union[str, torch.device]):
        """Move model to specified device."""
        # Update config device
        self.cfg.MODEL.DEVICE = str(device)
        # Recreate predictor with new device
        self._predictor = DefaultPredictor(self.cfg)

    def set_threshold(self, threshold: float):
        """Set confidence threshold for detections."""
        self.threshold = threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        # Recreate predictor with new threshold
        self._predictor = DefaultPredictor(self.cfg)

    def __str__(self):
        return self.model_name.split("/")[-1].split(".")[0]


class Detectron_obj(DetectronBase, ObjectDetectionModelI):
    def __init__(
        self,
        config_file: str,
        weights_file: str,
        threshold: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(config_file, weights_file, threshold, device)

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
        image = convert_image_to_numpy(image)

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
                # Ensure the array is contiguous in memory
                image = np.ascontiguousarray(image)

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

    def identify_for_image_batch(
        self, batched_images, debug: bool = False, **kwargs
    ) -> List[ObjectDetectionResultI]:
        assert (
            batched_images.ndimension() == 4
        ), "Input tensor must be of shape (B, C, H, W) in RGB format"
        batched_images = batched_images[:, [2, 1, 0], ...]  # Convert RGB to BGR
        list_of_images = []
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            for i in range(batched_images.shape[0]):
                image = batched_images[i]
                image = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC
                image = self._predictor.aug.get_transform(image).apply_image(image)
                image = torch.as_tensor(
                    image.astype("float32").transpose(2, 0, 1)
                )  # Convert back to CHW
                image = image.to(self.cfg.MODEL.DEVICE).detach()
                height, width = image.shape[1:]
                list_of_images.append(
                    {"image": image, "height": height, "width": width}
                )

            predictions = self._predictor.model(list_of_images)

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

        return formatted_results

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
                image = convert_image_to_numpy(image)
                frame_results = self._process_single_image(image)
                batch_results.append(frame_results)

            yield batch_results


class Detectron_seg(DetectronBase, InstanceSegmentationModelI):
    def __init__(
        self,
        config_file: str,
        weights_file: str,
        threshold: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(config_file, weights_file, threshold, device)

    def identify_for_image(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs,
    ) -> List[InstanceSegmentationResultI]:
        """
        Run instance segmentation on an image.
        Args:
            image: Input image as PIL Image, numpy array, or tensor
        Returns:
            A list of InstanceSegmentationResultI objects
        """
        image = convert_image_to_numpy(image)
        return self._process_single_image(image)

    def _process_single_image(
        self, image: np.ndarray
    ) -> List[InstanceSegmentationResultI]:
        """Process a single image for instance segmentation."""
        predictions = self._predictor(image)

        if len(predictions) == 0:
            print("Predictions were empty and not found in this image.")
            return []

        if "instances" not in predictions or len(predictions["instances"]) == 0:
            print("No instances or predictions in this image.")
            return []

        instances = predictions["instances"]

        if not hasattr(instances, "pred_masks") or len(instances.pred_masks) == 0:
            print("Prediction masks attribute missing or not found in instances.")
            return []

        formatted_results = []
        height, width = image.shape[:2]

        for i in range(len(instances)):
            score = instances.scores[i].item()
            cls_id = int(instances.pred_classes[i].item())
            label = self._metadata.thing_classes[cls_id]
            mask = instances.pred_masks[i]

            # Create BitMasks object from the mask tensor
            bitmask = BitMasks(mask.unsqueeze(0))

            result = InstanceSegmentationResultI(
                score=score,
                cls=cls_id,
                label=label,
                instance_id=i,
                mask=bitmask,
                image_hw=(height, width),
                mask_format=Mask_Format.BITMASK,
            )

            formatted_results.append(result)

        return formatted_results

    def identify_for_image_batch(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs,
    ) -> List[List[InstanceSegmentationResultI]]:
        """
        Run instance segmentation on a batch of images.
        Args:
            image: Batched images as tensor of shape (B, C, H, W)
        Returns:
            A list of lists of InstanceSegmentationResultI objects
        """

        if isinstance(image, torch.Tensor):
            assert (
                image.ndimension() == 4
            ), "Input tensor must be of shape (B, C, H, W) in RGB format"

            # Convert RGB to BGR and prepare images for model
            batched_images = image[:, [2, 1, 0], ...]  # Convert RGB to BGR
            list_of_images = []

            with torch.no_grad():
                for i in range(batched_images.shape[0]):
                    img = batched_images[i]
                    img = img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC

                    # Apply preprocessing transformations from predictor
                    img = self._predictor.aug.get_transform(img).apply_image(img)
                    img = torch.as_tensor(
                        img.astype("float32").transpose(2, 0, 1)
                    )  # Convert back to CHW
                    img = img.to(self.cfg.MODEL.DEVICE).detach()

                    height, width = img.shape[1:]
                    list_of_images.append(
                        {"image": img, "height": height, "width": width}
                    )

                # Process entire batch through model at once
                predictions = self._predictor.model(list_of_images)

            # Format results for each image in batch
            formatted_results = []
            for i in range(len(predictions)):
                img_results = []

                if (
                    "instances" not in predictions[i]
                    or len(predictions[i]["instances"]) == 0
                ):
                    formatted_results.append(img_results)
                    continue

                instances = predictions[i]["instances"]

                if (
                    not hasattr(instances, "pred_masks")
                    or len(instances.pred_masks) == 0
                ):
                    formatted_results.append(img_results)
                    continue

                height = list_of_images[i]["height"]
                width = list_of_images[i]["width"]

                for j in range(len(instances)):
                    score = instances.scores[j].item()
                    cls_id = int(instances.pred_classes[j].item())
                    label = self._metadata.thing_classes[cls_id]
                    mask = instances.pred_masks[j]

                    # Create BitMasks object from the mask tensor
                    bitmask = BitMasks(mask.unsqueeze(0))

                    result = InstanceSegmentationResultI(
                        score=score,
                        cls=cls_id,
                        label=label,
                        instance_id=j,
                        mask=bitmask,
                        image_hw=(height, width),
                        mask_format=Mask_Format.BITMASK,
                    )

                    img_results.append(result)

                formatted_results.append(img_results)

            return formatted_results
        else:
            # Single image case
            return [self.identify_for_image(image, debug=debug, **kwargs)]

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[InstanceSegmentationResultI]]:
        """
        Run instance segmentation on a video represented as iterator/list of images.
        Args:
            video: An iterator or list of PIL images
            batch_size: Number of images to process at a time
        Returns:
            An iterator of lists of InstanceSegmentationResultI objects
        """

        def batch_iterator(iterable, n):
            iterator = iter(iterable)
            return iter(lambda: list(islice(iterator, n)), [])

        video_iterator = batch_iterator(video, batch_size)

        for batch in video_iterator:
            if not batch:  # End of iterator
                break

            for image in batch:
                image = convert_image_to_numpy(image)
                frame_results = self._process_single_image(image)
                yield frame_results

    def visualize(self, image: Union[np.ndarray, torch.Tensor]):
        """Visualize segmentation results on an image."""
        image = convert_image_to_numpy(image)
        outputs = self._predictor(image)
        v = Visualizer(image[:, :, ::-1], self._metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()
