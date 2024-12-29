from itertools import islice
from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np
from scenic_reasoning.src.scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionResultI
import torch
from mmdet.apis import inference_detector, init_detector
from PIL import Image
from scenic_reasoning.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionModelI,
    ObjectDetectionResultI,
)
from scenic_reasoning.utilities.common import get_default_device
from scenic_reasoning.interfaces.InstanceSegmentationI import (
    Mask_Format,
    InstanceSegmentationModelI,
    InstanceSegmentationResultI,
)

class MMDetection(ObjectDetectionModelI):
    def __init__(
        self, config: str, checkpoint: str, device: Optional[str] = None
    ) -> None:
        if device is None:
            device = get_default_device()
        self._model = init_detector(config, checkpoint, device=device)

    def identify_for_image(
        self, image: Union[Image.Image, torch.Tensor], **kwargs
    ) -> List[List[ObjectDetectionResultI]]:
        """
        Run object detection on a single image.

        Args:
            image: A PIL image or a tensor of shape (C, H, W).

        Returns:
            A list of list of ObjectDetectionResultI, where each inner list represents
            detections in a single image.
        """
        result = inference_detector(self._model, image)

        formatted_results = []
        if len(result.pred_instances) > 0:
            boxes = result.pred_instances.bboxes
            scores = result.pred_instances.scores
            labels = result.pred_instances.labels

            result_for_image = []
            for i in range(len(boxes)):
                attributes = {}

                odr = ObjectDetectionResultI(
                    score=float(scores[i]),
                    cls=int(labels[i]),
                    label=str(labels[i]),
                    bbox=boxes[i].tolist(),
                    image_hw=(
                        image.size
                        if isinstance(image, Image.Image)
                        else image.shape[1:]
                    ),
                    bbox_format=BBox_Format.XYXY,
                    attributes=attributes,
                )
                result_for_image.append(odr)

            formatted_results.append(result_for_image)

        return formatted_results
    
    def identify_for_image_as_tensor(
    self,
    image: Union[
        str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
    ],
    debug: bool = False,
    **kwargs
    ) -> List[Optional[ObjectDetectionResultI]]:
        """
        Run object detection on an image or a batch of images, returning results in tensor-compatible format.

        Args:
            image: An image (or batch of images) in one of the following formats:
                - PIL.Image
                - torch.Tensor of shape (B, C, H, W) or (C, H, W)
                - NumPy array of shape (B, H, W, C) or (H, W, C)
                - Path or string path to an image.
            debug: If True, visualizes the detections.
            **kwargs: Additional arguments for `inference_detector`.

        Returns:
            A list of `ObjectDetectionResultI` objects for each image in the batch.
        """
        result = inference_detector(self._model, image, **kwargs)

        formatted_results = []

        # If there are no detected instances
        if len(result.pred_instances) == 0:
            return [None] * (len(image) if isinstance(image, (list, tuple)) else 1)

        # Process each image's results
        for frame_result in result.pred_instances:
            boxes = frame_result.bboxes
            scores = frame_result.scores
            labels = frame_result.labels

            result_for_image = []

            for i in range(len(boxes)):
                attributes = {}

                # Create ObjectDetectionResultI with tensor-compatible format
                odr = ObjectDetectionResultI(
                    score=float(scores[i]),
                    cls=int(labels[i]),
                    label=str(labels[i]),
                    bbox=torch.tensor(boxes[i]).tolist(),
                    image_hw=(
                        image.size if isinstance(image, Image.Image) else image.shape[1:]
                    ),
                    bbox_format=BBox_Format.XYXY,
                    attributes=attributes,
                )
                result_for_image.append(odr)

            formatted_results.append(result_for_image)

        return formatted_results


    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[List[ObjectDetectionResultI]]]:
        def batch_iterator(iterable, n):
            iterator = iter(iterable)
            return iter(lambda: list(islice(iterator, n)), [])

        # Convert video to iterator of batches
        if isinstance(video, list):
            video_iterator = batch_iterator(video, batch_size)
        else:
            video_iterator = batch_iterator(video, batch_size)

        for batch in video_iterator:
            if not batch:  # End of iterator
                break

            images = [np.array(img) for img in batch]
            batch_results = inference_detector(self._model, images)

            boxes_across_frames = []

            if len(batch_results) == 0:
                boxes_across_frames = [[None] * len(batch)]
            else:
                for frame_result in batch_results:
                    per_frame_results = []
                    boxes = frame_result.pred_instances.bboxes
                    scores = frame_result.pred_instances.scores
                    labels = frame_result.pred_instances.labels

                    for i in range(len(boxes)):
                        attributes = { }

                        odr = ObjectDetectionResultI(
                            score=float(scores[i]),
                            cls=int(labels[i]),
                            label=str(labels[i]),
                            bbox=boxes[i].tolist(),
                            image_hw=(
                                batch[0].size
                                if isinstance(batch[0], Image.Image)
                                else batch[0].shape[1:]
                            ),
                            bbox_format=BBox_Format.XYXY,
                            attributes=attributes,
                        )
                        per_frame_results.append(odr)

                    boxes_across_frames.append(per_frame_results)

            yield boxes_across_frames
            
    
    def to(self, device: Union[str, torch.device]):
        pass
    

class MMDetectionInstanceSegmentation(InstanceSegmentationModelI):
    def __init__(
        self, config: str, checkpoint: str, device: Optional[str] = None
    ) -> None:
        if device is None:
            device = get_default_device()
        self._model = init_detector(config, checkpoint, device=device)

    def identify_for_image(
        self, image: Union[Image.Image, torch.Tensor], **kwargs
    ) -> List[List[InstanceSegmentationResultI]]:
        """
        Run instance segmentation on a single image.

        Args:
            image: A PIL image or a tensor of shape (C, H, W).

        Returns:
            A list of list of InstanceSegmentationResultI, where each inner list represents
            segmentation in a single image.
        """
        result = inference_detector(self._model, image)

        formatted_results = []
        if len(result.pred_instances) > 0:
            boxes = result.pred_instances.bboxes
            scores = result.pred_instances.scores
            labels = result.pred_instances.labels
            masks = result.pred_instances.masks

            result_for_image = []
            for i in range(len(boxes)):
                attributes = {}  # Placeholder for potential attributes

                # Create InstanceSegmentationResultI with mask
                odr = InstanceSegmentationResultI(
                    score=float(scores[i]),
                    cls=int(labels[i]),
                    label=str(labels[i]),
                    instance_id=i,  # Assign unique instance ID
                    mask=masks[i],
                    image_hw=(
                        image.size
                        if isinstance(image, Image.Image)
                        else image.shape[1:]
                    ),
                    mask_format=Mask_Format.BITMASK,
                )
                result_for_image.append(odr)

            formatted_results.append(result_for_image)

        return formatted_results
    
    def identify_for_image_as_tensor(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs
    ) -> List[Optional[InstanceSegmentationResultI]]:
        """ Run instance segmentation on an image or a batch of images.

        Args:
            image: either a PIL image or a tensor of shape (B, C, H, W) where B is the batch size,
                C is the channel size, H is the height, and W is the width.
            debug: If True, displays the image with segmentations.

        Returns:
            A list of InstanceSegmentationResultI for each image in the batch.
        """
        results = inference_detector(self._model, image)

        if len(results.pred_instances) == 0:
            # Return a list of `None` if no predictions
            return [None] * (len(image) if isinstance(image, (list, tuple)) else 1)
        all_boxes = []
        all_scores = []
        all_classes = []
        all_labels = []
        all_masks = []
        image_hw = (
            image.size if isinstance(image, Image.Image) else image.shape[1:]  # H, W
        )
        
        for frame_result in results.pred_instances:
            boxes = frame_result.bboxes
            scores = frame_result.scores
            labels = frame_result.labels
            masks = frame_result.masks

            # Append results for this image
            all_boxes.append(torch.tensor(boxes))
            all_scores.append(torch.tensor(scores))
            all_classes.append(torch.tensor(labels))
            all_labels.append([str(label) for label in labels])
            all_masks.append(torch.stack([mask.bool() for mask in masks]))  # Stack all masks

        formatted_results = []
        # Create InstanceSegmentationResultI for each image
        for bboxes, scores, classes, labels, masks in zip(all_boxes, all_scores, all_classes, all_labels, all_masks):
            odr = InstanceSegmentationResultI(
                score=scores,
                cls=classes,
                label=labels,  # List of class names
                instance_id=torch.arange(len(bboxes)),  # Generate unique instance IDs
                mask=masks,   # Tensor of masks
                image_hw=image_hw,
                mask_format=Mask_Format.BITMASK,
            )
        formatted_results.append(odr)

        return formatted_results

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[List[InstanceSegmentationResultI]]]:
        def batch_iterator(iterable, n):
            iterator = iter(iterable)
            return iter(lambda: list(islice(iterator, n)), [])

        # Convert video to iterator of batches
        if isinstance(video, list):
            video_iterator = batch_iterator(video, batch_size)
        else:
            video_iterator = batch_iterator(video, batch_size)

        for batch in video_iterator:
            if not batch:  # End of iterator
                break

            images = [np.array(img) for img in batch]
            batch_results = inference_detector(self._model, images)

            boxes_across_frames = []

            if len(batch_results) == 0:
                boxes_across_frames = [[None] * len(batch)]
            else:
                for frame_result in batch_results:
                    per_frame_results = []
                    bboxes = frame_result.pred_instances.bboxes
                    scores = frame_result.pred_instances.scores
                    labels = frame_result.pred_instances.labels
                    masks = frame_result.pred_instances.masks

                    for i in range(len(bboxes)):
                        attributes = {}  # Placeholder for potential attributes

                        # Create InstanceSegmentationResultI with mask
                        odr = InstanceSegmentationResultI(
                            score=float(scores[i]),
                            cls=int(labels[i]),
                            label=str(labels[i]),
                            instance_id=i,  # Assign unique instance ID
                            mask=masks[i],
                            image_hw=(
                                batch[0].size
                                if isinstance(batch[0], Image.Image)
                                else batch[0].shape[1:]
                            ),
                            mask_format=Mask_Format.BITMASK,
                        )
                        per_frame_results.append(odr)

                    boxes_across_frames.append(per_frame_results)

            yield boxes_across_frames
    
    def to(self, device: Union[str, torch.device]):
        pass
