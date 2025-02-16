from itertools import islice
from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from scenic_reasoning.interfaces.InstanceSegmentationI import (
    InstanceSegmentationModelI,
    InstanceSegmentationResultI,
    Mask_Format,
)
from scenic_reasoning.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionModelI,
    ObjectDetectionResultI,
    ObjectDetectionUtils
)
from scenic_reasoning.utilities.common import get_default_device
from ultralytics import YOLO

# TODO: Need class for InstanceSegmentation here


class Yolo(ObjectDetectionModelI):
    def __init__(self, model: Union[str, Path], **kwargs) -> None:
        self._model = YOLO(model, **kwargs)

    def identify_for_image(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs
    ) -> List[List[Optional[ObjectDetectionResultI]]]:
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
        predictions = self._model.predict(image, device=get_default_device(), **kwargs)

        if len(predictions) == 0:
            return [[None]]

        formatted_results = []
        for y_hat in predictions:
            result_for_image = []
            boxes = y_hat.boxes
            names = y_hat.names

            if boxes is None or len(boxes) == 0:
                formatted_results.append(None)
                continue

            for box in boxes:
                odr = ObjectDetectionResultI(
                    score=box.conf.item(),
                    cls=int(box.cls.item()),
                    label=names[int(box.cls.item())],
                    bbox=box.cpu(),
                    image_hw=box.orig_shape,
                    bbox_format=BBox_Format.UltralyticsBox,
                )

                result_for_image.append(odr)

            formatted_results.append(result_for_image)
        
        if debug:
            image_list = [image[i].permute(1, 2, 0).cpu().numpy() for i in range(len(image))]
            for i in range(len(image_list)):
                curr_img = image_list[i]
                if curr_img.dtype == np.float32:
                    curr_img = curr_img.astype(np.uint8)
                ObjectDetectionUtils.show_image_with_detections(Image.fromarray(curr_img), formatted_results[i])

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
        predictions = self._model.predict(image, **kwargs)

        if len(predictions) == 0:
            return [None]

        result_per_image = []
        for y_hat in predictions:
            boxes = y_hat.boxes
            names = y_hat.names

            if debug:
                y_hat.show()

            bboxes = []
            scores = []
            classes = []

            if boxes is None or len(boxes) == 0:
                result_per_image.append(None)
                continue

            for box in boxes:
                bboxes.append(box.cpu())
                scores.append(box.conf.item())
                classes.append(int(box.cls.item()))

            bboxes = torch.stack(bboxes)
            scores = torch.tensor(scores)
            classes = torch.tensor(classes)

            odr = ObjectDetectionResultI(
                score=scores,
                cls=classes,
                label=names[classes],  # shape: (# of boxes,)
                bbox=bboxes,
                image_hw=boxes[0].orig_shape,
                bbox_format=BBox_Format.XYXY,
            )

            result_per_image.append(odr)

        return result_per_image

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[Optional[ObjectDetectionResultI]]]:
        def _batch_iterator(iterable, n):
            iterator = iter(iterable)
            return iter(lambda: list(islice(iterator, n)), [])

        # If video is a list, convert it to an iterator of batches
        if isinstance(video, list):
            video_iterator = _batch_iterator(video, batch_size)
        else:
            # If video is already an iterator, create batches from it
            video_iterator = _batch_iterator(video, batch_size)

        for batch in video_iterator:
            if not batch:  # End of iterator
                break

            images = torch.stack([torch.tensor(np.array(img)) for img in batch])
            batch_results = self._model(images)

            boxes_across_frames = []

            if len(batch_results) == 0:
                boxes_across_frames = [None for _ in batch]
            else:
                for frame_result in batch_results:
                    per_frame_results = []

                    boxes = frame_result.boxes
                    names = frame_result.names

                    for box in boxes:
                        odr = ObjectDetectionResultI(
                            score=box.conf.item(),
                            cls=int(box.cls.item()),
                            label=names[int(box.cls.item())],
                            bbox=box,
                            image_hw=box.orig_shape,
                            bbox_format=BBox_Format.UltralyticsBox,
                        )

                        per_frame_results.append(odr)

                    boxes_across_frames.append(per_frame_results)

            yield boxes_across_frames

    def to(self, device: Union[str, torch.device]):
        pass


class Yolo_seg(InstanceSegmentationModelI):
    def __init__(self, model: Union[str, Path], **kwargs) -> None:
        super().__init__()
        self._model = YOLO(model, **kwargs)
        self._instance_count = {}

    def identify_for_image(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs
    ) -> List[List[Optional[InstanceSegmentationResultI]]]:
        """
        Run instance segmentation on an image or a batch of images.

        Args:
            image: either a PIL image or a tensor of shape (B, C, H, W)
                where B is the batch size, C is the channel size, H is the
                height, and W is the width.

        Returns:
            A list of list of InstanceSegmentationResultI, where the outer list
            represents the batch of images, and the inner list represents the
            detections in a particular image.
        """
        results = self._model.predict(source=image)

        # results = self._model.track(source=image, persist=True)

        all_instances = []

        for result in results:

            if debug:
                result.show()

            instances = []
            if result.masks is None:
                all_instances.append([])
                continue

            masks = result.masks.data
            cls_ids = result.boxes.cls
            scores = result.boxes.conf
            name_map = result.names
            num_instances = masks.shape[0]

            for i in range(num_instances):
                mask = masks[i]
                cls_id = cls_ids[i].item()
                cls_label = name_map[cls_id]
                score = scores[i]

                instance = InstanceSegmentationResultI(
                    score=score,
                    cls=cls_id,
                    label=cls_label,
                    instance_id=i,
                    mask=mask.unsqueeze(0),
                    image_hw=result.orig_shape,
                    mask_format=Mask_Format.BITMASK,
                )

                instances.append(instance)
            all_instances.append(instances)

        return all_instances

    def identify_for_image_as_tensor(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs
    ) -> List[Optional[InstanceSegmentationResultI]]:
        """Run instance segmentation on an image or a batch of images.
        Args:
            image: either a PIL image or a tensor of shape (B, C, H, W) where B is the batch size,
                C is the channel size, H is the height, and W is the width.
            debug: If True, displays the image with segmentations.
        Returns:
            A list of InstanceSegmentationResultI for each image in the batch.
        """
        results = self._model.predict(image, **kwargs)

        if results.masks is None:
            return [None] * (len(image) if isinstance(image, (list, tuple)) else 1)

        instances = []

        masks = results.masks.data
        boxes = results.boxes
        names = results.names

        for img_idx in range(len(masks)):  # Process each image in the batch
            image_masks = masks[img_idx]
            image_boxes = boxes[img_idx]

            if debug:
                results.show(img_idx)

            aggregated_masks = []
            scores = []
            classes = []

            for mask, box in zip(image_masks, image_boxes):
                class_id = int(box.cls.item())
                if class_id not in self._instance_count:
                    self._instance_count[class_id] = 0
                self._instance_count[class_id] += 1

                mask_tensor = mask.bool().cpu()
                aggregated_masks.append(mask_tensor)
                scores.append(box.conf.item())
                classes.append(class_id)

            masks_tensor = torch.stack(aggregated_masks)
            scores_tensor = torch.tensor(scores)
            classes_tensor = torch.tensor(classes)

            instance = InstanceSegmentationResultI(
                score=scores_tensor,
                cls=classes_tensor,
                label=[names[class_id] for class_id in classes_tensor],
                instance_id=None,  # Not aggregating IDs across batch
                mask=masks_tensor,
                image_hw=results.orig_shape,
                mask_format=Mask_Format.BITMASK,
            )
            instances.append(instance)

        return instances

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[InstanceSegmentationResultI]]:
        def _batch_iterator(iterable, n):
            iterator = iter(iterable)
            return iter(lambda: list(islice(iterator, n)), [])

        video_iterator = (
            _batch_iterator(video, batch_size)
            if isinstance(video, list)
            else _batch_iterator(video, batch_size)
        )

        for batch in video_iterator:
            if not batch:
                break

            images = torch.stack([torch.tensor(np.array(img)) for img in batch])
            batch_results = self._model(images)

            results_per_frame = []
            for results in batch_results:
                if results.masks is None:
                    results_per_frame.append([])
                    continue

                instances = []
                masks = results.masks.data
                boxes = results.boxes

                for mask, box in zip(masks, boxes):
                    class_id = int(box.cls.item())

                    if class_id not in self._instance_count:
                        self._instance_count[class_id] = 0
                    self._instance_count[class_id] += 1

                    mask_tensor = mask.bool().cpu()
                    if len(mask_tensor.shape) == 2:
                        mask_tensor = mask_tensor.unsqueeze(0)

                    instance = InstanceSegmentationResultI(
                        score=box.conf.item(),
                        cls=class_id,
                        label=results.names[class_id],
                        instance_id=self._instance_count[class_id],
                        mask=mask_tensor,
                        image_hw=results.orig_shape,
                        mask_format=Mask_Format.BITMASK,
                    )
                    instances.append(instance)

                results_per_frame.append(instances)

            yield results_per_frame

    def to(self, device: Union[str, torch.device]):
        # self._model.to(device)
        pass
