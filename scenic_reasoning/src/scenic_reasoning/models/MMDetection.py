#  hack_registry.py
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Type, Union

import numpy as np
import torch
from mmengine.logging import print_log
from mmengine.registry import Registry
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
    ObjectDetectionUtils,
)
from scenic_reasoning.utilities.coco import coco_labels


# https://github.com/open-mmlab/mmdetection/issues/12008
def _register_module(
    self,
    module: Type,
    module_name: Optional[Union[str, List[str]]] = None,
    force: bool = False,
) -> None:
    """Register a module.

    Args:
        module (type): Module to be registered. Typically a class or a
            function, but generally all ``Callable`` are acceptable.
        module_name (str or list of str, optional): The module name to be
            registered. If not specified, the class name will be used.
            Defaults to None.
        force (bool): Whether to override an existing class with the same
            name. Defaults to False.
    """
    if not callable(module):
        raise TypeError(f"module must be Callable, but got {type(module)}")

    if module_name is None:
        module_name = module.__name__
    if isinstance(module_name, str):
        module_name = [module_name]
    for name in module_name:
        if not force and name in self._module_dict:
            existed_module = self.module_dict[name]
            # raise KeyError(f'{name} is already registered in {self.name} '
            #                f'at {existed_module.__module__}')
            print_log(
                f"{name} is already registered in {self.name} "
                f"at {existed_module.__module__}. Registration ignored.",
                logger="current",
                level=logging.INFO,
            )
        self._module_dict[name] = module


Registry._register_module = _register_module

# fmt: off
import mmdet
from mmdet.apis import inference_detector, init_detector
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

# fmt: on


class MMdetection_obj(ObjectDetectionModelI):
    def __init__(self, config_file: str, checkpoint_file, **kwargs) -> None:
        if kwargs.get("device", None):
            device = "cpu"  # Using mps will error, see: https://github.com/open-mmlab/mmdetection/issues/11794
            if torch.cuda.is_available():
                device = "cuda"
        else:
            device = kwargs["device"]

        self._model = init_detector(config_file, checkpoint_file, device=device)
        self.model_name = config_file

        # set class_agnostic to True to avoid overlaps: https://github.com/open-mmlab/mmdetection/issues/6254
        # self._model.test_cfg.rcnn.nms.class_agnostic = True

    def collect_env(self):
        """Collect the information of the running environments."""
        env_info = collect_base_env()
        env_info["MMDetection"] = f"{mmdet.__version__}+{get_git_hash()[:7]}"
        return env_info

    def identify_for_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        debug: bool = False,
        **kwargs,
    ) -> List[List[ObjectDetectionResultI]]:
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
        # if len(image.shape) == 3:
        #     # single image, add batch dimension
        #     image = image.unsqueeze(0) if isinstance(image, torch.Tensor) else np.expand_dims(image, 0)
        # image_list = [
        #     image[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        #     for i in range(len(image))
        # ]
        # image_hw = image_list[0].shape[:-1]
        image = (
            image.permute(1, 2, 0).cpu().numpy()
            if isinstance(image, torch.Tensor)
            else image
        )
        image = image.astype(np.uint8)

        predictions = inference_detector(self._model, image, **kwargs)

        all_objects = []

        for pred in predictions:
            bboxes = pred.pred_instances.bboxes.tolist()
            labels = pred.pred_instances.labels
            scores = pred.pred_instances.scores
            image_hw = pred.pad_shape  # TODO: should I use pad_shape or img_shape?
            objects = []

            for i in range(len(labels)):
                cls_id = labels[i].item()
                score = scores[i].item()
                bbox = bboxes[i]

                odr = ObjectDetectionResultI(
                    score=score,
                    cls=cls_id,
                    label=coco_labels[cls_id],
                    bbox=bbox,
                    image_hw=image_hw,
                    bbox_format=BBox_Format.XYXY,
                )
                objects.append(odr)
            all_objects.append(objects)

        if debug:
            if image.dtype == np.float32:
                curr_img = curr_img.astype(np.uint8)
            ObjectDetectionUtils.show_image_with_detections(
                Image.fromarray(curr_img), all_objects[i]
            )

        return all_objects

    def identify_for_image_batch(
        self,
        image: Union[np.ndarray, torch.Tensor],
        debug: bool = False,
        **kwargs,
    ) -> List[List[ObjectDetectionResultI]]:
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
        image_list = [
            image[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            for i in range(len(image))
        ]
        image_hw = image_list[0].shape[:-1]
        predictions = inference_detector(self._model, image_list)

        all_objects = []

        for pred in predictions:
            bboxes = pred.pred_instances.bboxes.tolist()
            labels = pred.pred_instances.labels
            scores = pred.pred_instances.scores
            image_hw = pred.pad_shape  # TODO: should I use pad_shape or img_shape?
            objects = []

            for i in range(len(labels)):
                cls_id = labels[i].item()
                score = scores[i].item()
                bbox = bboxes[i]

                odr = ObjectDetectionResultI(
                    score=score,
                    cls=cls_id,
                    label=coco_labels[cls_id],
                    bbox=bbox,
                    image_hw=image_hw,
                    bbox_format=BBox_Format.XYXY,
                )
                objects.append(odr)
            all_objects.append(objects)

        if debug:
            for i in range(len(image_list)):
                curr_img = image_list[i]
                if image_list[i].dtype == np.float32:
                    curr_img = curr_img.astype(np.uint8)
                ObjectDetectionUtils.show_image_with_detections(
                    Image.fromarray(curr_img), all_objects[i]
                )

        return all_objects

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> List[List[ObjectDetectionResultI]]:
        pass

    def to(self, device: Union[str, torch.device]):
        self._model.to(device)

    def set_threshold(self, threshold: float):
        pass

    def __str__(self):
        return self.model_name


class MMdetection_seg(InstanceSegmentationModelI):
    def __init__(self, config_file: str, checkpoint_file, **kwargs) -> None:
        device = "cpu"  # Using mps will error, see: https://github.com/open-mmlab/mmdetection/issues/11794
        if torch.cuda.is_available():
            device = "cuda"
        self._model = init_detector(config_file, checkpoint_file, device=device)

        # set class_agnostic to True to avoid overlaps: https://github.com/open-mmlab/mmdetection/issues/6254
        self._model.test_cfg.rcnn.nms.class_agnostic = True

    def identify_for_image(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs,
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
        image_list = [
            image[i].permute(1, 2, 0).cpu().numpy() for i in range(len(image))
        ]
        predictions = inference_detector(self._model, image_list)

        # if debug:
        #     # TODO: design a new visualizer
        #     visualizer = VISUALIZERS.build(self._model.cfg.visualizer)
        #     visualizer.dataset_meta = self._model.dataset_meta
        #     for i in range(len(image_list)):
        #         visualizer.add_datasample(
        #             'result',
        #             image_list[i],
        #             data_sample=predictions[i],
        #             draw_gt = None,
        #             wait_time=0
        #         )
        #         visualizer.show()

        all_instances = []
        for pred in predictions:
            masks = pred.pred_instances.masks
            labels = pred.pred_instances.labels
            scores = pred.pred_instances.scores
            image_hw = pred.pad_shape  # TODO: should I use pad_shape or img_shape?
            instances = []
            for i in range(len(labels)):
                cls_id = labels[i].item()
                mask = masks[i]
                score = scores[i].item()
                instance = InstanceSegmentationResultI(
                    score=score,
                    cls=cls_id,
                    label=coco_labels[cls_id],
                    instance_id=i,
                    mask=mask.unsqueeze(0),
                    image_hw=image_hw,
                    mask_format=Mask_Format.BITMASK,
                )

                instances.append(instance)
            all_instances.append(instances)

        return all_instances

    def identify_for_image_batch(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs,
    ) -> List[Optional[InstanceSegmentationResultI]]:
        """Run instance segmentation on an image or a batch of images.
        Args:
            image: either a PIL image or a tensor of shape (B, C, H, W) where B is the batch size,
                C is the channel size, H is the height, and W is the width.
            debug: If True, displays the image with segmentations.
        Returns:
            A list of InstanceSegmentationResultI for each image in the batch.
        """
        pass

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[InstanceSegmentationResultI]]:
        pass

    def to(self, device: Union[str, torch.device]):
        # self._model.to(device)
        pass
