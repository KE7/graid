import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Type, Union

import mmdet
import numpy as np
import pycocotools.mask as mask_util
import torch
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
from graid.utilities.coco import coco_labels, coco_panoptic_labels
from graid.utilities.common import convert_batch_to_numpy, convert_image_to_numpy
from mmdet.utils import collect_env as collect_base_env
from mmengine.logging import print_log
from mmengine.registry import Registry
from mmengine.utils import get_git_hash
from PIL import Image


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
from mmdet.apis import inference_detector, init_detector

# fmt: on


class MMDetectionBase:
    """Base class for MMDetection models with shared functionality."""

    def __init__(
        self, config_file: str, checkpoint_file: str, device: Optional[str] = None
    ):
        # Use provided device or auto-detect
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = init_detector(config_file, checkpoint_file, device=device)
        self.model_name = config_file
        self._device = device
        self.threshold = 0.0

    def collect_env(self):
        env_info = collect_base_env()
        env_info["MMDetection"] = f"{mmdet.__version__}+{get_git_hash()[:7]}"
        return env_info

    def to(self, device: Union[str, torch.device]):
        """Move model to specified device."""
        self._model.to(device)
        self._device = str(device)

    def set_threshold(self, threshold: float):
        """Set confidence threshold for detections."""
        self.threshold = threshold

    def __str__(self):
        return self.model_name


class MMdetection_obj(MMDetectionBase, ObjectDetectionModelI):
    def __init__(self, config_file: str, checkpoint_file: str, **kwargs) -> None:
        device = kwargs.get("device", None)
        super().__init__(config_file, checkpoint_file, device)

    def _extract_detections(self, pred) -> List[ObjectDetectionResultI]:
        """Extract object detection results from prediction."""
        bboxes = pred.pred_instances.bboxes.tolist()
        labels = pred.pred_instances.labels
        scores = pred.pred_instances.scores
        image_hw = pred.pad_shape

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
        return objects

    def identify_for_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        debug: bool = False,
        **kwargs,
    ) -> List[ObjectDetectionResultI]:
        """Run object detection on a single image."""
        image = convert_image_to_numpy(image)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        predictions = inference_detector(self._model, image, **kwargs)
        # Handle single image case - predictions is a single DetDataSample
        if not isinstance(predictions, list):
            predictions = [predictions]

        detections = self._extract_detections(predictions[0])

        # Apply manual threshold filtering if threshold is set
        if self.threshold is not None:
            detections = [det for det in detections if det.score >= self.threshold]

        return detections

    def identify_for_image_batch(
        self,
        image: Union[Union[np.ndarray, torch.Tensor], str],
        debug: bool = False,
        **kwargs,
    ) -> List[List[ObjectDetectionResultI]]:
        """
        Args:
            image: either a PIL image or a tensor of shape(B, C, H, W)
                where B is the batch size, C is the channel size, H is the
                height, and W is the width.

        Returns:
            A list of list of ObjectDetectionResultI, where the outer list
            represents the batch of images, and the inner list represents the
            detections in a particular image.
        """
        image_list = convert_batch_to_numpy(image)

        predictions = inference_detector(self._model, image_list)

        # Extract detections and apply manual threshold filtering
        all_detections = []
        for pred in predictions:
            detections = self._extract_detections(pred)
            # Apply manual threshold filtering if threshold is set
            if self.threshold is not None:
                detections = [det for det in detections if det.score >= self.threshold]
            all_detections.append(detections)

        return all_detections

    def identify_for_video(
        self,
        video: Union[
            Iterator[Union[np.ndarray, torch.Tensor]],
            List[Union[np.ndarray, torch.Tensor]],
        ],
        batch_size: int = 1,
    ) -> Iterator[List[Optional[ObjectDetectionResultI]]]:
        raise NotImplementedError


class MMdetection_seg(MMDetectionBase, InstanceSegmentationModelI):
    def __init__(self, config_file: str, checkpoint_file, **kwargs) -> None:
        device = kwargs.get("device", None)
        super().__init__(config_file, checkpoint_file, device)

        # set class_agnostic to True to avoid overlaps: https://github.com/open-mmlab/mmdetection/issues/6254
        if hasattr(self._model.test_cfg, "rcnn") and hasattr(
            self._model.test_cfg.rcnn, "nms"
        ):
            self._model.test_cfg.rcnn.nms.class_agnostic = True

        self._supports_panoptic = self._check_panoptic_support()

    def _check_panoptic_support(self) -> bool:
        """Check if the model supports panoptic segmentation."""
        if hasattr(self._model, "test_cfg") and hasattr(
            self._model.test_cfg, "panoptic_on"
        ):
            return self._model.test_cfg.panoptic_on
        return False

    def _extract_panoptic_results(
        self, pred, image_hw
    ) -> List[InstanceSegmentationResultI]:
        """Extract panoptic segmentation results."""
        labels = []
        scores = []
        masks = []

        if hasattr(pred, "pred_panoptic_seg") and pred.pred_panoptic_seg is not None:
            panoptic_seg = pred.pred_panoptic_seg
            if hasattr(panoptic_seg, "sem_seg"):
                sem_seg = panoptic_seg.sem_seg[0]
                unique_ids = torch.unique(sem_seg)

                for segment_id in unique_ids:
                    if segment_id == 0:
                        continue

                    segment_id_int = segment_id.item()
                    class_id = (
                        segment_id_int
                        if segment_id_int < 1000
                        else segment_id_int // 1000
                    )

                    if class_id in coco_panoptic_labels:
                        mask = sem_seg == segment_id
                        labels.append(class_id)
                        scores.append(0.0)
                        masks.append(mask)

        if labels:
            out = {"labels": labels, "scores": scores, "masks": masks}
            return self.out_to_seg(out, image_hw)
        return []

    def out_to_seg(self, out: dict, image_hw: Tuple[int, int]):
        seg = []
        i = 0
        for label, score, mask in zip(out["labels"], out["scores"], out["masks"]):
            if isinstance(mask, dict):
                decoded = mask_util.decode(mask)
                mask_tensor = torch.from_numpy(decoded.astype(bool)).unsqueeze(0)
            elif isinstance(mask, (list, tuple)):
                decoded = mask_util.decode(mask_util.frPyObjects(mask, *image_hw))
                mask_tensor = torch.from_numpy(decoded.astype(bool)).unsqueeze(0)
            elif isinstance(mask, torch.Tensor):
                # Handle tensors that are already in the correct format
                if mask.dim() == 2:
                    mask_tensor = mask.unsqueeze(0)
                else:
                    mask_tensor = mask
                # Ensure it's boolean
                if mask_tensor.dtype != torch.bool:
                    mask_tensor = mask_tensor.bool()
            else:
                raise TypeError(f"Unknown mask type: {type(mask)}")

            seg += [
                InstanceSegmentationResultI(
                    score=float(score),
                    cls=float(label),
                    label=coco_labels[int(label)],
                    instance_id=i,
                    image_hw=image_hw,
                    mask=mask_tensor,
                    mask_format=Mask_Format.BITMASK,
                )
            ]
            i += 1

        return seg

    def _extract_instance_results(
        self, pred, image_hw
    ) -> List[InstanceSegmentationResultI]:
        """Extract instance segmentation results."""
        if hasattr(pred, "pred_instances") and hasattr(pred.pred_instances, "masks"):
            masks = pred.pred_instances.masks
            labels = pred.pred_instances.labels
            scores = pred.pred_instances.scores

            out = {
                "labels": [label.item() for label in labels],
                "scores": [score.item() for score in scores],
                "masks": [mask for mask in masks],
            }
            return self.out_to_seg(out, image_hw)

        return []

    def _process_single_image(
        self, image: np.ndarray
    ) -> List[InstanceSegmentationResultI]:
        """Process a single image for segmentation."""
        predictions = inference_detector(self._model, image)
        pred = predictions[0] if isinstance(predictions, list) else predictions
        image_hw = pred.pad_shape

        if self._supports_panoptic:
            instances = self._extract_panoptic_results(pred, image_hw)
        else:
            instances = self._extract_instance_results(pred, image_hw)

        # Apply manual threshold filtering if threshold is set
        if self.threshold is not None:
            instances = [inst for inst in instances if inst.score >= self.threshold]

        return instances

    def identify_for_image(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs,
    ) -> List[InstanceSegmentationResultI]:
        """Run segmentation on a single image."""
        image = convert_image_to_numpy(image)
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        return self._process_single_image(image)

    def identify_for_image_batch(
        self,
        image: Union[Union[np.ndarray, torch.Tensor], str],
        debug: bool = False,
        **kwargs,
    ) -> List[List[InstanceSegmentationResultI]]:
        """Run segmentation on a batch of images."""
        if isinstance(image, torch.Tensor):
            image_list = convert_batch_to_numpy(image)

            predictions = inference_detector(self._model, image_list)

            all_instances = []
            for pred in predictions:
                image_hw = pred.pad_shape

                if self._supports_panoptic:
                    instances = self._extract_panoptic_results(pred, image_hw)
                else:
                    instances = self._extract_instance_results(pred, image_hw)

                # Apply manual threshold filtering if threshold is set
                if self.threshold is not None:
                    instances = [
                        inst for inst in instances if inst.score >= self.threshold
                    ]

                all_instances.append(instances)

            return all_instances
        else:
            return [self.identify_for_image(image, debug=debug, **kwargs)]

    def identify_for_video(
        self,
        video: Union[
            Iterator[Union[np.ndarray, torch.Tensor]],
            List[Union[np.ndarray, torch.Tensor]],
        ],
        batch_size: int = 1,
    ) -> Iterator[List[Optional[ObjectDetectionResultI]]]:
        raise NotImplementedError
