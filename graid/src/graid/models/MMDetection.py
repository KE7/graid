#  MMDetection.py

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Type, Union, Tuple

import cv2
import torch
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util

from graid.interfaces.InstanceSegmentationI import (
    InstanceSegmentationModelI,
    InstanceSegmentationResultI,
    Mask_Format,
)
from graid.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionModelI,
    ObjectDetectionResultI,
    ObjectDetectionUtils,
)
from graid.utilities.coco import coco_labels

import mmdet
from mmdet.apis import DetInferencer
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

from mmengine.logging import print_log
from mmengine.registry import Registry

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


class MMdetection_obj(ObjectDetectionModelI):
    def __init__(
        self,
        config_file: str,
        checkpoint_file: str,
        **kwargs
    ) -> None:

        # Not MPS compatible!

        self.model = config_file
        self.weights = checkpoint_file
        self.inferencer = DetInferencer(
            model=self.model,
            weights=self.weights,
            device=kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.batch_size = kwargs.get('batch_size', 1)

    def collect_env(self):
        env_info = collect_base_env()
        env_info['MMDetection'] = f'{mmdet.__version__}+{get_git_hash()[:7]}'
        return env_info

    def out_to_obj(self, out: dict, image_hw: Tuple[int, int]):
        obj = []
        for label, score, bbox in zip(out['labels'], out['scores'], out['bboxes']):
            obj += [
                ObjectDetectionResultI(
                    score=score,
                    cls=label,
                    label=coco_labels[label],
                    bbox=bbox,
                    image_hw=image_hw,
                    bbox_format=BBox_Format.XYXY
                )
            ]

        return obj

    def identify_for_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        debug: bool = False,
        **kwargs,
    ) -> List[List[ObjectDetectionResultI]]:

        if isinstance(image, str):
            image_hw = cv2.imread(image).shape[:2]
        else:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()

            image = image.astype(np.uint8)
            image_hw = image.shape[:2]

        pred = self.inferencer(
            inputs=image,
            out_dir=kwargs.get('out_dir', ''),
            batch_size=1
        )['predictions'][0]
        out = self.out_to_obj(pred, image_hw)

        if debug:
            ObjectDetectionUtils.show_image_with_detections(
                Image.fromarray(image), out
            )

        return [out]

    def identify_for_image_batch(
        self,
        image: Union[Union[np.ndarray, torch.Tensor], str],
        debug: bool = False,
        **kwargs,
    ) -> List[List[ObjectDetectionResultI]]:
        
        image_hws = []
        input_data = image

        if isinstance(image, str):
            image_dir = Path(image)
            
            image_paths = sorted([
                p for p in image_dir.iterdir() 
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            ])
            if not image_paths:
                return []
            
            for path in image_paths:
                image_hws.append(cv2.imread(str(path)).shape[:2])
            
            input_data = [str(p) for p in image_paths]

        elif isinstance(image, list):
            if not image:
                return []
            
            processed_list = []
            for img_item in image:
                if isinstance(img_item, str):
                    image_hw = cv2.imread(img_item).shape[:2]
                    processed_list.append(img_item)
                elif isinstance(img_item, torch.Tensor):
                    img_np = img_item.detach().cpu().numpy().astype(np.uint8)
                    image_hw = img_np.shape[:2]
                    processed_list.append(img_np)
                elif isinstance(img_item, np.ndarray):
                    img_np = img_item.astype(np.uint8)
                    image_hw = img_np.shape[:2]
                    processed_list.append(img_np)
                else:
                    raise TypeError(f"Unsupported image type in list: {type(img_item)}")
                image_hws.append(image_hw)
            input_data = processed_list
        else:
            raise TypeError("Input must be a list of images or a path to a directory.")

        predictions = self.inferencer(
            inputs=input_data,
            out_dir=kwargs.get('out_dir', ''),
            batch_size=kwargs.get('batch_size', self.batch_size)
        )['predictions']
        out = [self.out_to_obj(pred, hw) for pred, hw in zip(predictions, image_hws)]

        if debug:
            for i in range(len(input_data)):
                ObjectDetectionUtils.show_image_with_detections(
                    Image.fromarray(input_data[i]), out[i]
                )

        return out
    
    def identify_for_video(
        self,
        video: Union[
            Iterator[Union[np.ndarray, torch.Tensor]],
            List[Union[np.ndarray, torch.Tensor]],
        ],
        batch_size: int = 1,
    ) -> Iterator[List[Optional[ObjectDetectionResultI]]]:
        raise NotImplementedError

    def to(self, device: Union[str, torch.device]):
        self.inferencer = DetInferencer(
            model=self.model,
            weights=self.weights,
            device=device
        )

    def set_threshold(self, threshold: float):
        raise NotImplementedError

    def __str__(self):
        return self.model


class MMdetection_seg(InstanceSegmentationModelI):
    def __init__(
        self,
        config_file: str,
        checkpoint_file: str,
        **kwargs
    ) -> None:

        # Not MPS compatible!

        self.model = config_file
        self.weights = checkpoint_file

        # TODO: Modify configuration before initalization?
        # self.model.test_cfg.rcnn.nms.class_agnostic = True

        self.inferencer = DetInferencer(
            model=self.model,
            weights=self.weights,
            device=kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.batch_size = kwargs.get('batch_size', 1)

    def out_to_seg(self, out: dict, image_hw: Tuple[int, int]):
        seg = []
        i = 0
        for label, score, mask in zip(out['labels'], out['scores'], out['masks']):
            if isinstance(mask, dict):
                decoded = mask_util.decode(mask)
            elif isinstance(mask, (list, tuple)):
                decoded = mask_util.decode(
                    mask_util.frPyObjects(mask, *image_hw))
            elif isinstance(mask, torch.Tensor):
                decoded = mask.cpu().numpy()
            else:
                raise TypeError(f'Unknown mask type: {type(mask)}')

            mask_tensor = torch.from_numpy(decoded.astype(bool)).unsqueeze(0)

            seg += [
                InstanceSegmentationResultI(
                    score=float(score),
                    cls=float(label),
                    label=coco_labels[int(label)],
                    instance_id=i,
                    image_hw=image_hw,
                    mask=mask_tensor,
                    mask_format=Mask_Format.BITMASK
                )
            ]
            i += 1

        return seg

    def identify_for_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        debug: bool = False,
        **kwargs,
    ) -> List[List[InstanceSegmentationResultI]]:

        if isinstance(image, str):
            image_hw = cv2.imread(image).shape[:2]
        else:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()

            image = image.astype(np.uint8)
            image_hw = image.shape[:2]

        pred = self.inferencer(
            inputs=image,
            out_dir=kwargs.get('out_dir', ''),
            batch_size=1
        )['predictions'][0]
        out = self.out_to_seg(pred, image_hw)

        # TODO: Visualization

        return [out]

    def identify_for_image_batch(
        self,
        image: Union[Union[np.ndarray, torch.Tensor], str],
        debug: bool = False,
        **kwargs,
    ) -> List[List[ObjectDetectionResultI]]:
        
        image_hws = []
        input_data = image

        if isinstance(image, str):
            image_dir = Path(image)
            
            image_paths = sorted([
                p for p in image_dir.iterdir() 
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            ])
            if not image_paths:
                return []
            
            for path in image_paths:
                image_hws.append(cv2.imread(str(path)).shape[:2])
            
            input_data = [str(p) for p in image_paths]

        elif isinstance(image, list):
            if not image:
                return []
            
            processed_list = []
            for img_item in image:
                if isinstance(img_item, str):
                    image_hw = cv2.imread(img_item).shape[:2]
                    processed_list.append(img_item)
                elif isinstance(img_item, torch.Tensor):
                    img_np = img_item.detach().cpu().numpy().astype(np.uint8)
                    image_hw = img_np.shape[:2]
                    processed_list.append(img_np)
                elif isinstance(img_item, np.ndarray):
                    img_np = img_item.astype(np.uint8)
                    image_hw = img_np.shape[:2]
                    processed_list.append(img_np)
                else:
                    raise TypeError(f"Unsupported image type in list: {type(img_item)}")
                image_hws.append(image_hw)
            input_data = processed_list
        else:
            raise TypeError("Input must be a list of images or a path to a directory.")

        predictions = self.inferencer(
            inputs=input_data,
            out_dir=kwargs.get('out_dir', ''),
            batch_size=kwargs.get('batch_size', self.batch_size)
        )['predictions']
        out = [self.out_to_seg(pred, hw) for pred, hw in zip(predictions, image_hws)]

        # TODO: Visualization

        return out

    def identify_for_video(
        self,
        video: Union[
            Iterator[Union[np.ndarray, torch.Tensor]],
            List[Union[np.ndarray, torch.Tensor]],
        ],
        batch_size: int = 1,
    ) -> Iterator[List[Optional[ObjectDetectionResultI]]]:
        raise NotImplementedError

    def to(self, device: Union[str, torch.device]):
        self.inferencer = DetInferencer(
            model=self.model,
            weights=self.weights,
            device=device
        )

    def set_threshold(self, threshold: float):
        raise NotImplementedError

    def __str__(self):
        return self.model
