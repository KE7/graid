from typing import Union, List, Tuple
from pathlib import Path

import torch
import numpy as np
import cv2

from scenic_reasoning.interfaces.ObjectDetectionI import (
    ObjectDetectionModelI,
    ObjectDetectionResultI,
    BBox_Format
)
from scenic_reasoning.utilities.coco import coco_labels
from mmdet.apis import DetInferencer


Image = Union[np.ndarray, torch.Tensor, str]


class MMDetection(ObjectDetectionModelI):
    def __init__(self, **kwargs) -> None:

        # TODO: Take config_file and/or checkpoint_file as input

        self.model = kwargs.get('model', 'rtmdet_tiny_8xb32-300e_coco.py')
        self.weights = kwargs.get('weights', 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')
        self.inferencer = DetInferencer(
            model=self.model,
            weights=self.weights,
            device=kwargs.get('device', 'cpu')
        )

        self.batch_size = kwargs.get('batch_size', 1)

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
        image: Image,
        debug: bool = False,
        **kwargs
    ):

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

        return [self.out_to_obj(pred, image_hw)]

    def identify_for_image_batch(
        self,
        images: Union[List[Image], str],
        debug: bool = False,
        **kwargs,
    ):
        
        image_hws = []
        input_data = images

        if isinstance(images, str):
            image_dir = Path(images)
            
            image_paths = sorted([
                p for p in image_dir.iterdir() 
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            ])
            if not image_paths:
                return []
            
            for path in image_paths:
                image_hws.append(cv2.imread(str(path)).shape[:2])
            
            input_data = [str(p) for p in image_paths]

        elif isinstance(images, list):
            if not images:
                return []
            
            processed_list = []
            for img_item in images:
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

        return [self.out_to_obj(pred, hw) for pred, hw in zip(predictions, image_hws)]


    def to(self, device: Union[str, torch.device]):
        self.inferencer = DetInferencer(
            model=self.model,
            weights=self.weights,
            device=device
        )