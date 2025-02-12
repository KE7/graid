from itertools import islice
from pathlib import Path
from typing import Iterator, List, Optional, Union
import os
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
from mmdet.apis import DetInferencer, init_detector, inference_detector
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env
from mmdet.registry import VISUALIZERS
import mmdet
import mmcv
import mmengine

# https://github.com/HumanSignal/label-studio/blob/develop/docs/source/tutorials/object-detector.md
coco_label = {
    0: "unlabeled",
    1: "airplane",
    2: "apple",
    3: "backpack",
    4: "banana",
    5: "baseball_bat",
    6: "baseball_glove",
    7: "bear",
    8: "bed",
    9: "bench",
    10: "bicycle",
    11: "bird",
    12: "boat",
    13: "book",
    14: "bottle",
    15: "bowl",
    16: "broccoli",
    17: "bus",
    18: "cake",
    19: "car",
    20: "carrot",
    21: "cat",
    22: "cell_phone",
    23: "chair",
    24: "clock",
    25: "couch",
    26: "cow",
    27: "cup",
    28: "dining_table",
    29: "dog",
    30: "donut",
    31: "elephant",
    32: "fire_hydrant",
    33: "fork",
    34: "frisbee",
    35: "giraffe",
    36: "hair_drier",
    37: "handbag",
    38: "horse",
    39: "hot_dog",
    40: "keyboard",
    41: "kite",
    42: "knife",
    43: "laptop",
    44: "microwave",
    45: "motorcycle",
    46: "mouse",
    47: "orange",
    48: "oven",
    49: "parking_meter",
    50: "person",
    51: "pizza",
    52: "potted_plant",
    53: "refrigerator",
    54: "remote",
    55: "sandwich",
    56: "scissors",
    57: "sheep",
    58: "sink",
    59: "skateboard",
    60: "skis",
    61: "snowboard",
    62: "spoon",
    63: "sports_ball",
    64: "stop_sign",
    65: "suitcase",
    66: "surfboard",
    67: "teddy_bear",
    68: "tennis_racket",
    69: "tie",
    70: "toaster",
    71: "toilet",
    72: "toothbrush",
    73: "traffic_light",
    74: "train",
    75: "truck",
    76: "tv",
    77: "umbrella",
    78: "vase",
    79: "wine_glass",
    80: "zebra"
}


class MMdetection_obj(ObjectDetectionModelI):
    def __init__(self, config_file: str, checkpoint_file, **kwargs) -> None:
        device = "cpu"   # Using mps will error, see: https://github.com/open-mmlab/mmdetection/issues/11794
        self._model = init_detector(config_file, checkpoint_file, device=device)

    def collect_env(self):
        """Collect the information of the running environments."""
        env_info = collect_base_env()
        env_info['MMDetection'] = f'{mmdet.__version__}+{get_git_hash()[:7]}'
        return env_info

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

        # image is batched input. MMdetection only supports Union[InputType, Sequence[InputType]], where InputType = Union[str, np.ndarray]
        image_list = [image[i].permute(1, 2, 0).cpu().numpy() for i in range(len(image))]
        image_hw = image_list[0].shape[:-1]
        predictions = inference_detector(self._model, image_list)

        all_objects = []

        for pred in predictions:
            bboxes = pred.pred_instances.bboxes.tolist()
            labels = pred.pred_instances.labels
            scores = pred.pred_instances.scores
            image_hw = pred.pad_shape   #TODO: should I use pad_shape or img_shape?
            objects = []

            for i in range(len(labels)):
                cls_id = labels[i].item()
                score = scores[i].item()
                bbox = bboxes[i]

                odr = ObjectDetectionResultI(
                        score=score,
                        cls=cls_id,
                        label=coco_label[cls_id],
                        bbox=bbox,
                        image_hw=image_hw,
                        bbox_format=BBox_Format.XYXY,
                    )
                objects.append(odr)
            all_objects.append(objects)

        if debug:
            for i in range(len(image_list)):
                ObjectDetectionUtils.show_image_with_detections(Image.fromarray(image_list[i]), all_objects[i])

        return all_objects

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
        pass

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[Optional[ObjectDetectionResultI]]]:
        pass

    def to(self, device: Union[str, torch.device]):
        pass


class MMdetection_seg(InstanceSegmentationModelI):
    def __init__(self, config_file: str, checkpoint_file, **kwargs) -> None:
        device = "cpu"   # Using mps will error, see: https://github.com/open-mmlab/mmdetection/issues/11794
        self._model = init_detector(config_file, checkpoint_file, device=device)

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
        image_list = [image[i].permute(1, 2, 0).cpu().numpy() for i in range(len(image))]
        predictions = inference_detector(self._model, image_list)

        if debug:
            # TODO: design a new visualizer
            visualizer = VISUALIZERS.build(self._model.cfg.visualizer)
            visualizer.dataset_meta = self._model.dataset_meta
            for i in range(len(image_list)):
                visualizer.add_datasample(
                    'result',
                    image_list[i],
                    data_sample=predictions[i],
                    draw_gt = None,
                    wait_time=0
                )
                visualizer.show()

        all_instances = []
        for pred in predictions:
            masks = pred.pred_instances.masks
            labels = pred.pred_instances.labels
            scores = pred.pred_instances.scores
            image_hw = pred.pad_shape   #TODO: should I use pad_shape or img_shape?
            instances = []
            for i in range(len(labels)):
                cls_id = labels[i].item()
                mask = masks[i]
                score = scores[i].item()
                instance = InstanceSegmentationResultI(
                    score=score,
                    cls=cls_id,
                    label=coco_label[cls_id],
                    instance_id=i,
                    mask=mask.unsqueeze(0),
                    image_hw=image_hw,
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