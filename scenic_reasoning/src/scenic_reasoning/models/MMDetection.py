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
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
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

        if image_list[0].dtype == np.float32:
            image_list = [i * 255 for i in image_list]
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
                curr_img = image_list[i]
                if image_list[i].dtype == np.float32:
                    curr_img = curr_img.astype(np.uint8)
                ObjectDetectionUtils.show_image_with_detections(Image.fromarray(curr_img), all_objects[i])

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