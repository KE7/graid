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
from mmdet.apis import DetInferencer
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env
import mmdet

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
    def __init__(self, model: Union[str, Path], checkpoint, **kwargs) -> None:
        for name, val in self.collect_env().items():
            print(f'{name}: {val}')
        device = "cpu"   # Using mps will error, see: https://github.com/open-mmlab/mmdetection/issues/11794
        self._model = DetInferencer(model, checkpoint, device)

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

        out_dir = '../output'
        predictions = self._model(image_list, out_dir=out_dir)['predictions']

        # if debug:
        #     for filename in os.listdir(f"{out_dir}/vis"):
        #         file_path = os.path.join(f"{out_dir}/vis", filename)
        #         with Image.open(file_path) as img:
        #             img.show()

        formatted_results = []

        for prediction in predictions:
            result_for_image = []
            for bbox, label, score in zip(prediction['bboxes'], prediction['labels'], prediction['scores']):
                odr = ObjectDetectionResultI(
                        score=score,
                        cls=label,
                        label=coco_label[label],
                        bbox=bbox,
                        image_hw=image_hw,
                        bbox_format=BBox_Format.XYXY,
                    )
                result_for_image.append(odr)
            formatted_results.append(result_for_image)

        if debug:
            for i in len(image_list):
                ObjectDetectionUtils.show_image_with_detections(Image.fromarray(image_list[i]), formatted_results[i])

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
        pass

    def identify_for_video(
        self,
        video: Union[Iterator[Image.Image], List[Image.Image]],
        batch_size: int = 1,
    ) -> Iterator[List[Optional[ObjectDetectionResultI]]]:
        pass

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