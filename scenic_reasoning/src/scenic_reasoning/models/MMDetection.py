from pathlib import Path
from typing import Iterator, List, Optional, Union

import mmdet
import numpy as np
import torch
from mmdet.apis import DetInferencer, inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env
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
from scenic_reasoning.utilities.coco import coco_label
from scenic_reasoning.utilities.common import get_default_device


class MMdetection_obj(ObjectDetectionModelI):
    def __init__(self, config_file: str, checkpoint_file, **kwargs) -> None:
        device = "cpu"  # Using mps will error, see: https://github.com/open-mmlab/mmdetection/issues/11794
        if torch.cuda.is_available():
            device = "cuda"

        self._model = init_detector(config_file, checkpoint_file, device=device)
        self.model_name = config_file

        # set class_agnostic to True to avoid overlaps: https://github.com/open-mmlab/mmdetection/issues/6254
        self._model.test_cfg.rcnn.nms.class_agnostic = True

        

    def collect_env(self):
        """Collect the information of the running environments."""
        env_info = collect_base_env()
        env_info["MMDetection"] = f"{mmdet.__version__}+{get_git_hash()[:7]}"
        return env_info

    def identify_for_image(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs,
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
        image_list = [
            image[i].permute(1, 2, 0).cpu().numpy() for i in range(len(image))
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
                ObjectDetectionUtils.show_image_with_detections(
                    Image.fromarray(curr_img), all_objects[i]
                )

        return all_objects

    def identify_for_image_as_tensor(
        self,
        image: Union[
            str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor
        ],
        debug: bool = False,
        **kwargs,
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
    
    def set_threshold(self, threshold: float):
        self._model.cfg.model.test_cfg.rcnn.score_thr = threshold

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
