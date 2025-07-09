import sys
from typing import Iterator, List, Optional, Union

import numpy as np
import torch
from torchvision.transforms import ToPILImage

from graid.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionModelI,
    ObjectDetectionResultI,
)
from graid.utilities.coco import coco_labels

# fmt: off
from graid.utilities.common import project_root_dir

sys.path.insert(0, str(project_root_dir() / "install" / "DINO"))
sys.path.insert(0, str(project_root_dir() / "install" ))
from DINO.datasets import transforms as T
from DINO.main import build_model_main
from DINO.util import box_ops
from DINO.util.slconfig import SLConfig

# fmt: on


to_coco = {
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 8,
    "10": 9,
    "11": 10,
    "13": 11,
    "14": 12,
    "15": 13,
    "16": 14,
    "17": 15,
    "18": 16,
    "19": 17,
    "20": 18,
    "21": 19,
    "22": 20,
    "23": 21,
    "24": 22,
    "25": 23,
    "27": 24,
    "28": 25,
    "31": 26,
    "32": 27,
    "33": 28,
    "34": 29,
    "35": 30,
    "36": 31,
    "37": 32,
    "38": 33,
    "39": 34,
    "40": 35,
    "41": 36,
    "42": 37,
    "43": 38,
    "44": 39,
    "46": 40,
    "47": 41,
    "48": 42,
    "49": 43,
    "50": 44,
    "51": 45,
    "52": 46,
    "53": 47,
    "54": 48,
    "55": 49,
    "56": 50,
    "57": 51,
    "58": 52,
    "59": 53,
    "60": 54,
    "61": 55,
    "62": 56,
    "63": 57,
    "64": 58,
    "65": 59,
    "67": 60,
    "70": 61,
    "72": 62,
    "73": 63,
    "74": 64,
    "75": 65,
    "76": 66,
    "77": 67,
    "78": 68,
    "79": 69,
    "80": 70,
    "81": 71,
    "82": 72,
    "84": 73,
    "85": 74,
    "86": 75,
    "87": 76,
    "88": 77,
    "89": 78,
    "90": 79,
}


class DINO_IDEA(ObjectDetectionModelI):
    def __init__(self, config_file: str, checkpoint_file, **kwargs) -> None:
        args = SLConfig.fromfile(config_file)

        device = "cpu"  # Using mps will error, see: https://github.com/open-mmlab/mmdetection/issues/11794
        if torch.cuda.is_available():
            device = "cuda"
            args.device = "cuda"

        model, criterion, postprocessors = build_model_main(args)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        self._model = model.to(device)
        self._model.eval()
        self._postprocessors = postprocessors
        self.transform = T.Compose(
            [
                T.RandomResize([1333], max_size=1333),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Use the same normalization as in DINO
            ]
        )

        # set class_agnostic to True to avoid overlaps: https://github.com/open-mmlab/mmdetection/issues/6254
        # self._model.test_cfg.rcnn.nms.class_agnostic = True

    def identify_for_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        debug: bool = False,
        verbose: bool = False,
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

        # if isinstance(image, np.ndarray):
        #     pass
        # elif isinstance(image, torch.Tensor):
        #     image = image.cpu().numpy()
        images = None
        if image.ndim == 3:
            img_tensor, _ = self.transform(image, None)
            images = img_tensor.unsqueeze(0)  # (1, C, H, W)
        elif image.ndim == 4:
            # already a batch of images
            images = []
            for img in image:
                img = ToPILImage()(img)
                img_tensor, _ = self.transform(img, None)
                images.append(img_tensor)
            images = torch.stack(images)
        else:
            raise ValueError(
                f"Input image should be either a numpy array or a tensor of shape (B, C, H, W), but got {image.shape}"
            )

        outputs = self._model.cuda()(images.cuda())
        predictions = self._postprocessors["bbox"](
            outputs, torch.Tensor([[1.0, 1.0]]).cuda()
        )

        if len(predictions) == 0:
            return []

        formatted_results = []
        for y_hat in predictions:

            result_for_image = []
            scores = y_hat["scores"]
            boxes = y_hat["boxes"]  # is in xyxy format
            labels = y_hat["labels"]

            if boxes is None or len(boxes) == 0:
                formatted_results.append([])
                continue

            for i in range(len(boxes)):
                image_hw = (int(images.shape[2]), int(images.shape[3]))
                box = boxes[i].tolist()
                box[0] = int(box[0] * image_hw[1])
                box[1] = int(box[1] * image_hw[0])
                box[2] = int(box[2] * image_hw[1])
                box[3] = int(box[3] * image_hw[0])

                score = scores[i].item()
                label = str(labels[i].item())
                label = to_coco[label]

                odr = ObjectDetectionResultI(
                    score=score,
                    cls=label,
                    label=coco_labels[label],
                    bbox=box,
                    image_hw=image_hw,
                    bbox_format=BBox_Format.XYXY,
                )

                result_for_image.append(odr)

            formatted_results.append(result_for_image)

        return formatted_results

    def identify_for_image_batch(
        self,
        image: Union[np.ndarray, torch.Tensor],
        debug: bool = False,
        verbose: bool = False,
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
        return self.identify_for_image(
            image=image, debug=debug, verbose=verbose, **kwargs
        )

    def identify_for_video(
        self,
        video: Union[
            Iterator[Union[np.ndarray, torch.Tensor]],
            List[Union[np.ndarray, torch.Tensor]],
        ],
        batch_size: int = 1,
    ) -> Iterator[List[Optional[ObjectDetectionResultI]]]:
        raise NotImplementedError(
            "DINO_IDEA does not support video identification directly. "
            "Please use identify_for_image for each frame in the video."
        )

    def to(self, device: Union[str, torch.device]):
        self._model.to(device)
