import cv2
import torch
from detectron2.utils.logger import setup_logger
from scenic_reasoning.utilities.common import get_default_device

setup_logger()
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import BitMasks
from detectron2.utils.visualizer import Visualizer
from scenic_reasoning.interfaces.InstanceSegmentationI import (
    InstanceSegmentationResultI,
    Mask_Format,
)
from PIL import Image
from scenic_reasoning.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionModelI,
    ObjectDetectionResultI,
)


class Detectron2InstanceSegmentation:
    def __init__(
        self,
        config_file: str,
        weights_file: str,
        device: Optional[str] = None,
        threshold: float = 0.1,
    ):
        cfg = get_cfg()
        if device is None:
            device = get_default_device()
        cfg.MODEL.DEVICE = device
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # new method for running the model for prediction
    def predict(self, image_tensor: torch.Tensor):
        image = image_tensor.permute(
            1, 2, 0
        ).numpy()  # Convert from (C, H, W) to (H, W, C)
        outputs = self.predictor(image)
        instances = outputs["instances"]
        results = []

        for idx in range(len(instances)):
            results.append(
                {
                    "score": instances.scores[idx].item(),
                    "cls": instances.pred_classes[idx].item(),
                    "bbox": instances.pred_boxes[idx].tensor.numpy()[
                        0
                    ],  # Convert bbox to array
                }
            )

        return results

    # method to see the segmentation on the image
    def visualize(self, im):  # image_tensor: torch.Tensor):
        # im = image_tensor.permute(1, 2, 0).numpy()
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], self.metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()

    def identify_for_tensor(
        self, image_tensor: torch.Tensor
    ) -> List[InstanceSegmentationResultI]:
        # Convert PyTorch tensor (C, H, W) to OpenCV format (H, W, C)
        image = image_tensor.permute(1, 2, 0).numpy()
        height, width = image.shape[:2]
        outputs = self.predictor(image)
        instances = outputs["instances"]
        results = []

        for idx in range(len(instances)):
            mask = BitMasks(instances.pred_masks[idx].unsqueeze(0))
            result = InstanceSegmentationResultI(
                score=instances.scores[idx].item(),
                cls=instances.pred_classes[idx].item(),
                label=self.metadata.thing_classes[instances.pred_classes[idx].item()],
                instance_id=idx,
                mask=mask,
                image_hw=(height, width),
                mask_format=Mask_Format.BITMASK,
            )
            results.append(result)

        return results
