import os
import cv2
import torch
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from itertools import islice
from typing import List, Iterator, Union

from scenic_reasoning.interfaces.InstanceSegmentationI import InstanceSegmentationResultI, Mask_Format  
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionResultI, BBox_Format 


class Detectron2InstanceSegmentation:
    def __init__(self, model_path: str, config_file: str):
        cfg = get_cfg()
        #cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # new method for running the model for prediction
    def predict(self, image_tensor: torch.Tensor):
        image = image_tensor.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        outputs = self.predictor(image)
        instances = outputs["instances"]
        results = []

        for idx in range(len(instances)):
            results.append({
                "score": instances.scores[idx].item(),
                "cls": instances.pred_classes[idx].item(),
                "bbox": instances.pred_boxes[idx].tensor.numpy()[0],  # Convert bbox to array
            })

        return results
    
    def identify_for_tensor(self, image_tensor: torch.Tensor) -> List[InstanceSegmentationResultI]:
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