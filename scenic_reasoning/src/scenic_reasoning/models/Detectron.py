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

from InstanceSegmentation import InstanceSegmentationResultI, Mask_Format  
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionResultI, BBox_Format 

# changes are definitely needed, idk what yet

class Detectron2InstanceSegmentation:
    def __init__(self, model_path: str, config_file: str):
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def identify_for_image(self, image_path: str) -> List[InstanceSegmentationResultI]:
        image = cv2.imread(image_path)
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
                mask_format=Mask_Format.BITMASK
            )
            results.append(result)

        return results
