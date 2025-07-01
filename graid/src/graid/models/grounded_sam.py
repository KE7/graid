from pathlib import Path
from enum import Enum, auto

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
from graid.utilities.coco import coco_labels

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 


class GroundedSAM2(InstanceSegmentationModelI):
    def __init__(
        self,
        sam2_cfg: str,
        sam2_ckpt: str,
        gnd_model_id: str,
        classes: dict[int, str] = coco_labels,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        device: torch.device | None = None
    ):
        super().__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.sam2_cfg = sam2_cfg
        self.sam2_ckpt = sam2_ckpt
        self.gnd_model_id = gnd_model_id

        self.sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.processor = AutoProcessor.from_pretrained(gnd_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(gnd_model_id).to(self.device)

        self.idx_to_cls = classes
        self.cls_to_idx = {v: k for k, v in classes.items()}

        self.labels = list(classes.values())
        self.prompt = '. '.join(self.labels) + '.'

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def out_to_seg(self, out: dict, image_hw: tuple[int, int]):
        seg = []
        i = 0
        for mask, score, label, class_id in zip(
            out['masks'], out['scores'], out['class_names'], out['class_ids']
        ):
            if isinstance(mask, torch.Tensor):
                decoded = mask.cpu().numpy()
            elif isinstance(mask, np.ndarray):
                decoded = mask
            else:
                raise TypeError(f'Unknown mask type: {type(mask)}')

            mask_tensor = torch.from_numpy(decoded.astype(bool)).unsqueeze(0)

            seg += [
                InstanceSegmentationResultI(
                    score=float(score),
                    cls=int(class_id),
                    label=str(label),
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
        image: str | np.ndarray | torch.Tensor | Image.Image,
        debug: bool = False,
        **kwargs,
    ):
        if isinstance(image, str):
            img_pil = Image.open(image)
            input = np.array(img_pil.convert("RGB"))
        elif isinstance(image, Image.Image):
            input = np.array(image.convert("RGB"))
        elif isinstance(image, torch.Tensor):
            input = image.cpu().numpy()
        else:
            input = image

        labels = kwargs.get('labels', self.labels)
        prompt = '. '.join(labels) + '.'

        self.sam2_predictor.set_image(input)
        inputs = self.processor(
            images=input,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=kwargs.get('box_threshold', self.box_threshold),
            text_threshold=kwargs.get('text_threshold', self.text_threshold),
            target_sizes=[input.shape[:2]]
        )

        input_boxes = results[0]["boxes"].cpu().numpy()
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        scores = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = [0 for cls in class_names] # TODO: Fix non-matching classes
        # class_ids = [self.cls_to_idx[cls] for cls in results[0]["labels"]]
        shape = input.shape[:2]

        out = [self.out_to_seg(
            out={
                'masks': masks,
                'scores': scores,
                'class_names': class_names,
                'class_ids': class_ids
            },
            image_hw=(shape[0], shape[1])
        )]

        # TODO: Visualization

        return out

    def identify_for_image_batch(self, image, debug = False, **kwargs):
        return super().identify_for_image_batch(image, debug, **kwargs)
    
    def identify_for_video(self, video, batch_size = 1):
        return super().identify_for_video(video, batch_size)
    
    def to(self, device):

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.sam2_model = build_sam2(self.sam2_cfg, self.sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.processor = AutoProcessor.from_pretrained(self.gnd_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.gnd_model_id).to(self.device)