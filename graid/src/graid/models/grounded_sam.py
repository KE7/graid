from pathlib import Path
from enum import Enum, auto

import re
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
        classes: dict[str, int] | None = None,
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

        self.classes = classes or {v: k for k, v in coco_labels.items()}
        self.cls_to_idx = self.classes
        self.idx_to_cls = {v: k for k, v in self.classes.items()}

        self.labels = list(self.classes.keys())
        self.prompt = '. '.join(self.labels) + '.'

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def out_to_seg(self, out: dict, image_hw: tuple[int, int]):
        seg = []
        i = 0
        for mask, score, label in zip(
            out['masks'], out['scores'], out['labels']
        ):
            if isinstance(mask, torch.Tensor):
                decoded = mask.cpu().numpy()
            elif isinstance(mask, np.ndarray):
                decoded = mask
            else:
                raise TypeError(f'Unknown mask type: {type(mask)}')

            mask_tensor = torch.from_numpy(decoded.astype(bool)).unsqueeze(0)

            aliases = sorted(self.cls_to_idx.keys(), key=len, reverse=True)
            pattern = re.compile(r'\b(' + '|'.join(map(re.escape, aliases)) + r')\b')
            for sub_label in pattern.findall(label):
                if sub_label:
                    seg += [
                        InstanceSegmentationResultI(
                            score=float(score),
                            cls=int(self.cls_to_idx[sub_label]),
                            label=str(i),
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
        labels = results[0]["labels"]
        shape = input.shape[:2]

        out = [self.out_to_seg(
            out={
                'masks': masks,
                'scores': scores,
                'labels': labels
            },
            image_hw=(shape[0], shape[1])
        )]

        # TODO: Visualization

        return out

    def identify_for_image_batch(
        self,
        image: str | list | np.ndarray | torch.Tensor,
        debug: bool = False,
        **kwargs,
    ):
        imgs: list[np.ndarray] = []
        hws:  list[tuple[int, int]] = []

        def _load_to_np(item):
            if isinstance(item, str):
                arr = np.array(Image.open(item).convert("RGB"))
            elif isinstance(item, Image.Image):
                arr = np.array(item.convert("RGB"))
            elif isinstance(item, torch.Tensor):
                arr = item.detach().cpu().numpy()
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                if arr.ndim == 3 and arr.shape[0] in {1,3}:
                    arr = np.moveaxis(arr, 0, -1)
            else:
                arr = item
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            return arr.astype(np.uint8)

        if isinstance(image, str) and Path(image).is_dir():
            for p in sorted(Path(image).iterdir()):
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    arr = _load_to_np(str(p))
                    imgs.append(arr); hws.append(arr.shape[:2])
        elif isinstance(image, (list, tuple)):
            for item in image:
                arr = _load_to_np(item)
                imgs.append(arr); hws.append(arr.shape[:2])
        else:
            arr = _load_to_np(image)
            imgs.append(arr); hws.append(arr.shape[:2])

        if not imgs:
            return []

        prompt_labels = kwargs.get('labels', self.labels)
        prompt = '. '.join(prompt_labels) + '.'
        box_th = kwargs.get('box_threshold',  self.box_threshold)
        text_th = kwargs.get('text_threshold', self.text_threshold)

        out: list[list[InstanceSegmentationResultI]] = []

        for img_np, (h, w) in zip(imgs, hws):
            inputs = self.processor(
                images=img_np,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                dino_out = self.grounding_model(**inputs)

            det = self.processor.post_process_grounded_object_detection(
                dino_out,
                inputs.input_ids,
                box_threshold=box_th,
                text_threshold=text_th,
                target_sizes=[(h, w)]
            )[0]

            boxes   = det["boxes"].cpu().numpy()
            scores  = det["scores"].cpu().numpy().tolist()
            labels  = det["labels"]

            self.sam2_predictor.set_image(img_np)
            masks, _, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            seg = self.out_to_seg(
                out={
                    'masks': masks,
                    'scores': scores,
                    'labels': labels
                },
                image_hw=(h, w)
            )
            out.append(seg)

        return out
    
    def identify_for_video(self, video, batch_size = 1):
        return super().identify_for_video(video, batch_size)
    
    def to(self, device):

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.sam2_model = build_sam2(self.sam2_cfg, self.sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.processor = AutoProcessor.from_pretrained(self.gnd_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.gnd_model_id).to(self.device)