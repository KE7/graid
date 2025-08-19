from collections.abc import Iterator
from typing import Optional, Union

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion

from graid.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionModelI,
    ObjectDetectionResultI,
)
from graid.models.Detectron import Detectron_obj
from graid.models.MMDetection import MMdetection_obj
from graid.models.Ultralytics import Yolo
from graid.utilities.coco import coco_labels
from graid.utilities.common import convert_image_to_numpy


class WBF(ObjectDetectionModelI):
    """Weighted Box Fusion ensemble across Detectron2, Ultralytics and MMDetection models."""

    def __init__(
        self,
        detectron2_models: Optional[list["Detectron_obj"]] = None,
        ultralytics_models: Optional[list["Yolo"]] = None,
        mmdet_models: Optional[list["MMdetection_obj"]] = None,
        model_weights: Optional[list[float]] = None,
        iou_threshold: float = 0.55,
        skip_box_threshold: float = 0.0,
    ) -> None:
        """Create a new Weighted Box Fusion ensemble.

        Args:
            detectron2_models: List of Detectron2 object detection models.
            ultralytics_models: List of Ultralytics YOLO object detection models.
            mmdet_models:      List of MMDetection object detection models.
            model_weights:     Per-model weight for WBF (same ordering as the
                               concatenation of the three model lists).  If
                               ``None`` all models get uniform weight.
            iou_threshold:     IoU threshold for box matching inside WBF.
            skip_box_threshold:Boxes with *score < skip_box_threshold* will be
                               ignored by WBF.
        """
        super().__init__()

        self.detectron2_models = detectron2_models or []
        self.mmdet_models = mmdet_models or []
        self.ultralytics_models = ultralytics_models or []

        self._all_models: list[ObjectDetectionModelI] = (
            self.detectron2_models + self.mmdet_models + self.ultralytics_models
        )  # Flatten in a deterministic order so that weight list lines up

        if model_weights is None:
            self.model_weights = [1.0] * len(self._all_models)
        else:
            assert len(model_weights) == len(
                self._all_models
            ), "Length of model_weights must match total number of models."
            self.model_weights = model_weights

        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold

        self.model_name = "WBF_Ensemble"

    # ---------------------------------------------------------------------
    # Helper extraction functions
    # ---------------------------------------------------------------------
    @staticmethod
    def _normalize_boxes_basic(
        boxes: np.ndarray, image_hw: tuple[int, int]
    ) -> np.ndarray:
        """Convert absolute XYXY boxes to normalized format [0-1] without corrections."""
        h, w = image_hw
        # Ensure float32 for downstream operations
        boxes_norm = boxes.copy().astype(np.float32)
        boxes_norm[:, [0, 2]] /= w  # x coords
        boxes_norm[:, [1, 3]] /= h  # y coords
        # Clip to 0-1 just in case
        boxes_norm = np.clip(boxes_norm, 0.0, 1.0)
        return boxes_norm

    @staticmethod
    def _has_reversed_boxes(boxes: np.ndarray) -> bool:
        """Check if boxes have reversed coordinates (x1 > x2 or y1 > y2)."""
        if len(boxes) == 0:
            return False
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return np.any(x2 < x1) or np.any(y2 < y1)

    @staticmethod
    def _fix_reversed_boxes(boxes: np.ndarray) -> np.ndarray:
        """Fix reversed boxes by swapping coordinates where necessary."""
        boxes_fixed = boxes.copy()

        # Fix x coordinates where x1 > x2
        swapped_x = boxes_fixed[:, 2] < boxes_fixed[:, 0]
        if np.any(swapped_x):
            # Swap x1 and x2 for reversed boxes
            temp = boxes_fixed[swapped_x, 0].copy()
            boxes_fixed[swapped_x, 0] = boxes_fixed[swapped_x, 2]
            boxes_fixed[swapped_x, 2] = temp

        # Fix y coordinates where y1 > y2
        swapped_y = boxes_fixed[:, 3] < boxes_fixed[:, 1]
        if np.any(swapped_y):
            # Swap y1 and y2 for reversed boxes
            temp = boxes_fixed[swapped_y, 1].copy()
            boxes_fixed[swapped_y, 1] = boxes_fixed[swapped_y, 3]
            boxes_fixed[swapped_y, 3] = temp

        return boxes_fixed

    def _normalize_boxes_detectron2(
        self, boxes: np.ndarray, image_hw: tuple[int, int]
    ) -> np.ndarray:
        """Normalize boxes from Detectron2 models with targeted corrections."""
        boxes_norm = self._normalize_boxes_basic(boxes, image_hw)

        # Check if this model produces reversed boxes and fix if needed
        if self._has_reversed_boxes(boxes_norm):
            boxes_norm = self._fix_reversed_boxes(boxes_norm)

        return boxes_norm

    def _normalize_boxes_ultralytics(
        self, boxes: np.ndarray, image_hw: tuple[int, int]
    ) -> np.ndarray:
        """Normalize boxes from Ultralytics models with targeted corrections."""
        boxes_norm = self._normalize_boxes_basic(boxes, image_hw)

        # Check if this model produces reversed boxes and fix if needed
        if self._has_reversed_boxes(boxes_norm):
            boxes_norm = self._fix_reversed_boxes(boxes_norm)

        return boxes_norm

    def _normalize_boxes_mmdet(
        self, boxes: np.ndarray, image_hw: tuple[int, int]
    ) -> np.ndarray:
        """Normalize boxes from MMDetection models with targeted corrections."""
        boxes_norm = self._normalize_boxes_basic(boxes, image_hw)

        # Check if this model produces reversed boxes and fix if needed
        if self._has_reversed_boxes(boxes_norm):
            boxes_norm = self._fix_reversed_boxes(boxes_norm)

        return boxes_norm

    def _extract_detectron2_raw_predictions(
        self, model: "Detectron_obj", image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract raw predictions (boxes, scores, classes) from a Detectron2 model.

        NOTE: Detectron2 simplifies inference and returns post-NMS instances by
        default.  For a quick implementation we temporarily raise the NMS
        threshold to 1.0, which effectively disables NMS while keeping the
        existing pipeline intact.
        """
        # Backup original thresholds
        orig_nms = model.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        orig_score_thr = model.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST

        # Disable NMS & lower score threshold to capture as many boxes as possible
        model.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 1.0
        model.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
        # Re-create predictor with new cfg (cheap – only wraps forward pass)
        predictor = model._predictor.__class__(model.cfg)

        outputs = predictor(image)  # dict with key "instances"
        instances = outputs.get("instances", None)

        # Restore cfg (important for subsequent calls)
        model.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = orig_nms
        model.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = orig_score_thr
        model._predictor = predictor.__class__(model.cfg)  # revert predictor

        if instances is None or len(instances) == 0:
            return (np.empty((0, 4), dtype=np.float32), np.empty(0), np.empty(0))

        boxes = instances.pred_boxes.tensor.cpu().numpy().astype(np.float32)
        scores = instances.scores.cpu().numpy().astype(np.float32)
        classes = instances.pred_classes.cpu().numpy().astype(int)
        return boxes, scores, classes

    def _extract_ultralytics_raw_predictions(
        self, model: "Yolo", image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract raw predictions from Ultralytics YOLO before NMS."""
        # Ensure the model predictor is initialised
        model._model.predict(image, verbose=False)  # warm-up call (does NMS)

        # Pre-process like Ultralytics internal pipeline
        im_tensor = model._model.predictor.preprocess(np.array(image)[np.newaxis, ...])
        infer_out = model._model.predictor.inference(im_tensor)
        # Ultralytics may return (preds, proto) or (preds, proto, loss). Handle both cases.
        if isinstance(infer_out, tuple):
            preds = infer_out[0]
        else:
            preds = infer_out

        if preds is None or len(preds) == 0:
            return (np.empty((0, 4), dtype=np.float32), np.empty(0), np.empty(0))

        # `preds` has shape (batch, num_boxes, 6) OR (batch, 1, num_boxes, 6) depending on model.
        pred0 = preds[0]  # take first batch element
        if pred0.ndim == 3 and pred0.shape[0] == 1:
            # Some models return shape (1, num_boxes, 6)
            pred0 = pred0[0]
        elif pred0.ndim == 3 and pred0.shape[-1] == 6:
            # shape (1, num_boxes, 6) or (num_levels, num_boxes, 6)
            pred0 = pred0.reshape(-1, 6)  # Flatten any leading dims

        boxes = pred0[:, :4].cpu().numpy().astype(np.float32)
        scores = pred0[:, 4].cpu().numpy().astype(np.float32)
        classes = pred0[:, 5].cpu().numpy().astype(int)

        return boxes, scores, classes

    def _extract_mmdet_raw_predictions(
        self, model: "MMdetection_obj", image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract raw predictions from MMDetection model prior to NMS."""
        from mmdet.apis import (
            inference_detector,  # local import to avoid heavy dep if unused
        )

        cfg_test = model._model.cfg.model.test_cfg
        original_nms = None
        try:
            # Two-stage detectors often keep NMS here
            original_nms = cfg_test.rcnn.nms.iou_threshold  # type: ignore
            cfg_test.rcnn.nms.iou_threshold = 1.0  # type: ignore
        except AttributeError:
            # Single-stage / transformer models may store it directly under test_cfg
            if hasattr(cfg_test, "nms") and hasattr(cfg_test.nms, "iou_threshold"):
                original_nms = cfg_test.nms.iou_threshold  # type: ignore
                cfg_test.nms.iou_threshold = 1.0  # type: ignore

        predictions = inference_detector(model._model, image)
        pred = predictions[0] if isinstance(predictions, list) else predictions

        if original_nms is not None:
            try:
                cfg_test.rcnn.nms.iou_threshold = original_nms  # type: ignore
            except AttributeError:
                if hasattr(cfg_test, "nms"):
                    cfg_test.nms.iou_threshold = original_nms  # type: ignore

        instances = pred.pred_instances
        boxes = instances.bboxes.cpu().numpy().astype(np.float32)
        scores = instances.scores.cpu().numpy().astype(np.float32)
        classes = instances.labels.cpu().numpy().astype(int)
        return boxes, scores, classes

    # ------------------------------------------------------------------
    # Core ensemble routine
    # ------------------------------------------------------------------
    def _gather_all_predictions(
        self, image: np.ndarray
    ) -> tuple[list[list[float]], list[list[float]], list[list[int]]]:
        """Collect raw predictions from every child model, in order."""
        all_boxes: list[list[float]] = []
        all_scores: list[list[float]] = []
        all_labels: list[list[int]] = []

        # Detectron2 models -------------------------------------------
        for mdl in self.detectron2_models:
            boxes, scores, classes = self._extract_detectron2_raw_predictions(
                mdl, image
            )
            if len(boxes) > 0:
                h, w = image.shape[:2]
                normalized = self._normalize_boxes_detectron2(boxes, (h, w))
                flat_boxes = [
                    [
                        (
                            float(coord[0])
                            if isinstance(coord, (list, tuple, np.ndarray))
                            else float(coord)
                        )
                        for coord in b
                    ]
                    for b in normalized.tolist()
                ]
                all_boxes.append(flat_boxes)
                all_scores.append(
                    [
                        (
                            float(s[0])
                            if isinstance(s, (list, tuple, np.ndarray))
                            else float(s)
                        )
                        for s in scores.tolist()
                    ]
                )
                all_labels.append(
                    [
                        (
                            int(c[0])
                            if isinstance(c, (list, tuple, np.ndarray))
                            else int(c)
                        )
                        for c in classes.tolist()
                    ]
                )

        # Ultralytics models ------------------------------------------
        for mdl in self.ultralytics_models:
            boxes, scores, classes = self._extract_ultralytics_raw_predictions(
                mdl, image
            )
            if len(boxes) > 0:
                h, w = image.shape[:2]
                normalized = self._normalize_boxes_ultralytics(boxes, (h, w))
                flat_boxes = [
                    [
                        (
                            float(coord[0])
                            if isinstance(coord, (list, tuple, np.ndarray))
                            else float(coord)
                        )
                        for coord in b
                    ]
                    for b in normalized.tolist()
                ]
                all_boxes.append(flat_boxes)
                all_scores.append(
                    [
                        (
                            float(s[0])
                            if isinstance(s, (list, tuple, np.ndarray))
                            else float(s)
                        )
                        for s in scores.tolist()
                    ]
                )
                all_labels.append(
                    [
                        (
                            int(c[0])
                            if isinstance(c, (list, tuple, np.ndarray))
                            else int(c)
                        )
                        for c in classes.tolist()
                    ]
                )

        # MMDetection models ------------------------------------------
        for mdl in self.mmdet_models:
            boxes, scores, classes = self._extract_mmdet_raw_predictions(mdl, image)
            if len(boxes) > 0:
                h, w = image.shape[:2]
                normalized = self._normalize_boxes_mmdet(boxes, (h, w))
                flat_boxes = [
                    [
                        (
                            float(coord[0])
                            if isinstance(coord, (list, tuple, np.ndarray))
                            else float(coord)
                        )
                        for coord in b
                    ]
                    for b in normalized.tolist()
                ]
                all_boxes.append(flat_boxes)
                all_scores.append(
                    [
                        (
                            float(s[0])
                            if isinstance(s, (list, tuple, np.ndarray))
                            else float(s)
                        )
                        for s in scores.tolist()
                    ]
                )
                all_labels.append(
                    [
                        (
                            int(c[0])
                            if isinstance(c, (list, tuple, np.ndarray))
                            else int(c)
                        )
                        for c in classes.tolist()
                    ]
                )

        return all_boxes, all_scores, all_labels

    def _fuse_predictions(
        self, image: np.ndarray
    ) -> dict[str, Union[np.ndarray, list[float], list[int]]]:
        """Run Weighted Box Fusion on a single image and return fused detections."""
        all_boxes, all_scores, all_labels = self._gather_all_predictions(image)

        if not all_boxes:
            return {
                "boxes": np.empty((0, 4)),
                "scores": np.empty(0),
                "labels": np.empty(0, dtype=int),
            }

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes,
            all_scores,
            all_labels,
            weights=self.model_weights,
            iou_thr=self.iou_threshold,
            skip_box_thr=self.skip_box_threshold,
        )

        # Convert back to pixel coordinates
        h, w = image.shape[:2]
        if len(fused_boxes) > 0:
            fused_boxes[:, [0, 2]] *= w  # x coords
            fused_boxes[:, [1, 3]] *= h  # y coords

        return {
            "boxes": fused_boxes,
            "scores": fused_scores,
            "labels": fused_labels.astype(int),
        }

    # ------------------------------------------------------------------
    # Public API (ObjectDetectionModelI)
    # ------------------------------------------------------------------
    def identify_for_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        debug: bool = False,
        **kwargs,
    ) -> list[ObjectDetectionResultI]:
        image_np = convert_image_to_numpy(image)
        fused = self._fuse_predictions(image_np)

        boxes = fused["boxes"]
        scores = fused["scores"]
        labels = fused["labels"]

        results: list[ObjectDetectionResultI] = []
        for box, score, cls_id in zip(boxes, scores, labels):
            results.append(
                ObjectDetectionResultI(
                    score=float(score),
                    cls=int(cls_id),
                    label=coco_labels.get(int(cls_id), str(int(cls_id))),
                    bbox=box.tolist(),
                    image_hw=image_np.shape[:2],
                    bbox_format=BBox_Format.XYXY,
                )
            )

        # Optionally visualize – left for caller or debug flag
        if debug and len(results) == 0:
            print("[WBF] No detections for this image.")

        return results

    def identify_for_image_batch(
        self,
        image: Union[np.ndarray, torch.Tensor],
        debug: bool = False,
        **kwargs,
    ) -> list[list[ObjectDetectionResultI]]:
        if isinstance(image, torch.Tensor):
            # Assume batch shape (B, C, H, W) or (C, H, W)
            if image.ndimension() == 3:
                return [self.identify_for_image(image, debug=debug)]
            elif image.ndimension() == 4:
                return [self.identify_for_image(img, debug=debug) for img in image]
            else:
                raise ValueError("Unsupported tensor shape for batch images")
        elif isinstance(image, list):
            return [self.identify_for_image(img, debug=debug) for img in image]
        else:
            # Single image numpy array
            return [self.identify_for_image(image, debug=debug)]

    def identify_for_video(
        self,
        video: Union[
            Iterator[Union[np.ndarray, torch.Tensor]],
            list[Union[np.ndarray, torch.Tensor]],
        ],
        batch_size: int = 1,
    ) -> Iterator[list[Optional[ObjectDetectionResultI]]]:
        # Simple implementation: iterate frame-by-frame (no batching)
        for frame in video:
            yield self.identify_for_image(frame)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def to(self, device: Union[str, torch.device]):
        """Move underlying models to given device."""
        for mdl in self._all_models:
            mdl.to(device)

    def set_threshold(self, threshold: float):
        """Set skip_box_threshold (score threshold) for WBF."""
        self.skip_box_threshold = threshold

    def __str__(self):
        return self.model_name
