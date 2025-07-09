from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from ultralytics.engine.results import Results

from graid.data.ImageLoader import ImageDataset
from graid.interfaces.ObjectDetectionI import (
    ObjectDetectionModelI,
    ObjectDetectionResultI,
    ObjectDetectionUtils,
)
from graid.models.Ultralytics import Yolo
from graid.utilities.common import get_default_device


class ObjectDetectionMeasurements:
    """
    Types of measurements we will report:
        - mAP - mean average precision
        - Precision
        - Recall
        - Number of detections
        - Number of detections per class
        - IoU per class
        - IoU per image (over all classes for that image)
    """

    def __init__(
        self,
        model: ObjectDetectionModelI,
        dataset: ImageDataset,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the ObjectDetectionMeasurements object.

        Args:
            model (ObjectDetectionModelI): Object detection model to use.
            dataset (ImageDataset): Dataset to use for measurements.
            batch_size (int, optional): Batch size for data loader. Defaults to 1.
            collate_fn (function, optional): Function to use for collating batches.
                Defaults to None.
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def iter_measurements(
        self,
        bbox_offset: int = 0,
        class_metrics: bool = False,
        extended_summary: bool = False,
        debug: bool = False,
        fake_boxes: bool = False,
        **kwargs,
    ) -> Iterator[Union[List[Dict], Tuple[List[Dict], List[Results]]]]:
        if self.collate_fn is not None:
            data_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
        else:
            data_loader = DataLoader(
                self.dataset, batch_size=self.batch_size, shuffle=False
            )

        for batch in data_loader:
            x = torch.stack([sample["image"] for sample in batch])
            y = [sample["labels"] for sample in batch]

            x = x.to(device=get_default_device())
            if isinstance(self.model, Yolo):
                # Convert RGB to BGR because Ultralytics YOLO expects BGR
                # https://github.com/ultralytics/ultralytics/issues/9912
                x = x[:, [2, 1, 0], ...]
                x = x / 255.0
                prediction = self.model.identify_for_image(x, debug=debug, **kwargs)
                # undo the conversion
                x = x[:, [2, 1, 0], ...]
                x = x * 255.0
            else:
                self.model.to(device=get_default_device())
                prediction = self.model.identify_for_image(x, debug=debug)
                self.model.to(device="cpu")

            x = x.cpu()

            results = []
            ims = []
            for idx, (odrs, gt) in enumerate(
                zip(prediction, y)
            ):  # odr = object detection result, gt = ground truth
                measurements: dict = self._calculate_measurements(
                    odrs,
                    gt,
                    class_metrics=class_metrics,
                    extended_summary=extended_summary,
                    image=x[idx],
                    penalize_for_extra_predicitions=fake_boxes,
                )
                full_image_result = dict()
                full_image_result["image"] = x[idx]
                full_image_result["measurements"] = measurements
                full_image_result["predictions"] = odrs
                full_image_result["labels"] = gt
                results.append(full_image_result)
                if debug:
                    im = self._show_debug_image(x[idx], gt, bbox_offset)
                    ims.append(im)

            yield results

    def _show_debug_image(
        self,
        image: torch.Tensor,
        gt: List[ObjectDetectionResultI],
        bbox_offset: int = 0,
    ) -> Results:
        names = {}
        boxes = []
        for ground_truth in gt:
            cls = ground_truth.cls
            label = ground_truth.label
            names[cls] = label
            box = ground_truth.as_ultra_box.xyxy.tolist()[0]
            # box = ground_truth[0].as_ultra_box.xyxy.tolist()[
            #     0
            # ]  # TODO: fix this hack. BDD GT is a tuple of (ODR, attributes, timestamp) but we can preprocess and drop the attributes and timestamp
            box[1] += bbox_offset
            box[3] += bbox_offset
            # box += [ground_truth[0].score, ground_truth[0].cls]
            box += [ground_truth.score, ground_truth.cls]
            boxes.append(torch.tensor(box))

        boxes = torch.stack(boxes)

        ObjectDetectionUtils.show_image_with_detections(
            Image.fromarray((image.permute(1, 2, 0).cpu().numpy()).astype(np.uint8)),
            gt,
        )

        return None

    def _calculate_measurements(
        self,
        odr: List[ObjectDetectionResultI],
        gt: List[ObjectDetectionResultI],
        class_metrics: bool,
        extended_summary: bool,
        image: Optional[torch.Tensor] = None,
        penalize_for_extra_predicitions: bool = False,
    ) -> Dict:
        return ObjectDetectionUtils.compute_metrics_for_single_img(
            ground_truth=gt,
            # ground_truth=[  # TODO: this should be done by the caller all the way up
            #     res[0] for res in gt
            # ],  # BDD GT is a tuple of (ODR, attributes, timestamp)
            predictions=odr,
            class_metrics=class_metrics,
            extended_summary=extended_summary,
            image=image,
            penalize_for_extra_predicitions=penalize_for_extra_predicitions,
        )
