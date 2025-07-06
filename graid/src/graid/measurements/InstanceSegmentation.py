from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from ultralytics.engine.results import Results

from graid.data.ImageLoader import ImageDataset
from graid.interfaces.InstanceSegmentationI import (
    InstanceSegmentationModelI,
    InstanceSegmentationResultI,
    InstanceSegmentationUtils,
)
from graid.models.Ultralytics import Yolo, Yolo_seg
from graid.utilities.common import get_default_device

# TODO: torch metrics and validate comparison methods
#       implement onto YOLO and other datasets


class InstanceSegmentationMeasurements:
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
        model: InstanceSegmentationModelI,
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
        class_metrics: bool = False,
        extended_summary: bool = False,
        debug: bool = False,
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
                x = x[:, [2, 1, 0], ...]
                x = x * 255.0
            else:
                self.model.to(device=get_default_device())
                prediction = self.model.identify_for_image(x, debug=debug)
                self.model.to(device="cpu")

            results = []
            ims = []
            for idx, (isrs, gt) in enumerate(
                zip(prediction, y)
            ):  # isr = instance segmentation result, gt = ground truth

                measurements: dict = self._calculate_measurements(
                    isrs,
                    gt,
                    class_metrics=class_metrics,
                    extended_summary=extended_summary,
                )
                results.append(measurements)
                if debug:
                    im = self._show_debug_image(x[idx], gt)
                    ims.append(im)

            if debug:
                yield results, ims
            else:
                yield results

    def _show_debug_image(
        self, image: torch.Tensor, gt: List[InstanceSegmentationResultI]
    ) -> Results:

        import tempfile

        names = {}
        masks = []
        fake_boxes = []

        for ground_truth in gt:
            cls = ground_truth._class
            label = ground_truth._label
            names[cls] = label
            mask = ground_truth._bitmask
            masks.append(mask.tensor)
            fake_boxes.append(torch.tensor([0, 0, 0, 0, 0.0, -1]))

        masks = torch.cat(masks) if masks else torch.tensor([])
        fake_boxes = (
            torch.stack(fake_boxes)
            if fake_boxes
            else torch.tensor([0, 0, 0, 0, 0.0, -1])
        )

        im = Results(
            orig_img=image.unsqueeze(0),  # Add batch dimension
            path=tempfile.mktemp(suffix=".jpg"),
            names=names,
            masks=masks,
            boxes=fake_boxes,
        )

        # TODO: add class label display for ins seg gt

        im.show(boxes=False)

        return im

    def _calculate_measurements(
        self,
        isr: List[InstanceSegmentationResultI],
        gt: List[InstanceSegmentationResultI],
        class_metrics: bool,
        extended_summary: bool,
    ) -> Dict:
        return InstanceSegmentationUtils.compute_metrics_for_single_img(
            ground_truth=gt,
            # ground_truth=[  # TODO: this should be done by the caller all the way up
            #     res[0] for res in gt
            # ],  # BDD GT is a tuple of (ODR, attributes, timestamp)
            predictions=isr,
            class_metrics=class_metrics,
            extended_summary=extended_summary,
        )
