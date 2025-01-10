import torch
import tempfile
from detectron2.structures import BitMasks, pairwise_iou
import matplotlib.pyplot as plt
from scenic_reasoning.interfaces.InstanceSegmentationI import (
    InstanceSegmentationModelI,
    InstanceSegmentationResultI,
    InstanceSegmentationUtils,
)
from scenic_reasoning.data.ImageLoader import Bdd100kDataset, ImageDataset
from scenic_reasoning.models.Detectron import Detectron2InstanceSegmentation
from scenic_reasoning.data.ImageLoader import Bdd100kDataset, NuImagesDataset
from torch.utils.data import DataLoader
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union
from scenic_reasoning.models.UltralyticsYolo import Yolo
from scenic_reasoning.utilities.common import get_default_device
from torch.utils.data import DataLoader
from ultralytics.engine.results import Results

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
        bbox_offset: int = 0,
        class_metrics: bool = False,
        extended_summary: bool = False,
        debug: bool = False,
        **kwargs
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
                prediction = self.model.identify_for_image(x, debug=debug, **kwargs)
            else:
                self.model.to(device=get_default_device())
                prediction = self.model.identify_for_image(x)
                self.model.to(device="cpu")

            results = []
            ims = []
            for idx, (isrs, gt) in enumerate(
                zip(prediction, y)
            ):  # odr = object detection result, gt = ground truth
                
                
                measurements: dict = self._calculate_measurements(
                    isrs,
                    gt,
                    class_metrics=class_metrics,
                    extended_summary=extended_summary,
                )
                results.append(measurements)
                if debug:
                    im = self._show_debug_image(x[idx], gt, bbox_offset)
                    ims.append(im)

            if debug:
                yield results, ims
            else:
                yield results

    def _show_debug_image(
        self,
        image: torch.Tensor,
        gt: List[InstanceSegmentationResultI],
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

        im = Results(
            orig_img=image.unsqueeze(0),  # Add batch dimension
            path=tempfile.mktemp(suffix=".jpg"),
            names=names,
            boxes=boxes,
        )
        im.show()

        return im

    def _calculate_measurements(
        self,
        isr: List[InstanceSegmentationResultI],
        gt: List[InstanceSegmentationResultI],
        class_metrics: bool,
        extended_summary: bool,
    ) -> Dict:
        # import pdb
        # pdb.set_trace()
        return InstanceSegmentationUtils.compute_metrics_for_single_img(
            ground_truth=gt,
            # ground_truth=[  # TODO: this should be done by the caller all the way up
            #     res[0] for res in gt
            # ],  # BDD GT is a tuple of (ODR, attributes, timestamp)
            predictions=isr,
            class_metrics=class_metrics,
            extended_summary=extended_summary,
        )



# bdd_dataset = Bdd100kDataset(split="val")

# dataloader = DataLoader(bdd_dataset, batch_size=1, shuffle=False)
# detectron2_segmenter = Detectron2InstanceSegmentation()

# def compare(segmenter, dataloader):
#     results = []

#     for batch in dataloader:
#         image = batch["image"][0]  # First image in the batch
#         ground_truth_labels = batch["labels"]  # Ground truth labels

#         # Get predictions
#         predictions = segmenter.predict(image)

#         # Compare ground truth and predictions
#         for gt in ground_truth_labels:
#             gt_bbox = gt["bbox"]  # Assuming ground truth includes bounding boxes
#             best_iou, best_pred = 0, None

#             for pred in predictions:
#                 pred_bbox = pred["bbox"]
#                 iou = pairwise_iou(gt_bbox, pred_bbox)

#                 if iou > best_iou:
#                     best_iou, best_pred = iou, pred

#             results.append({
#                 "gt_label": gt["label"],
#                 "pred_label": best_pred["cls"] if best_pred else None,
#                 "iou": best_iou,
#             })

#     return results

# # Perform the benchmarking on detectron2
# results = compare(detectron2_segmenter, dataloader)

# # Summarize metrics
# mean_iou = sum(r["iou"] for r in results) / len(results)
# print(f"Mean IoU: {mean_iou:.2f}")

# for idx, result in enumerate(results):
#     print(f"Sample {idx}:")
#     print(f"  Ground Truth: {result['gt_label']}")
#     print(f"  Prediction:  {result['pred_label']}")
#     print(f"  IoU:         {result['iou']:.2f}")
