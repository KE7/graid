
from typing import Iterator
from scenic_reasoning.data.ImageLoader import Bdd100kDataset, ImageDataset
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionModelI
from scenic_reasoning.models.UltralyticsYolo import Yolo
import torch
from torch.utils.data import DataLoader


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
    def __init__(self, model : ObjectDetectionModelI, dataset : ImageDataset, batch_size : int = 1, collate_fn=None):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def iter_measurements(self) -> Iterator:
        if self.collate_fn is not None:
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        else:
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        for batch in data_loader:
            x, y = batch

            prediction = self.model.identify_for_image(x)

            results = []
            for odr, gt in zip(prediction, y): # odr = object detection result, gt = ground truth
                measurements : dict = self._calculate_measurements(odr, gt)
                results.append(measurements)

            yield results

    def _calculate_measurements(self, odr, gt):
        raise NotImplementedError
    

bdd = Bdd100kDataset(split="val")
model = Yolo(model="yolov8x.pt")
measurements = ObjectDetectionMeasurements(model, bdd, batch_size=2)

for results in measurements.iter_measurements():
    print(results)
