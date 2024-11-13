
from typing import Iterator
from scenic_reasoning.data.ImageLoader import Bdd100kDataset, ImageDataset
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionModelI
from scenic_reasoning.models.UltralyticsYolo import Yolo
from scenic_reasoning.utilities.common import get_default_device
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


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
        """
        Initialize the ObjectDetectionMeasurements object.

        Args:
            model (ObjectDetectionModelI): Object detection model to use.
            dataset (ImageDataset): Dataset to use for measurements.
            batch_size (int, optional): Batch size for data loader. Defaults to 1.
            collate_fn (function, optional): Function to use for collating batches. Defaults to None.
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def iter_measurements(self, **kwargs) -> Iterator:
        if self.collate_fn is not None:
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        else:
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        for batch in data_loader:
            x = torch.stack([sample["image"] for sample in batch])
            y = [sample["labels"] for sample in batch]

            x = x.to(device=get_default_device())
            if isinstance(self.model, Yolo):
                prediction = self.model.identify_for_image(x, **kwargs)
            else:
                self.model.to(device=get_default_device())
                prediction = self.model.identify_for_image(x)
                self.model.to(device="cpu")

            prediction = prediction.to(device="cpu")

            results = []
            for odrs, gt in zip(prediction, y): # odr = object detection result, gt = ground truth
                measurements : dict = self._calculate_measurements(odrs, gt)
                results.append(measurements)

            yield results

        data_loader.close()
        self.model.to(device="cpu")

    def _calculate_measurements(self, odr, gt):
        raise NotImplementedError
    

def transform_image_for_yolo(image):
    # YOLO requires images to be 640x640 and 
    # if we are using batching they should be in BCHW float32 (0.0 - 1.0) format
    # https://docs.ultralytics.com/modes/predict/#inference-sources
    # resized_image = transforms.Resize((640, 640), interpolation=transforms.InterpolationMode.BICUBIC)(image)
    scaled_image = image.to(torch.float32) / 255.0
    return scaled_image

bdd = Bdd100kDataset(split="val", transform=transform_image_for_yolo) # YOLO requires images to be 640x640
# https://docs.ultralytics.com/models/yolov5/#performance-metrics
model = Yolo(model="yolov5x6u.pt") # v5 can handle 1280 while v8 can handle 640. makes no sense ><
measurements = ObjectDetectionMeasurements(model, bdd, batch_size=2, collate_fn=lambda x: x) # hacky way to avoid RuntimeError: each element in list of batch should be of equal size

for results in measurements.iter_measurements(device=get_default_device(), imgsz=(720, 1280)):
    print(results)
