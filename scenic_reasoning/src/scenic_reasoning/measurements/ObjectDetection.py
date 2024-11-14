
from itertools import islice
from typing import Iterator
from scenic_reasoning.data.ImageLoader import Bdd100kDataset, ImageDataset
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionModelI
from scenic_reasoning.models.UltralyticsYolo import Yolo
from scenic_reasoning.utilities.common import get_default_device
import torch
from torch.utils.data import DataLoader
from ultralytics.data.augment import LetterBox


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

            results = []
            for odrs, gt in zip(prediction, y): # odr = object detection result, gt = ground truth
                measurements : dict = self._calculate_measurements(odrs, gt)
                results.append(measurements)

            yield results

        data_loader.close()
        self.model.to(device="cpu")

    def _calculate_measurements(self, odr, gt):
        total_objects = len(odr) / len(gt)
        total_objects_detected_per_class = {}
        for object in odr:
            if object.labels in total_objects_detected_per_class:
                total_objects_detected_per_class[object.labels] += 1
            else:
                total_objects_detected_per_class[object.labels] = 1

        true_total_objects_per_class = {}
        for gt_object in gt:
            if gt_object[0].labels in true_total_objects_per_class:
                true_total_objects_per_class[gt_object[0].labels] += 1
            else:
                true_total_objects_per_class[gt_object[0].labels] = 1

        percentage_objects_detected_per_class = {}
        for key, value in true_total_objects_per_class.items():
            if key in total_objects_detected_per_class:
                percentage_objects_detected_per_class[key] = total_objects_detected_per_class[key] / value
            else:
                percentage_objects_detected_per_class[key] = 0

        # add -1 for the classes that were detected but not in the ground truth
        for key, value in total_objects_detected_per_class.items():
            if key not in true_total_objects_per_class:
                percentage_objects_detected_per_class[key] = -1

        return {
            "total_objects": total_objects,
            "percentage_objects_detected": percentage_objects_detected_per_class
        }
    

# def transform_image_for_yolo(image):
#     # if we are using batching they should be in BCHW float32 (0.0 - 1.0) format
#     # https://docs.ultralytics.com/modes/predict/#inference-sources
#     # resized_image = transforms.Resize((640, 640), interpolation=transforms.InterpolationMode.BICUBIC)(image)
#     scaled_image = image.to(torch.float32) / 255.0
#     return scaled_image

shape_transform = LetterBox(new_shape=(768, 1280))
def transform_image_for_yolo(image : torch.Tensor):
    # 1) convert from tensor to cv2 image
    image_np  = image.permute(1, 2, 0).numpy()
    # 2) resize to 768x1280
    image_np = shape_transform(image=image_np)
    # 3) convert back to tensor
    image = torch.tensor(image_np).permute(2, 0, 1)
    # 4) normalize to 0-1
    image = image.to(torch.float32) / 255.0

    return image



bdd = Bdd100kDataset(split="val", transform=transform_image_for_yolo) # YOLO requires images to be 640x640 but BDD100K images are 720x1280
# https://docs.ultralytics.com/models/yolov5/#performance-metrics
model = Yolo(model="yolov5x6u.pt") # v5 can handle 1280 while v8 can handle 640. makes no sense ><
measurements = ObjectDetectionMeasurements(model, bdd, batch_size=1, collate_fn=lambda x: x) # hacky way to avoid RuntimeError: each element in list of batch should be of equal size

# WARNING ⚠️ imgsz=[720, 1280] must be multiple of max stride 64, updating to [768, 1280]
from pprint import pprint
for results in islice(measurements.iter_measurements(device=get_default_device(), imgsz=[768, 1280]), 3):
    pprint(results)
