import json
import os
from typing import Any, Dict, List, Literal, Tuple, Union, Callable

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image

from scenic_reasoning.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionResultI,
)
from scenic_reasoning.utilities.common import project_root_dir


class ImageDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        img_dir: str,
        transform: Union[Callable, None] = None,
        target_transform: Union[Callable, None] = None,
        merge_transform: Union[Callable, None] = None,
    ):
        self.img_lables = json.load(open(annotations_file))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.merge_transform = merge_transform

    def __len__(self):
        return len(self.img_lables)

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Tensor, Dict, Dict, str]]:
        img_path = os.path.join(self.img_dir, self.img_lables[idx]["name"])
        image = decode_image(img_path)
        labels = self.img_lables[idx]["labels"]
        attributes = self.img_lables[idx]["attributes"]
        timestamp = self.img_lables[idx]["timestamp"]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        if self.merge_transform:
            image, labels, attributes, timestamp = self.merge_transform(image, labels, attributes, timestamp)

        return {
            "image": image,
            "labels": labels,
            "attributes": attributes,
            "timestamp": timestamp,
        }


class Bdd100kDataset(ImageDataset):
    """
    The structure of how BDD100K labels are stored.
    Mapping = {
        "name": "name",
        "attributes": {
            "weather": "weather",
            "timeofday": "timeofday",
            "scene": "scene"
        },
        "timestamp": "timestamp",
        "labels": [
            {
                "id": "id",
                "attributes": {
                    "occluded": "occluded",
                    "truncated": "truncated",
                    "trafficLightColor": "trafficLightColor"
                },
                "category": "category",
                "box2d": {
                    "x1": "x1",
                    "y1": "y1",
                    "x2": "x2",
                    "y2": "y2"
                }
            }
        ]
    }

    Example:
        "name": "b1c66a42-6f7d68ca.jpg",
        "attributes": {
        "weather": "overcast",
        "timeofday": "daytime",
        "scene": "city street"
        },
        "timestamp": 10000,
        "labels": [
        {
            "id": "0",
            "attributes": {
                "occluded": false,
                "truncated": false,
                "trafficLightColor": "NA"
            },
            "category": "traffic sign",
            "box2d": {
                "x1": 1000.698742,
                "y1": 281.992415,
                "x2": 1040.626872,
                "y2": 326.91156
            }
            ...
        }
    """

    def __init__(self, split: Union[Literal["train", "val", "test"]] = "train", **kwargs):

        root_dir = project_root_dir() / "data" / "bdd100k"
        img_dir = root_dir / "images" / "100k" / split
        annotations_file = root_dir / "labels" / "det_20" / f"det_{split}.json"

        def merge_transform(
            image : Tensor, 
            labels : List[Dict[str, Any]], 
            attributes : Dict[str, Any],
            timestamp : str
        ) -> Tuple[Tensor, List[Tuple[ObjectDetectionResultI, Dict[str, Any], str]], Dict[str, Any], str]:
            results = []

            for label in labels:
                channels, height, width = image.shape
                results.append(
                    (
                        ObjectDetectionResultI(
                            score=1.0,
                            cls=-1,
                            label=label["category"],
                            bbox=[
                                label["box2d"]["x1"],
                                label["box2d"]["y1"],
                                label["box2d"]["x2"],
                                label["box2d"]["y2"],
                            ],
                            image_hw=(height, width),
                            bbox_format=BBox_Format.XYXY,
                            attributes=label["attributes"],
                        ),
                        label["attributes"],
                        timestamp,
                    )
                )

            return (image, results, attributes, timestamp)

        super().__init__(annotations_file, img_dir, merge_transform=merge_transform, **kwargs)
