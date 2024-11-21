import json
import os
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

from scenic_reasoning.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionResultI,
)
from scenic_reasoning.utilities.common import project_root_dir
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image


class ImageDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        img_dir: str,
        transform: Union[Callable, None] = None,
        target_transform: Union[Callable, None] = None,
        merge_transform: Union[Callable, None] = None,
        use_extended_annotations: bool = False,
    ):
        self.img_lables = json.load(open(annotations_file))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.merge_transform = merge_transform
        self.use_extended_annotations = use_extended_annotations

    def __len__(self) -> int:
        return len(self.img_lables)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = os.path.join(self.img_dir, self.img_lables[idx]["name"])
        image: Tensor = decode_image(img_path)
        labels = self.img_lables[idx]["labels"]
        attributes = self.img_lables[idx]["attributes"]
        timestamp = self.img_lables[idx]["timestamp"]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        if self.merge_transform:
            if self.use_extended_annotations:
                image, labels, attributes, timestamp = self.merge_transform(
                    image, labels, attributes, timestamp
                )
            else:
                image, labels = self.merge_transform(
                    image, labels, attributes, timestamp
                )
                return {
                    "image": image,
                    "labels": labels,
                }

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

    _CATEGORIES_TO_COCO = {
        "pedestrian": 0,  # in COCO there is no pedestrian so map to person
        "person": 0,
        "rider": 0,  # in COCO there is no rider so map to person
        "car": 2,
        "truck": 7,
        "bus": 5,
        "train": 6,
        "motorcycle": 3,
        "bicycle": 1,
        "traffic light": 9,
        "traffic sign": 11,  # in COCO there is no traffic sign. closest is a stop sign
        "sidewalk": 0,  # in COCO there is no sidewalk so map to person
    }

    _CATEGORIES = {
        "pedestrian": 0,
        "person": 1,
        "rider": 2,
        "car": 3,
        "truck": 4,
        "bus": 5,
        "train": 6,
        "motorcycle": 7,
        "bicycle": 8,
        "traffic light": 9,
        "traffic sign": 10,
        "sidewalk": 11,
    }

    def category_to_cls(self, category: str) -> int:
        return self._CATEGORIES[category]

    def category_to_coco_cls(self, category: str) -> int:
        return self._CATEGORIES_TO_COCO[category]

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        use_original_categories: bool = True,
        use_extended_annotations: bool = True,
        **kwargs,
    ):

        root_dir = project_root_dir() / "data" / "bdd100k"
        img_dir = root_dir / "images" / "100k" / split
        annotations_file = root_dir / "labels" / "det_20" / f"det_{split}.json"

        def merge_transform(
            image: Tensor,
            labels: List[Dict[str, Any]],
            attributes: Dict[str, Any],
            timestamp: str,
        ) -> Union[
            Tuple[Tensor, List[ObjectDetectionResultI]],
            Tuple[
                Tensor,
                List[Tuple[ObjectDetectionResultI, Dict[str, Any], str]],
                Dict[str, Any],
                str,
            ],
        ]:
            results = []

            for label in labels:
                channels, height, width = image.shape
                if use_original_categories:
                    cls = self.category_to_cls(label["category"])
                    res_label = label["category"]
                else:
                    cls = self.category_to_coco_cls(label["category"])
                    # handle the case where exact category is not in COCO aka different names for people
                    res_label = label["category"] if cls != 0 else "person"

                odr = ObjectDetectionResultI(
                    score=1.0,
                    cls=cls,
                    label=res_label,
                    bbox=[
                        label["box2d"]["x1"],
                        label["box2d"]["y1"],
                        label["box2d"]["x2"],
                        label["box2d"]["y2"],
                    ],
                    image_hw=(height, width),
                    bbox_format=BBox_Format.XYXY,
                    attributes=label["attributes"],
                )

                if use_extended_annotations:
                    results.append(
                        (
                            odr,
                            label["attributes"],
                            timestamp,
                        )
                    )
                else:
                    results.append(odr)

            if use_extended_annotations:
                return (image, results, attributes, timestamp)
            else:
                return (image, results)

        super().__init__(
            str(annotations_file),
            str(img_dir),
            merge_transform=merge_transform,
            use_extended_annotations=use_extended_annotations,
            **kwargs,
        )
