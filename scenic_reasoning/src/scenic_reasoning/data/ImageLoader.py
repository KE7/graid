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
        self.img_labels = json.load(open(annotations_file))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.merge_transform = merge_transform
        self.use_extended_annotations = use_extended_annotations

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Tensor, Dict, Dict, str]]:
        try:
            img_path = os.path.join(self.img_dir, self.img_labels[idx]["name"])
            image = decode_image(img_path)
        except Exception as err:
            raise RuntimeError(f"Error: {img_path} failed to read, {err}")
        
        labels = self.img_labels[idx]["labels"]
        attributes = self.img_labels[idx]["attributes"]
        timestamp = self.img_labels[idx]["timestamp"]

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
    
class NuImagesDataset(ImageDataset):
    """
    The structure of how NuImages labels are stored
    nuim.table_names: 
    'attribute',
    'calibrated_sensor',
    'category',
    'ego_pose',
    'log',
    'object_ann',
    'sample',
    'sample_data',
    'sensor',
    'surface_ann'

    <v1.0-{split}/sample_data.json>, sample data label
    sample_data = {
    "token": "003bf191da774ac3b7c47e44075d9cf9",
    "sample_token": "d626e96768f44c2890c2a5693dd11ec4",
    "ego_pose_token": "2c731fd2f92b4956b15cbeed160417c1",
    "calibrated_sensor_token": "d9480acc4135525dbcffb2a0db6d7c11",
    "filename": "samples/CAM_BACK_LEFT/n013-2018-08-03-14-44-49+0800__CAM_BACK_LEFT__1533278795447155.jpg",
    "fileformat": "jpg",
    "width": 1600,
    "height": 900,
    "timestamp": 1533278795447155,
    "is_key_frame": true,
    "prev": "20974c9684ae4b5d812604e099d433e2",
    "next": "ca3edcbb46d041a4a2662d91ab68b59d"
    }

    <v1.0-{split}/object_ann.json>, sample object
    object_ann = 
    {
    "token": "251cb138f0134f038b37e272a3ff88e6",
    "category_token": "85abebdccd4d46c7be428af5a6173947",
    "bbox": [
    101,
    503,
    174,
    594
    ],
    "mask": {
    "size": [
    900,
    1600
    ],
    "counts": "Z15oMjFTbDAyTjFPMk4yTjFPMDAwMDAwMDAwMDAwMDAwMDAwMU8wTTNKNks1SjZLNUo2SzVKNks1SjdKNUo2TTMwMEhlTWdWT1syVmkwaE1qVk9YMlZpMGhNalZPWDJWaTBoTWpWT1gyVmkwaE1qVk9YMlZpMGhNalZPWTJVaTBnTWtWT1gyVmkwaE1qVk9YMlZpMGhNalZPWDJWaTBoTWpWT1gyVmkwaE1qVk9YMlVpMGpNalZPVjJhaTAxM0w1TDVLNUs0TDVLNUs0TDVKNks0TDVLNUs0TDNNME8xMDAwMDAxTzAwMDAwMDBPMk8wMDAwMDAwMDRMbWdUVzE="
    },
    "attribute_tokens": [],
    "sample_data_token": "003bf191da774ac3b7c47e44075d9cf9"
    }

    <v1.0-{split}/attribute.json>, sample attribute
    {
    "token": "271f6773e4d2496cbb9942c204c8a4c1",
    "name": "cycle.with_rider",
    "description": "There is a rider on the bicycle or motorcycle."
    }

    <v1.0-{split}/category.json, sample category
    {
    "token": "63a94dfa99bb47529567cd90d3b58384",
    "name": "animal",
    "description": "All animals, e.g. cats, rats, dogs, deer, birds."
    },
    """

    _CATEGORIES = {
        "animal": 0,
        "flat.driveable_surface": 1,
        "human.pedestrian.adult": 2,
        "human.pedestrian.child": 3,
        "human.pedestrian.construction_worker": 4,
        "human.pedestrian.personal_mobility": 5,
        "human.pedestrian.police_officer": 6,
        "human.pedestrian.stroller": 7,
        "human.pedestrian.wheelchair": 8,
        "movable_object.barrier": 9,
        "movable_object.debris": 10,
        "movable_object.pushable_pullable": 11,
        "movable_object.trafficcone": 12,
        "static_object.bicycle_rack": 13,
        "vehicle.bicycle": 14,
        "vehicle.bus.bendy": 15,
        "vehicle.bus.rigid": 16,
        "vehicle.car": 17,
        "vehicle.construction": 18,
        "vehicle.ego": 19,
        "vehicle.emergency.ambulance": 20,
        "vehicle.emergency.police": 21,
        "vehicle.motorcycle": 22,
        "vehicle.trailer": 23,
        "vehicle.truck": 24
        }

    def category_to_cls(self, category: str) -> int:
        return self._CATEGORIES[category]
    
    def filter_by_token(self, data: List[Dict[str, Any]], field: str, match_value: str) -> List[Dict[str, Any]]:
        filtered_list = []
        for item in data:
            if item.get(field) == match_value:
                filtered_list.append(item)
        return filtered_list
    
    def __init__(self, split: Union[Literal["train", "val", "test"]] = "train", **kwargs):
        root_dir = project_root_dir() / "data" / "nuimages"
        img_dir = root_dir / "nuimages-v1.0-all-samples"
        obj_annotations_file = root_dir / "nuimages-v1.0-all-metadata" / f"v1.0-{split}" / "object_ann.json"
        categories_file = root_dir / "nuimages-v1.0-all-metadata" / f"v1.0-{split}" / "category.json"
        sample_data_labels_file = root_dir / "nuimages-v1.0-all-metadata" / f"v1.0-{split}" / "sample_data.json"
        attributes_file = root_dir / "nuimages-v1.0-all-metadata" / f"v1.0-{split}" / "attribute.json"
        
        self.sample_data_labels = json.load(open(sample_data_labels_file))
        self.attribute_labels = json.load(open(attributes_file))
        self.category_labels = json.load(open(categories_file))
        self.obj_annotations = json.load(open(obj_annotations_file))

        def merge_transform(
            image: Tensor,
            labels: List[Dict[str, Any]],
            attributes: Dict[str, Any],
            timestamp: str,
        ) -> Tuple[
            Tensor,
            List[Tuple[ObjectDetectionResultI, Dict[str, Any], str]],
            Dict[str, Any],
            str,
        ]:
            results = []
            obj_attributes = {}

            for obj_label in labels:
                _, height, width = image.shape
                object_category_obj = self.filter_by_token(self.category_labels, "token", obj_label["category_token"])
                object_category_name = ""
                if len(object_category) == 0:
                    object_category = "Unknown"
                else:
                    object_category = object_category_obj[0]["name"]         # Take the first object category
                if len(attributes) > 0:
                    obj_attributes = self.attributes_labels[attributes[0]]   # Take the first attribute token

                results.append(
                    (
                        ObjectDetectionResultI(
                            score=1.0,
                            cls=self.category_to_cls(object_category),
                            label=object_category,
                            bbox=obj_label["bbox"],
                            image_hw=(height, width),
                            bbox_format=BBox_Format.XYXY,
                            attributes=obj_attributes,
                        ),
                        obj_attributes,
                        timestamp,
                    )
                )

            return (image, results, attributes, timestamp)

        super().__init__(
            sample_data_labels_file, img_dir, merge_transform=merge_transform, **kwargs
        )
    
    def __getitem__(self, idx: int) -> Union[Any, Tuple[Tensor, Dict, Dict, str]]:
        img_filename = self.img_labels[idx]["filename"]
        img_token = self.img_labels[idx]["token"]
        timestamp = self.img_labels[idx]["timestamp"]

        try:
            img_path = os.path.join(self.img_dir, img_filename)
            image = decode_image(img_path)
        except Exception as err:
            raise RuntimeError(f"Error: {img_path} failed to read, {err}")
        
        obj_labels = self.filter_by_token(self.obj_annotations, "sample_data_token", img_token)
        obj_attribute_tokens = []
        for obj_label in obj_labels:
            obj_attribute_tokens.append(obj_label["attribute_tokens"])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        if self.merge_transform:
            image, labels, attributes, timestamp = self.merge_transform(
                image, obj_labels, obj_attribute_tokens, timestamp
            )

        return {
            "image": image,
            "labels": labels,
            "attributes": attributes,
            "timestamp": timestamp,
        }

