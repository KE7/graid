import base64
import io
import json
import logging
import os
import pickle
import re
from datetime import datetime, time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools import mask as cocomask
from graid.interfaces.InstanceSegmentationI import (
    InstanceSegmentationResultI,
    Mask_Format,
)
from graid.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionResultI,
)
from graid.utilities.coco import inverse_coco_label
from graid.utilities.common import (
    convert_to_xyxy,
    project_root_dir,
    read_image,
)
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(
        self,
        annotations_file: Optional[str] = None,
        mask_dir: Optional[str] = None,
        img_dir: str = "",
        transform: Union[Callable, None] = None,
        target_transform: Union[Callable, None] = None,
        merge_transform: Union[Callable, None] = None,
        use_extended_annotations: bool = False,
        img_labels: Optional[List[Dict]] = None,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.merge_transform = merge_transform
        self.use_extended_annotations = use_extended_annotations
        self.img_labels = img_labels or []  # either pass it in or default empty
        self.mask_dir = mask_dir
        self.masks = []
        # Load annotations if annotations_file is provided, else keep img_labels empty

        if annotations_file:
            self.img_labels = self.load_annotations(annotations_file)

    def load_annotations(self, annotations_file: str) -> List[Dict]:
        """Load annotations from a JSON file."""
        with open(annotations_file, "r") as file:
            return json.load(file)

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclasses must implement __getitem__")


class Bdd10kDataset(ImageDataset):

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
        0: "unlabeled",
        1: "dynamic",
        2: "ego vehicle",
        3: "ground",
        4: "static",
        5: "parking",
        6: "rail track",
        7: "road",
        8: "sidewalk",
        9: "bridge",
        10: "building",
        11: "fence",
        12: "garage",
        13: "guard rail",
        14: "tunnel",
        15: "wall",
        16: "banner",
        17: "billboard",
        18: "lane divider",
        19: "parking sign",
        20: "pole",
        21: "polegroup",
        22: "street light",
        23: "traffic cone",
        24: "traffic device",
        25: "traffic light",
        26: "traffic sign",
        27: "traffic sign frame",
        28: "terrain",
        29: "vegetation",
        30: "sky",
        31: "person",
        32: "rider",
        33: "bicycle",
        34: "bus",
        35: "car",
        36: "caravan",
        37: "motorcycle",
        38: "trailer",
        39: "train",
        40: "truck",
    }

    def category_to_cls(self, category: str) -> int:
        return self._CATEGORIES[category]

    def category_to_coco_cls(self, category: str) -> int:
        return self._CATEGORIES_TO_COCO.get(category, -1)

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        **kwargs,
    ):

        root_dir = project_root_dir() / "data" / "bdd100k"
        img_dir = root_dir / "images" / "10k" / split
        rle = root_dir / "labels" / "ins_seg" / "rles" / f"ins_seg_{split}.json"

        def merge_transform(image: Tensor, labels, timestamp):
            results = []
            attributes = []

            for instance_id, label in enumerate(labels):
                rle = label["rle"]
                mask = cocomask.decode(rle)
                class_label = label["category"]
                class_id = self.category_to_coco_cls(class_label)
                result = InstanceSegmentationResultI(
                    score=1.0,
                    cls=int(class_id),
                    label=class_label,
                    instance_id=int(instance_id),
                    image_hw=rle["size"],
                    mask=torch.from_numpy(mask).unsqueeze(0),
                )
                results.append(result)
                attributes.append(label["attributes"])

            return image, results, timestamp

        super().__init__(
            img_dir=str(img_dir),
            annotations_file=rle,
            merge_transform=merge_transform,
            **kwargs,
        )

    def __len__(self):
        return len(self.img_labels["frames"])

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Tensor, Dict, Dict, str]]:
        data = self.img_labels["frames"][idx]
        img_path = os.path.join(self.img_dir, data["name"])
        labels = data["labels"]
        timestamp = data["timestamp"]
        
        image = cv2.imread(img_path)
        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        if self.merge_transform:
            image, labels, timestamp = self.merge_transform(image, labels, timestamp)

        return {
            "name": data["name"],
            "path": img_path,
            "image": image,
            "labels": labels,
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
        # TODO: test a COCO model on a trailer. try image 2357
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

    def __repr__(self):
        return f"BDD100K Dataset {self.split} split with {self.__len__()} images"

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        use_original_categories: bool = True,
        use_extended_annotations: bool = True,
        use_time_filtered: bool = True,
        rebuild: bool = False,
        **kwargs,
    ):
        self.split = split
        self.use_time_filtered = use_time_filtered

        root_dir = project_root_dir() / "data" / "bdd100k"
        img_dir = root_dir / "images" / "100k" / split
        annotations_file = (
            root_dir
            / "labels"
            / "det_20"
            / (
                f"det_{split}_filtered.json"
                if use_time_filtered
                else f"det_{split}.json"
            )
        )

        self.img_labels = self.load_annotations(annotations_file)

        def merge_transform(
            image: Tensor,
            labels: List[Dict[str, Any]],
            timestamp: str,
        ) -> Union[
            Tuple[
                Tensor, List[Union[ObjectDetectionResultI, InstanceSegmentationResultI]]
            ],
            Tuple[
                Tensor,
                List[
                    Tuple[
                        Union[ObjectDetectionResultI, InstanceSegmentationResultI],
                        Dict[str, Any],
                        str,
                    ]
                ],
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

                result = ObjectDetectionResultI(
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
                    attributes=[label["attributes"]],
                )

                results.append(result)

            return image, results, timestamp

        # finally, filter out following labels
        #   'other person', 'other vehicle' and 'trail'
        # because they are uncertain objects: https://github.com/bdd100k/bdd100k/blob/master/bdd100k/common/typing.py#L4
        self.img_labels = [
            label
            for label in self.img_labels
            if not any(
                filter(
                    lambda l: l["category"]
                    in ["other person", "other vehicle", "trail", "trailer"],
                    label.get("labels", []),
                )
            )
            and "labels" in label
        ]

        # Define paths for original and filtered data
        orig_dir = project_root_dir() / "data" / f"bdd_{self.split}"

        filtered_dir = (
            project_root_dir() / "data" / f"bdd_{self.split}_time_and_weather_filtered"
        )

        filtered_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(filtered_dir, 0o777)
        except Exception as e:
            logger.warning(f"Failed to set permissions on {filtered_dir}: {e}")

        self.mapping_file = filtered_dir / "mapping.json"

        if rebuild:
            # Create a mapping from filtered indices to original indices
            filtered_to_orig_mapping = {}
            filtered_idx = 0

            print(f"Building mapping for filtered BDD100K dataset...")
            for idx, label in tqdm(
                enumerate(self.img_labels),
                total=len(self.img_labels),
                desc="Filtering BDD100K dataset...",
            ):
                # Check if image meets filtering criteria
                if not self._meets_filtering_criteria(label):
                    continue

                # Store the mapping from filtered index to original index
                filtered_to_orig_mapping[str(filtered_idx)] = idx
                filtered_idx += 1

            # Save the mapping as a JSON file
            print(f"Saving mapping file with {len(filtered_to_orig_mapping)} entries")
            with open(self.mapping_file, "w") as f:
                json.dump(filtered_to_orig_mapping, f)
            os.chmod(self.mapping_file, 0o777)

            # Load the mapping for use in __getitem__ and __len__
            self.filtered_to_orig_mapping = filtered_to_orig_mapping
        elif os.path.exists(self.mapping_file):
            # Load existing mapping if available
            with open(self.mapping_file, "r") as f:
                self.filtered_to_orig_mapping = json.load(f)
        else:
            # No mapping file and not rebuilding, so use empty mapping
            self.filtered_to_orig_mapping = {}

        super().__init__(
            annotations_file=str(annotations_file),
            img_dir=str(img_dir),
            merge_transform=merge_transform,
            use_extended_annotations=use_extended_annotations,
            **kwargs,
        )

    def __len__(self) -> int:
        if hasattr(self, "filtered_to_orig_mapping") and self.filtered_to_orig_mapping:
            return len(self.filtered_to_orig_mapping)
        elif os.path.exists(self.mapping_file):
            # Load mapping if not already loaded
            with open(self.mapping_file, "r") as f:
                self.filtered_to_orig_mapping = json.load(f)
            return len(self.filtered_to_orig_mapping)
        else:
            # Fallback to original dataset size if no mapping exists
            return len(self.img_labels)

    def _meets_filtering_criteria(self, label: Dict[str, Any]) -> bool:
        """
        Check if an image meets the filtering criteria:
        - timeofday must be 'daytime'
        - weather must not be 'foggy', 'snowy', or 'rainy'

        Args:
            label: Image label dictionary containing attributes

        Returns:
            True if the image meets filtering criteria, False otherwise
        """
        if "attributes" not in label:
            return False

        attributes = label["attributes"]

        # Check timeofday - keep only 'daytime'
        if attributes.get("timeofday", "") != "daytime":
            return False

        # Check weather - exclude 'foggy', 'snowy', 'rainy'
        if attributes.get("weather", "") in ["foggy", "snowy", "rainy"]:
            return False

        return True

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Tensor, Dict, Dict, str]]:
        # If we're using filtered dataset and have a mapping
        if self.use_time_filtered and hasattr(self, "filtered_to_orig_mapping"):
            if not self.filtered_to_orig_mapping and os.path.exists(self.mapping_file):
                # Load mapping if not already loaded
                with open(self.mapping_file, "r") as f:
                    self.filtered_to_orig_mapping = json.load(f)

            # Get the original index from our mapping
            if str(idx) not in self.filtered_to_orig_mapping:
                raise IndexError(f"Filtered index {idx} not found in mapping")

            # Use the original index to access the original data
            orig_idx = self.filtered_to_orig_mapping[str(idx)]

            # Get original file path
            orig_path = (
                project_root_dir() / "data" / f"bdd_{self.split}" / f"{orig_idx}.pkl"
            )
        else:
            # Use original dataset if not filtered
            orig_path = project_root_dir() / "data" / f"bdd_{self.split}" / f"{idx}.pkl"

        # Load the data from the original file
        with open(orig_path, "rb") as f:
            data = pickle.load(f)

        # get the image path
        img_path = data["name"]
        # join with the root dir
        img_path = os.path.join(
            project_root_dir(),
            "data",
            "bdd100k",
            "images",
            "100k",
            self.split,
            img_path,
        )

        # get the labels
        labels = data["labels"]

        # get the timestamp
        timestamp = data["timestamp"]

        # load the image
        img = read_image(img_path)

        # apply transforms if they exist
        if self.transform:
            img, labels = self.transform(img, labels)

        # load the labels
        if self.merge_transform is not None:
            return self.merge_transform(img, labels, timestamp)
        else:
            return img, labels, timestamp


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
        "vehicle.truck": 24,
    }

    _CATEGORIES_TO_COCO = {
        "animal": "undefined",  # TODO: change this to include speicfic animals
        "flat.driveable_surface": "undefined",
        "human.pedestrian.adult": "person",
        "human.pedestrian.child": "person",
        "human.pedestrian.construction_worker": "person",
        "human.pedestrian.personal_mobility": "person",
        "human.pedestrian.police_officer": "person",
        "human.pedestrian.stroller": "person",
        "human.pedestrian.wheelchair": "person",
        "movable_object.barrier": "undefined",
        "movable_object.debris": "undefined",
        "movable_object.pushable_pullable": "undefined",
        "movable_object.trafficcone": "undefined",
        "static_object.bicycle_rack": "undefined",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "truck",
        "vehicle.ego": "undefined",
        "vehicle.emergency.ambulance": "undefined",
        "vehicle.emergency.police": "person",
        "vehicle.motorcycle": "motorcycle",
        "vehicle.trailer": "undefined",
        "vehicle.truck": "truck",
    }

    def category_to_cls(self, category: str) -> int:
        return inverse_coco_label[category]

    def category_to_coco(self, category: str):
        return self._CATEGORIES_TO_COCO[category]

    def filter_by_token(
        self, data: List[Dict[str, Any]], field: str, match_value: str
    ) -> List[Dict[str, Any]]:
        filtered_list = []
        for item in data:
            if item.get(field) == match_value:
                filtered_list.append(item)
        return filtered_list

    def __repr__(self):
        return f"NuImages Dataset {self.split} split with {self.__len__()} images"

    def _get_image_label(self, i: int):
        sample = self.nuim.sample[i]
        sample_token = sample["token"]
        key_camera_token = sample["key_camera_token"]
        object_tokens, surface_tokens = self.nuim.list_anns(
            sample_token, verbose=False
        )  # verbose off to avoid excessive print statement
        if not object_tokens:
            return None

        object_data = []
        for object_token in object_tokens:
            obj = self.nuim.get("object_ann", object_token)
            category_token = obj["category_token"]
            attribute_tokens = obj["attribute_tokens"]
            attributes = []
            for attribute_token in attribute_tokens:
                attribute = self.nuim.get("attribute", attribute_token)
                attributes.append(attribute)

            category = self.nuim.get("category", category_token)["name"]
            obj["category"] = category
            obj["attributes"] = attributes
            object_data.append(obj)

        sample_data = self.nuim.get("sample_data", key_camera_token)
        img_filename = sample_data["filename"]
        timestamp = sample_data["timestamp"]

        return {
            "object_data": object_data,
            "img_filename": img_filename,
            "timestamp": timestamp,
        }

    def is_time_in_working_hours(self, filename: str) -> bool:
        match = re.search(r"\d{4}-\d{2}-\d{2}-(\d{2})-(\d{2})-", filename)
        if not match:
            raise ValueError("Time not found in filename.")

        hour = int(match.group(1))
        minute = int(match.group(2))
        t = time(hour, minute)

        return time(8, 0) <= t < time(18, 0)

    def __init__(
        self,
        split: Literal["train", "val", "test", "mini"] = "val",
        size: Literal["mini", "full"] = "full",
        rebuild: bool = False,
        use_time_filtered: bool = True,
        **kwargs,
    ):
        self.size = size
        self.split = split
        self.use_time_filtered = use_time_filtered

        root_dir = project_root_dir() / "data" / "nuimages" / size
        subdir = "v1.0-" + (size if size == "mini" else split)
        img_dir = root_dir
        obj_annotations_file = root_dir / subdir / "object_ann.json"
        categories_file = root_dir / subdir / "category.json"
        sample_data_labels_file = root_dir / subdir / "sample_data.json"
        attributes_file = root_dir / subdir / "attribute.json"

        self.sample_data_labels = json.load(open(sample_data_labels_file))
        self.attribute_labels = json.load(open(attributes_file))
        self.category_labels = json.load(open(categories_file))
        self.obj_annotations = json.load(open(obj_annotations_file))

        if rebuild:
            from nuimages import NuImages

            self.nuim = NuImages(
                dataroot=img_dir,
                version=subdir,
                verbose=False,
                lazy=True,  # verbose off to avoid excessive print statement
            )
            save_path_parent = (
                project_root_dir()
                / "data"
                / (
                    f"nuimages_{self.split}_filtered"
                    if use_time_filtered
                    else f"nuimages_{self.split}"
                )
            )
            save_path_parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(save_path_parent, 0o777)
            except Exception as e:
                logger.warning(f"Failed to set permissions on {save_path_parent}: {e}")

            empty_count = 0
            idx = 0
            for i in tqdm(
                range(len(self.nuim.sample)),
                desc="Processing NuImage dataset...",  # len(self.nuim.sample)
            ):
                # see: https://www.nuscenes.org/tutorials/nuimages_tutorial.html
                sample = self.nuim.sample[i]
                sample_token = sample["token"]
                key_camera_token = sample["key_camera_token"]
                object_tokens, surface_tokens = self.nuim.list_anns(
                    sample_token, verbose=False
                )  # verbose off to avoid excessive print statement
                if object_tokens == []:
                    empty_count += 1
                    continue

                object_data = []
                for object_token in object_tokens:
                    obj = self.nuim.get("object_ann", object_token)
                    category_token = obj["category_token"]
                    attribute_tokens = obj["attribute_tokens"]
                    attributes = []
                    for attribute_token in attribute_tokens:
                        attribute = self.nuim.get("attribute", attribute_token)
                        attributes.append(attribute)

                    category = self.nuim.get("category", category_token)["name"]
                    obj["category"] = category
                    obj["attributes"] = attributes
                    object_data.append(obj)

                sample_data = self.nuim.get("sample_data", key_camera_token)
                img_filename = sample_data["filename"]
                timestamp = sample_data["timestamp"]

                save_path = save_path_parent / f"{idx}.pkl"
                print("creating idx... ", idx)
                # if save_path.exists():
                #     continue
                if self.use_time_filtered and not self.is_time_in_working_hours(
                    img_filename
                ):
                    print("invalid")
                    continue

                with open(save_path, "wb") as f:
                    pickle.dump(
                        {
                            "filename": img_filename,
                            "labels": object_data,
                            "timestamp": timestamp,
                        },
                        f,
                    )
                os.chmod(save_path, 0o777)
                idx += 1

                # TODO: add error catching logic in case of empty token or token mismatch.

            print(
                f"{split} has {empty_count} out of {len(self.nuim.sample)} empty samples."
            )

        def merge_transform(
            image: Tensor, labels: List[Dict[str, Any]], timestamp: str
        ) -> Tuple[
            Tensor,
            List[Tuple[ObjectDetectionResultI, Dict[str, Any], str]],
            List[Dict[str, Any]],
            str,
        ]:
            results = []
            attributes = []

            for obj_label in labels:
                _, height, width = image.shape
                obj_category = obj_label["category"]
                obj_attributes = obj_label["attributes"]
                label = self.category_to_coco(obj_category)
                cls = self.category_to_cls(label)

                results.append(
                    ObjectDetectionResultI(
                        score=1.0,
                        cls=cls,
                        label=label,
                        bbox=obj_label["bbox"],
                        image_hw=(height, width),
                        bbox_format=BBox_Format.XYXY,
                        attributes=obj_attributes,
                    )
                )
                attributes.append(obj_attributes)

            return (image, results, attributes, timestamp)

        super().__init__(
            img_dir=img_dir,
            merge_transform=merge_transform,
            **kwargs,
        )

    def __len__(self) -> int:
        save_path = (
            project_root_dir()
            / "data"
            / (
                f"nuimages_{self.split}_filtered"
                if self.use_time_filtered
                else f"nuimages_{self.split}"
            )
        )
        return len(os.listdir(save_path))

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Tensor, Dict, Dict, str]]:
        # if isinstance(idx, slice):
        #     img_filename = self.img_labels[idx][0]["filename"]
        #     labels = self.img_labels[idx][0]["labels"]
        #     timestamp = self.img_labels[idx][0]["timestamp"]
        # else:
        save_path = (
            project_root_dir()
            / "data"
            / (
                f"nuimages_{self.split}_filtered"
                if self.use_time_filtered
                else f"nuimages_{self.split}"
            )
            / f"{idx}.pkl"
        )
        if not save_path.exists():
            raise FileNotFoundError(f"File not found: {save_path}")
        with open(save_path, "rb") as f:
            data = pickle.load(f)

        img_filename = data["filename"]
        labels = data["labels"]
        timestamp = data["timestamp"]

        img_path = os.path.join(self.img_dir, img_filename)
        image = read_image(img_path)

        if self.transform:
            image, labels = self.transform(image, labels)
        if self.target_transform:
            labels = self.target_transform(labels)
        if self.merge_transform:
            image, labels, attributes, timestamp = self.merge_transform(
                image, labels, timestamp
            )

        return {
            "name": img_filename,
            "path": img_path,
            "image": image,
            "labels": labels,
            "attributes": attributes,
            "timestamp": timestamp,
        }


class NuImagesDataset_seg(ImageDataset):
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
        "vehicle.truck": 24,
    }

    def category_to_cls(self, category: str) -> int:
        return self._CATEGORIES[category]

    def filter_by_token(
        self, data: List[Dict[str, Any]], field: str, match_value: str
    ) -> List[Dict[str, Any]]:
        filtered_list = []
        for item in data:
            if item.get(field) == match_value:
                filtered_list.append(item)
        return filtered_list

    def __init__(
        self,
        split: Literal["train", "val", "test", "mini"] = "val",
        size: Literal["mini", "full"] = "mini",
        **kwargs,
    ):

        from nuimages import NuImages

        root_dir = project_root_dir() / "data" / "nuimages" / size
        img_dir = root_dir
        mask_annotations_file = root_dir / f"v1.0-{split}" / "object_ann.json"
        categories_file = root_dir / f"v1.0-{split}" / "category.json"
        sample_data_labels_file = root_dir / f"v1.0-{split}" / "sample_data.json"
        attributes_file = root_dir / f"v1.0-{split}" / "attribute.json"

        self.nuim = NuImages(
            dataroot=img_dir, version=f"v1.0-{split}", verbose=False, lazy=True
        )

        self.sample_data_labels = json.load(open(sample_data_labels_file))
        self.attribute_labels = json.load(open(attributes_file))
        self.category_labels = json.load(open(categories_file))
        self.mask_annotations = json.load(open(mask_annotations_file))

        img_labels = []
        for i in range(len(self.nuim.sample)):
            # see: https://www.nuscenes.org/tutorials/nuimages_tutorial.html
            sample = self.nuim.sample[i]
            sample_token = sample["token"]
            key_camera_token = sample["key_camera_token"]
            object_tokens, surface_tokens = self.nuim.list_anns(
                sample_token, verbose=False
            )
            if object_tokens == []:
                continue

            object_data = []
            for object_token in object_tokens:
                obj = self.nuim.get("object_ann", object_token)
                category_token = obj["category_token"]
                attribute_tokens = obj["attribute_tokens"]
                attributes = []
                for attribute_token in attribute_tokens:
                    attribute = self.nuim.get("attribute", attribute_token)
                    attributes.append(attribute)

                category = self.nuim.get("category", category_token)["name"]
                obj["category"] = category
                obj["attributes"] = attributes
                object_data.append(obj)

            sample_data = self.nuim.get("sample_data", key_camera_token)
            img_filename = sample_data["filename"]
            timestamp = sample_data["timestamp"]
            img_labels.append(
                {
                    "filename": img_filename,
                    "labels": object_data,
                    "timestamp": timestamp,
                }
            )

        def merge_transform(
            image: Tensor, labels: List[Dict[str, Any]], timestamp: str
        ) -> Tuple[
            Tensor,
            List[Tuple[InstanceSegmentationResultI, Dict[str, Any], str]],
            Dict[str, Any],
            str,
        ]:
            results = []
            attributes = []

            for instance_id, obj_label in enumerate(labels):
                _, height, width = image.shape
                obj_category = obj_label["category"]
                obj_attributes = obj_label["attributes"]
                new_mask = obj_label["mask"].copy()
                new_mask["counts"] = base64.b64decode(new_mask["counts"])
                mask = cocomask.decode(new_mask)

                results.append(
                    InstanceSegmentationResultI(
                        score=1.0,
                        cls=self.category_to_cls(obj_category),
                        label=obj_category,
                        instance_id=instance_id,
                        image_hw=(height, width),
                        mask=torch.from_numpy(mask).unsqueeze(0),
                        mask_format=Mask_Format.BITMASK,
                    )
                )
                attributes.append(obj_attributes)

            return (image, results, attributes, timestamp)

        super().__init__(
            img_labels=img_labels,
            img_dir=img_dir,
            merge_transform=merge_transform,
            **kwargs,
        )

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Tensor, Dict, Dict, str]]:
        img_filename = self.img_labels[idx]["filename"]
        labels = self.img_labels[idx]["labels"]
        timestamp = self.img_labels[idx]["timestamp"]
        img_path = os.path.join(self.img_dir, img_filename)
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        if self.merge_transform:
            image, labels, attributes, timestamp = self.merge_transform(
                image, labels, timestamp
            )

        return {
            "name": img_filename,
            "path": img_path,
            "image": image,
            "labels": labels,
            "attributes": attributes,
            "timestamp": timestamp,
        }


class WaymoDataset(ImageDataset):
    """
    -    camera_image/{segment_context_name}.parquet
    -    15 columns
    -    Index(['key.segment_context_name', 'key.frame_timestamp_micros',
    -       'key.camera_name', '[CameraImageComponent].image',
    -       '[CameraImageComponent].pose.transform',
    -       '[CameraImageComponent].velocity.linear_velocity.x',
    -       '[CameraImageComponent].velocity.linear_velocity.y',
    -       '[CameraImageComponent].velocity.linear_velocity.z',
    -       '[CameraImageComponent].velocity.angular_velocity.x',
    -       '[CameraImageComponent].velocity.angular_velocity.y',
    -       '[CameraImageComponent].velocity.angular_velocity.z',
    -       '[CameraImageComponent].pose_timestamp',
    -       '[CameraImageComponent].rolling_shutter_params.shutter',
    -       '[CameraImageComponent].rolling_shutter_params.camera_trigger_time',
    -       '[CameraImageComponent].rolling_shutter_params.camera_readout_done_time'],
    -      dtype='object')
    -    (variable_size_rows, 15)
    -
    -    camera_box/{segment_context_name}.parquet
    -    11 columns
    -    Index(['key.segment_context_name', 'key.frame_timestamp_micros',
    -       'key.camera_name', 'key.camera_object_id',
    -       '[CameraBoxComponent].box.center.x',
    -       '[CameraBoxComponent].box.center.y', '[CameraBoxComponent].box.size.x',
    -       '[CameraBoxComponent].box.size.y', '[CameraBoxComponent].type',
    -       '[CameraBoxComponent].difficulty_level.detection',
    -       '[CameraBoxComponent].difficulty_level.tracking'],
    -      dtype='object')
    -    (variable_size_rows, 11)
    -
    -"""

    _CATEGORIES = {
        "TYPE_UNKNOWN": 0,
        "TYPE_VEHICLE": 1,
        "TYPE_PEDESTRIAN": 2,
        "TYPE_SIGN": 3,
        "TYPE_CYCLIST": 4,
    }

    _CATEGORIES_R = {v: k for k, v in _CATEGORIES.items()}

    _CLS_TO_COCO_CLS = {
        "TYPE_UNKNOWN": "undefined",
        "TYPE_VEHICLE": "car",
        "TYPE_PEDESTRIAN": "person",
        "TYPE_SIGN": "stop sign",
        "TYPE_CYCLIST": "person",
    }

    _CATEGORIES_TO_COCO = {
        "TYPE_UNKNOWN": -1,
        "TYPE_VEHICLE": 2,
        "TYPE_PEDESTRIAN": 0,
        "TYPE_SIGN": 11,
        "TYPE_CYCLIST": 0,
    }

    def category_to_cls(self, category: str) -> int:
        return self._CATEGORIES_TO_COCO[category]

    def category_to_coco(self, category: str):
        return self._CLS_TO_COCO_CLS[category]

    def cls_to_category(self, cls: int) -> str:
        return self._CATEGORIES_R[cls]

    def is_time_in_working_hours(self, timestamp_micro: str) -> bool:
        timestamp_sec = int(timestamp_micro) / 1e6
        dt = datetime.utcfromtimestamp(timestamp_sec)
        return time(8, 0) <= dt.time() < time(18, 0)

    def __repr__(self):
        return f"Waymo Dataset {self.split} split with {self.__len__()} images"

    def __init__(
        self,
        split: Literal["training", "validation", "testing"] = "training",
        rebuild: bool = False,
        use_time_filtered: bool = True,
        **kwargs,
    ):
        self.split = split
        self.use_time_filtered = use_time_filtered

        root_dir = project_root_dir() / "data" / "waymo"
        self.camera_img_dir = root_dir / f"{split}" / "camera_image"
        self.camera_box_dir = root_dir / f"{split}" / "camera_box"

        # Check if directories exist
        if not os.path.exists(self.camera_img_dir) or not os.path.exists(
            self.camera_box_dir
        ):
            raise FileNotFoundError(
                f"Directories not found: {self.camera_img_dir}, {self.camera_box_dir}"
            )

        self.img_labels = []

        camera_image_files = [
            f for f in os.listdir(self.camera_img_dir) if f.endswith(".parquet")
        ]

        if not camera_image_files:
            raise FileNotFoundError(
                f"No parquet image files found in {self.camera_img_dir}"
            )

        idx = 0

        if rebuild:
            save_path_parent = (
                # project_root_dir() / "data" / f"waymo_{self.split}_interesting"
                project_root_dir() / "data" / f"waymo_{self.split}_filtered"
                if self.use_time_filtered
                else f"waymo_{self.split}"
            )
            save_path_parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(save_path_parent, 0o777)
            except Exception as e:
                logger.warning(f"Failed to set permissions on {save_path_parent}: {e}")

            for image_file in tqdm(
                camera_image_files, desc="Indexing Waymo dataset..."
            ):
                box_file = image_file.replace("camera_image", "camera_box")
                image_path = self.camera_img_dir / image_file
                box_path = self.camera_box_dir / box_file

                # Check if the box file exists
                if not os.path.exists(box_path):
                    logger.warning(f"Box file not found for {image_file}: {box_path}")
                    continue

                # Load the dataframes
                image_df = pd.read_parquet(image_path)
                box_df = pd.read_parquet(box_path)

                unique_images_df = box_df.groupby(
                    [
                        "key.segment_context_name",
                        "key.frame_timestamp_micros",
                        "key.camera_name",
                    ]
                )
                # Merge image and box data
                merged_df = pd.merge(
                    image_df,
                    box_df,
                    on=[
                        "key.segment_context_name",
                        "key.frame_timestamp_micros",
                        "key.camera_name",
                    ],
                    how="inner",
                )

                if merged_df.empty:
                    logger.warning(f"No matches found for {image_file} and {box_file}.")
                    continue

                logger.debug(f"Merged DataFrame for {image_file}: {merged_df.shape}\n")

                # Group dataframes by unique identifiers and process them
                grouped_df = merged_df.groupby(
                    [
                        "key.segment_context_name",
                        "key.frame_timestamp_micros",
                        "key.camera_name",
                    ]
                )

                for group_name, group_data in grouped_df:
                    # Each group has one unique image frame, in which all the detected objects belong to
                    image_data = group_data.iloc[0]
                    img_bytes = image_data["[CameraImageComponent].image"]
                    frame_timestamp_micros = image_data["key.frame_timestamp_micros"]

                    if self.use_time_filtered and not self.is_time_in_working_hours(
                        frame_timestamp_micros
                    ):
                        print("invalid")
                        continue

                    labels = []
                    for _, row in group_data.iterrows():
                        labels.append(
                            {
                                "type": int(row["[CameraBoxComponent].type"]),
                                "bbox": convert_to_xyxy(
                                    row["[CameraBoxComponent].box.center.x"],
                                    row["[CameraBoxComponent].box.center.y"],
                                    row["[CameraBoxComponent].box.size.x"],
                                    row["[CameraBoxComponent].box.size.y"],
                                ),
                            }
                        )

                    # Save the current img label according to the idx as a pickle file
                    save_path = save_path_parent / f"{idx}.pkl"
                    if not save_path.exists():
                        with open(save_path, "wb") as f:
                            os.chmod(save_path, 0o777)
                            pickle.dump(
                                {
                                    "name": group_name[0],
                                    "path": f"{image_path}_{group_name[1]}_{group_name[2]}",
                                    "image": img_bytes,
                                    "labels": labels,
                                    "attributes": {},
                                    "timestamp": str(frame_timestamp_micros),
                                },
                                f,
                            )
                    idx += 1

        def merge_transform(image, labels, attributes, timestamp):
            results = []

            for label in labels:
                cls = label["type"]
                bbox = label["bbox"]
                class_label = self.cls_to_category(cls)
                cls = self.category_to_cls(class_label)
                label = self.category_to_coco(class_label)

                result = ObjectDetectionResultI(
                    score=1.0,
                    cls=cls,
                    label=label,
                    bbox=list(bbox),
                    image_hw=image.shape,
                    attributes=[attributes],
                    bbox_format=BBox_Format.XYXY,
                )
                results.append(result)

            return (image, results, attributes, timestamp)

        # Call the parent class constructor (no annotations_file argument)
        super().__init__(
            annotations_file=None,
            img_dir=str(self.camera_img_dir),
            merge_transform=merge_transform,
            **kwargs,
        )

    def __len__(self) -> int:
        save_path = (
            project_root_dir()
            / "data"
            / (
                f"waymo_{self.split}_interesting"
                if self.use_time_filtered
                else f"waymo_{self.split}"
            )
        )
        return len(os.listdir(save_path))

    def __getitem__(self, idx: int) -> Dict:
        """Retrieve an image and its annotations."""
        if idx >= self.__len__():
            raise IndexError(
                f"Index {idx} out of range for dataset with {len(self.img_labels)} samples."
            )

        save_path = (
            project_root_dir()
            / "data"
            / (
                f"waymo_{self.split}_interesting"
                if self.use_time_filtered
                else f"waymo_{self.split}"
            )
        )
        file_path = os.path.join(save_path, f"{idx}.pkl")
        with open(file_path, "rb") as f:
            img_data = pickle.load(f)

        img_bytes = img_data["image"]
        labels = img_data["labels"]
        timestamp = img_data["timestamp"]
        attributes = img_data["attributes"]

        # Decode the image
        image = transforms.ToTensor()(Image.open(io.BytesIO(img_bytes)))

        # Apply transformations if any
        if self.transform:
            image, labels = self.transform(image, labels)
        if self.target_transform:
            labels = self.target_transform(labels)
        if self.merge_transform:
            image, labels, attributes, timestamp = self.merge_transform(
                image, labels, attributes, timestamp
            )

        return {
            "name": img_data["name"],
            "path": img_data["path"],
            "image": image,
            "labels": labels,
            "attributes": attributes,
            "timestamp": timestamp,
        }


class WaymoDataset_seg(ImageDataset):

    _CATEGORIES = {
        "TYPE_UNDEFINED": 0,
        "TYPE_EGO_VEHICLE": 1,
        "TYPE_CAR": 2,
        "TYPE_TRUCK": 3,
        "TYPE_BUS": 4,
        "TYPE_OTHER_LARGE_VEHICLE": 5,
        "TYPE_BICYCLE": 6,
        "TYPE_MOTORCYCLE": 7,
        "TYPE_TRAILER": 8,
        "TYPE_PEDESTRIAN": 9,
        "TYPE_CYCLIST": 10,
        "TYPE_MOTORCYCLIST": 11,
        "TYPE_BIRD": 12,
        "TYPE_GROUND_ANIMAL": 13,
        "TYPE_CONSTRUCTION_CONE_POLE": 14,
        "TYPE_POLE": 15,
        "TYPE_PEDESTRIAN_OBJECT": 16,
        "TYPE_SIGN": 17,
        "TYPE_TRAFFIC_LIGHT": 18,
        "TYPE_BUILDING": 19,
        "TYPE_ROAD": 20,
        "TYPE_LANE_MARKER": 21,
        "TYPE_ROAD_MARKER": 22,
        "TYPE_SIDEWALK": 23,
        "TYPE_VEGETATION": 24,
        "TYPE_SKY": 25,
        "TYPE_GROUND": 26,
        "TYPE_DYNAMIC": 27,
        "TYPE_STATIC": 28,
    }

    _CLS_TO_CATEGORIES = {str(v): k for k, v in _CATEGORIES.items()}

    # See: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/camera_segmentation.proto

    def category_to_cls(self, category: str) -> int:
        return self._CATEGORIES[category]

    def cls_to_category(self, cls: int) -> str:
        return self._CLS_TO_CATEGORIES[str(cls)]

    def get_semantic_class(self, instance_map, semantic_map, instance_id):
        mask = instance_map == instance_id
        semantic_classes = semantic_map[mask]
        unique_semantic_classes = np.unique(semantic_classes)
        return unique_semantic_classes.tolist()

    def __init__(
        self,
        split: Literal["training", "validation", "testing"] = "training",
        **kwargs,
    ):
        root_dir = project_root_dir() / "data" / "waymo"
        self.camera_img_dir = root_dir / f"{split}" / "camera_image"
        self.camera_box_dir = root_dir / f"{split}" / "camera_box"

        # Check if directories exist
        if not os.path.exists(self.camera_img_dir) or not os.path.exists(
            self.camera_box_dir
        ):
            raise FileNotFoundError(
                f"Directories not found: {self.camera_img_dir}, {self.camera_box_dir}"
            )

        # Initialize img_labels
        self.img_labels = []

        # Get the camera image files in the directory
        camera_image_files = [
            str(self.camera_img_dir / f)
            for f in os.listdir(self.camera_img_dir)
            if f.endswith(".parquet")
        ]

        # Check if image files are found
        if not camera_image_files:
            raise FileNotFoundError(
                f"No parquet image files found in {self.camera_img_dir}"
            )

        merged_dfs = []
        num_empty = 0
        for image_file in camera_image_files:
            seg_file = image_file.replace("camera_image", "camera_segmentation")
            image_path = self.camera_img_dir / image_file
            seg_path = self.camera_box_dir / seg_file

            seg_df = pd.read_parquet(seg_path)
            if seg_df.empty:
                num_empty += 1
                continue

            image_df = pd.read_parquet(image_path)

            merged_df = pd.merge(
                image_df,
                seg_df,
                on=[
                    "key.segment_context_name",
                    "key.frame_timestamp_micros",
                    "key.camera_name",
                ],
                how="inner",
            )

            if merged_df.empty:
                logger.warning(f"No matches found for {image_file} and {seg_file}.")
            else:
                logger.debug(f"Merged DataFrame for {image_file}: {merged_df.shape}\n")
                merged_dfs.append(merged_df)

        logger.debug(f"{num_empty}/{len(camera_image_files)} are empty")

        # Group dataframes by unique identifiers and process them
        for merged_df in merged_dfs:
            grouped_df = merged_df.groupby(
                [
                    "key.segment_context_name",
                    "key.frame_timestamp_micros",
                    "key.camera_name",
                ]
            )

            for group_name, group_data in grouped_df:
                image_data = group_data.iloc[0]
                img_bytes = image_data["[CameraImageComponent].image"]
                frame_timestamp_micros = image_data["key.frame_timestamp_micros"]

                labels = []
                for _, row in group_data.iterrows():

                    labels.append(
                        {
                            "masks": row[
                                "[CameraSegmentationLabelComponent].panoptic_label"
                            ],
                            "global_id": row[
                                "[CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.global_instance_ids"
                            ],
                            "instance_id": row[
                                "[CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.local_instance_ids"
                            ],
                            "divisor": row[
                                "[CameraSegmentationLabelComponent].panoptic_label_divisor"
                            ],
                        }
                    )

                self.img_labels.append(
                    {
                        "name": group_name,
                        "path": image_path,
                        "image": img_bytes,
                        "labels": labels,
                        "attributes": {},  # empty for now, can adjust later to add more Waymo related attributes info
                        "timestamp": str(frame_timestamp_micros),
                    }
                )

        if not self.img_labels:
            raise ValueError(
                f"No valid data found in {self.camera_img_dir} and {self.camera_box_dir}"
            )

        def merge_transform(image, labels, attributes, timestamp):
            masks_bytes = labels[0]["masks"]
            divisor = labels[0]["divisor"]
            instance_id = labels[0]["instance_id"]
            masks = transforms.ToTensor()(Image.open(io.BytesIO(masks_bytes)))
            instance_masks = masks % divisor
            semantic_masks = masks // divisor

            results = []
            for i in instance_id:
                semantic_id = self.get_semantic_class(instance_masks, semantic_masks, i)
                class_id = semantic_id[
                    0
                ]  # see: https://github.com/waymo-research/waymo-open-dataset/issues/570 and page 6 of the original waymo paper: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890052.pdf
                instance_mask = instance_masks == i
                result = InstanceSegmentationResultI(
                    score=1.0,
                    cls=int(class_id),
                    label=self.cls_to_category(class_id),
                    instance_id=i,
                    image_hw=image.shape,
                    mask=instance_mask,
                )
                results.append(result)

            return image, results, attributes, timestamp

        # Call the parent class constructor (no annotations_file argument)
        super().__init__(
            annotations_file=None,
            img_dir=str(self.camera_img_dir),
            img_labels=self.img_labels,
            merge_transform=merge_transform,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Dict:
        """Retrieve an image and its annotations."""
        if idx >= len(self.img_labels):
            raise IndexError(
                f"Index {idx} out of range for dataset with {len(self.img_labels)} samples."
            )

        img_data = self.img_labels[idx]
        img_bytes = img_data["image"]
        labels = img_data["labels"]
        timestamp = img_data["timestamp"]
        attributes = img_data["attributes"]
        # Decode the image
        image = transforms.ToTensor()(Image.open(io.BytesIO(img_bytes)))

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        if self.merge_transform:
            image, labels, attributes, timestamp = self.merge_transform(
                image, labels, attributes, timestamp
            )

        return {
            "name": img_data["name"],
            "path": img_data["path"],
            "image": image,
            "labels": labels,
            "attributes": attributes,
            "timestamp": timestamp,
        }
