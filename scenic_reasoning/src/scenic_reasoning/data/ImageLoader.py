import io
import json
import os
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import pandas as pd
from PIL import Image
from scenic_reasoning.interfaces.ObjectDetectionI import (
    BBox_Format,
    ObjectDetectionResultI,
)
from scenic_reasoning.utilities.common import project_root_dir
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
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
        img_path = os.path.join(self.img_dir, self.img_labels[idx]["name"])
        image = decode_image(img_path)

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
        dataset: Union[Literal["all", "mini"]] = "mini",
        split: Union[Literal["train", "val", "test"]] = "train",
        **kwargs,
    ):
        root_dir = project_root_dir() / "data" / "nuimages"

        if dataset == "all":
            img_dir = root_dir / "all"
            metadata_dir = root_dir / "all" / f"v1.0-{split}"
        else:
            img_dir = root_dir / "mini"
            metadata_dir = root_dir / "mini" / "v1.0-mini"

        obj_annotations_file = metadata_dir / "object_ann.json"
        categories_file = metadata_dir / "category.json"
        sample_data_labels_file = metadata_dir / "sample_data.json"
        attributes_file = metadata_dir / "attribute.json"

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
                object_category_obj = self.filter_by_token(
                    self.category_labels, "token", obj_label["category_token"]
                )
                object_category_name = ""
                if len(object_category_obj) == 0:
                    object_category = "Unknown"
                else:
                    object_category = object_category_obj[0][
                        "name"
                    ]  # Take the first object category
                if len(obj_label["attribute_tokens"]) > 0:
                    for attribute_token in obj_label["attribute_tokens"]:
                        attribute_obj = self.filter_by_token(
                            self.attribute_labels, "token", attribute_token
                        )
                        if len(attribute_obj) == 0:
                            obj_attributes[attribute_token] = "Unknown"
                        else:
                            obj_attributes = dict(
                                [
                                    (str(i), attr_obj)
                                    for i, attr_obj in enumerate(attribute_obj)
                                ]
                            )

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
        img_path = os.path.join(self.img_dir, img_filename)
        image = decode_image(img_path)

        obj_labels = self.filter_by_token(
            self.obj_annotations, "sample_data_token", img_token
        )
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


class WaymoDataset(ImageDataset):
    """
    camera_image/{segment_context_name}.parquet
    15 columns
    Index(['key.segment_context_name', 'key.frame_timestamp_micros',
       'key.camera_name', '[CameraImageComponent].image',
       '[CameraImageComponent].pose.transform',
       '[CameraImageComponent].velocity.linear_velocity.x',
       '[CameraImageComponent].velocity.linear_velocity.y',
       '[CameraImageComponent].velocity.linear_velocity.z',
       '[CameraImageComponent].velocity.angular_velocity.x',
       '[CameraImageComponent].velocity.angular_velocity.y',
       '[CameraImageComponent].velocity.angular_velocity.z',
       '[CameraImageComponent].pose_timestamp',
       '[CameraImageComponent].rolling_shutter_params.shutter',
       '[CameraImageComponent].rolling_shutter_params.camera_trigger_time',
       '[CameraImageComponent].rolling_shutter_params.camera_readout_done_time'],
      dtype='object')
    (variable_size_rows, 15)

    camera_box/{segment_context_name}.parquet
    11 columns
    Index(['key.segment_context_name', 'key.frame_timestamp_micros',
       'key.camera_name', 'key.camera_object_id',
       '[CameraBoxComponent].box.center.x',
       '[CameraBoxComponent].box.center.y', '[CameraBoxComponent].box.size.x',
       '[CameraBoxComponent].box.size.y', '[CameraBoxComponent].type',
       '[CameraBoxComponent].difficulty_level.detection',
       '[CameraBoxComponent].difficulty_level.tracking'],
      dtype='object')
    (variable_size_rows, 11)

    """

    _CATEGORIES = {
        "TYPE_UNKNOWN": 0,
        "TYPE_VEHICLE": 1,
        "TYPE_PEDESTRIAN": 2,
        "TYPE_SIGN": 3,
        "TYPE_CYCLIST": 4,
    }

    _CLS_TO_CATEGORIES = {
        "0": "TYPE_UNKNOWN",
        "1": "TYPE VEHICLE",
        "2": "TYPE_PEDESTRIAN",
        "3": "TYPE_SIGN",
        "4": "TYPE_CYCLIST",
    }

    def category_to_cls(self, category: str) -> int:
        return self._CATEGORIES[category]

    def cls_to_category(self, cls: int) -> str:
        return self._CLS_TO_CATEGORIES[str(cls)]

    def convert_to_xyxy(center_x: int, center_y: int, width: int, height: int):
        """Converts bounding box from center-width-height format to XYXY format.
        Returns:
            A tuple (x1, y1, x2, y2) representing the bounding box in XYXY format.
        """
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        return x1, y1, x2, y2

    def __init__(
        self,
        split: Union[Literal["training", "validation", "testing"]] = "training",
        **kwargs,
    ):
        root_dir = project_root_dir() / "data" / "waymo"
        camera_img_dir = root_dir / f"{split}" / "camera_image"
        camera_box_dir = root_dir / f"{split}" / "camera_box"
        camera_image_files = [
            f for f in os.listdir(camera_img_dir) if f.endswith(".parquet")
        ]
        self.img_labels = []
        # Iterate over each file and pair with corresponding camera_box file
        for image_file in camera_image_files:
            # Get the matching camera_box filename for the image
            box_file = image_file.replace("camera_image", "camera_box")
            print(f"Box file: {box_file}")

            # Load the image data (camera image)
            image_df = pd.read_parquet(os.path.join(camera_img_dir, image_file))

            # Load the corresponding bounding box annotations
            box_file_path = os.path.join(camera_box_dir, box_file)
            box_data_df = pd.read_parquet(os.path.join(camera_box_dir, box_file))

            # Merge the image box dataframe
            merged_image_box_df = pd.merge(
                image_df,
                box_data_df,
                on=["key.segment_context_name", "key.frame_timestamp_micros"],
                how="inner",
            )
            merged_image_box_records = merged_image_box_df.to_dict("records")
            self.img_labels.append(merged_image_box_records)

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
                object_category_cls = obj_label["[CameraBoxComponent].type"]
                box_center_x = obj_label["[CameraBoxComponent].box.center.x"]
                box_center_y = obj_label["[CameraBoxComponent].box.center.y"]
                box_width = obj_label["[CameraBoxComponent].box.size.x"]
                box_height = obj_label["[CameraBoxComponent].box.size.y"]

                box_x1, box_y1, box_x2, box_y2 = self.convert_to_xyxy(
                    box_center_x, box_center_y, box_width, box_height
                )

                results.append(
                    (
                        ObjectDetectionResultI(
                            score=1.0,
                            cls=object_category_cls,
                            label=self.cls_to_category(object_category_cls),
                            bbox=[
                                box_x1,
                                box_y1,
                                box_x2,
                                box_y2,
                            ],
                            image_hw=(height, width),
                            bbox_format=BBox_Format.XYXY,
                            attributes=obj_attributes,
                        ),
                        obj_attributes,
                        timestamp,
                    )
                )

            return (image, results, attributes, timestamp)

        # initialize "annotations_file" as the camera_box_dir path
        super().__init__(
            camera_box_dir, camera_img_dir, merge_transform=merge_transform, **kwargs
        )

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Tensor, Dict, Dict, str]]:
        # take the first detected object of image for name and timestamp, they are the same
        segment_context_name = self.img_labels[idx][0]["key.segment_context_name"]
        frame_timestamp_micros = self.img_labels[idx][0]["key.frame_timestamp_micros"]
        img_bytes = self.img_labels[0]["[CameraImageComponent].image"]
        attributes = {}  # May add more information as needed in future
        timestamp = str(frame_timestamp_micros)
        img_path = os.path.join(self.img_dir, segment_context_name)
        image_from_PIL = Image.open(io.BytesIO(img_bytes))
        transform = transforms.ToTensor()
        image = transform(image_from_PIL)
        obj_attribute_tokens = []

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        if self.merge_transform:
            image, labels, attributes, timestamp = self.merge_transform(
                image, self.img_labels, obj_attribute_tokens, timestamp
            )

        return {
            "image": image,
            "labels": labels,
            "attributes": attributes,
            "timestamp": str(frame_timestamp_micros),
        }
