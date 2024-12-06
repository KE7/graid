import io
import json
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

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
        annotations_file: Optional[str] = None,
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
        self.img_labels = img_labels or []      # either pass it in or default empty
        # Load annotations if annotations_file is provided, else keep img_labels empty
        if annotations_file:
            self.img_labels = self.load_annotations(annotations_file)

    def load_annotations(self, annotations_file: str) -> List[Dict]:
        """Load annotations from a JSON file."""
        with open(annotations_file, 'r') as file:
            return json.load(file)
    
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
        self, split: Union[Literal["train", "val", "test"]] = "train", **kwargs
    ):
        root_dir = project_root_dir() / "data" / "nuimages"
        img_dir = root_dir / "nuimages-v1.0-all-samples"
        obj_annotations_file = (
            root_dir
            / "nuimages-v1.0-all-metadata"
            / f"v1.0-{split}"
            / "object_ann.json"
        )
        categories_file = (
            root_dir / "nuimages-v1.0-all-metadata" / f"v1.0-{split}" / "category.json"
        )
        sample_data_labels_file = (
            root_dir
            / "nuimages-v1.0-all-metadata"
            / f"v1.0-{split}"
            / "sample_data.json"
        )
        attributes_file = (
            root_dir / "nuimages-v1.0-all-metadata" / f"v1.0-{split}" / "attribute.json"
        )

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
                if len(object_category) == 0:
                    object_category = "Unknown"
                else:
                    object_category = object_category_obj[0][
                        "name"
                    ]  # Take the first object category
                if len(attributes) > 0:
                    obj_attributes = self.attributes_labels[
                        attributes[0]
                    ]  # Take the first attribute token

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
        print(f"__getitem__ entered, {len(self.img_labels)}")
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
-    """

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
    
    @staticmethod
    def convert_to_xyxy(center_x: int, center_y: int, width: int, height: int):
        """Converts bounding box from center-width-height format to XYXY format."""
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        return x1, y1, x2, y2
    
    def __init__(self, split: Union[Literal["training", "validation", "testing"]] = "training", **kwargs):
        root_dir = project_root_dir() / "data" / "waymo"
        self.camera_img_dir = root_dir / f"{split}" / "camera_image"
        self.camera_box_dir = root_dir / f"{split}" / "camera_box"

        # Check if directories exist
        if not os.path.exists(self.camera_img_dir) or not os.path.exists(self.camera_box_dir):
            raise FileNotFoundError(f"Directories not found: {self.camera_img_dir}, {self.camera_box_dir}")

        # Initialize img_labels
        self.img_labels = []

        # Get the camera image files in the directory
        camera_image_files = [
            f for f in os.listdir(self.camera_img_dir) if f.endswith(".parquet")
        ]

        # Check if image files are found
        if not camera_image_files:
            raise FileNotFoundError(f"No parquet image files found in {self.camera_img_dir}")
        
        merged_dfs = []
        for image_file in camera_image_files:
            box_file = image_file.replace("camera_image", "camera_box")
            image_path = self.camera_img_dir / image_file
            box_path = self.camera_box_dir / box_file

            # Check if the box file exists
            if not os.path.exists(box_path):
                print(f"Box file not found for {image_file}: {box_path}")
                continue

            # Load the dataframes
            image_df = pd.read_parquet(image_path)
            box_df = pd.read_parquet(box_path)

            unique_images_df = box_df.groupby(['key.segment_context_name', 'key.frame_timestamp_micros', 'key.camera_name'])
            # Count the number of unique groups
            print("Number of unique identifiers:", unique_images_df.ngroups) 

            # Debug: Print DataFrame shapes
            print(f"Image DataFrame: {image_df.shape}, Box DataFrame: {box_df.shape}")
            print(f"Image Columns: {image_df.columns}, Box Columns: {box_df.columns}")
            # Merge image and box data
            merged_df = pd.merge(
                image_df,
                box_df,
                on=["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name"],
                how="inner",
            )

            if merged_df.empty:
                print(f"No matches found for {image_file} and {box_file}.")
            else:
                print(f"Merged DataFrame for {image_file}: {merged_df.shape}\n")
                merged_dfs.append(merged_df)

        # Group dataframes by unique identifiers and process them
        for merged_df in merged_dfs:
            grouped_df = merged_df.groupby(['key.segment_context_name', 'key.frame_timestamp_micros', 'key.camera_name'])

            for group_name, group_data in grouped_df:
                # Each group has one unique image frame, in which all the detected objects belong to
                image_data = group_data.iloc[0]
                img_bytes = image_data["[CameraImageComponent].image"]
                frame_timestamp_micros = image_data["key.frame_timestamp_micros"]

                labels = []
                for _, row in group_data.iterrows():
                    labels.append({
                        "type": row["[CameraBoxComponent].type"],
                        "bbox": self.convert_to_xyxy(
                            row["[CameraBoxComponent].box.center.x"],
                            row["[CameraBoxComponent].box.center.y"],
                            row["[CameraBoxComponent].box.size.x"],
                            row["[CameraBoxComponent].box.size.y"],
                        ),
                    })

                self.img_labels.append({
                    "name": group_name,
                    "image": img_bytes,
                    "labels": labels,
                    "attributes": {},  # empty for now, can adjust later to add more Waymo related attributes info
                    "timestamp": str(frame_timestamp_micros)
                })

        if not self.img_labels:
            raise ValueError(f"No valid data found in {self.camera_img_dir} and {self.camera_box_dir}")
        
        print("Size of img_labels:", len(self.img_labels))
        print(type(self.img_labels))
        # Call the parent class constructor (no annotations_file argument)
        super().__init__(annotations_file=None, img_dir=str(self.camera_img_dir), img_labels=self.img_labels, **kwargs)

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Dict:
        """Retrieve an image and its annotations."""
        if idx >= len(self.img_labels):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.img_labels)} samples.")

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
            if self.use_extended_annotations:
                image, labels, attributes, timestamp = self.merge_transform(
                    image, labels, attributes, timestamp
                )
            else:
                image, labels = self.merge_transform(image, labels, attributes, timestamp)

        return {
            "image": image,
            "labels": labels,
            "attributes": attributes,
            "timestamp": timestamp,
        }