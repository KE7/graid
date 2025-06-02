from PIL import Image
from scenic_reasoning.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from scenic_reasoning.utilities.common import (
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)

waymo = WaymoDataset(
    split="val",
    transform=yolo_waymo_transform,
)
