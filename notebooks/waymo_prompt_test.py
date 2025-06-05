from PIL import Image
from graid.data.ImageLoader import (
    Bdd100kDataset,
    NuImagesDataset,
    WaymoDataset,
)
from graid.utilities.common import (
    yolo_bdd_transform,
    yolo_nuscene_transform,
    yolo_waymo_transform,
)

waymo = WaymoDataset(
    split="val",
    transform=yolo_waymo_transform,
)
