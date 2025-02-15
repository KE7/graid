from pathlib import Path
from torchvision.io import decode_image
from typing import Iterator, List, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics.data.augment import LetterBox
from ultralytics.utils.instance import Instances


def get_default_device() -> torch.device:
    """Get the default Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def project_root_dir() -> Path:
    current_dir = Path(__file__).parent.parent.parent.parent.parent
    return current_dir


def open_video(video_path: str, batch_size: int = 1) -> Iterator[List[Image.Image]]:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    while cap.isOpened():
        frames = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)

        if not frames:
            break
        yield frames

    cap.release()


def convert_to_xyxy(center_x: int, center_y: int, width: int, height: int):
    """Converts bounding box from center-width-height format to XYXY format."""
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    return x1, y1, x2, y2

def read_image(img_path):
    try:
        image = decode_image(img_path)
    except Exception as e:
        print(e)
        print("switching to cv2 ...")
        image = cv2.imread(img_path)
        image = torch.from_numpy(image).permute(2, 0, 1)
    return image

def yolo_waymo_transform(image, labels, stride=32):
    orig_H, orig_W = image.shape[1:]

    C, H, W = image.shape
    new_H = (H + stride - 1) // stride * stride
    new_W = (W + stride - 1) // stride * stride
    image = image.permute(1, 2, 0).cpu().numpy()
    resized_image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    resized_image = torch.from_numpy(resized_image).permute(2, 0, 1).float()

    scale_x = new_W / orig_W
    scale_y = new_H / orig_H
    for label in labels:
        x1, y1, x2, y2 = label["bbox"]
        label["bbox"] = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

    return resized_image, labels


def yolo_transform(
    image: torch.Tensor, labels: List[dict], new_shape: Tuple[int, int], bbox_key: str
):
    orig_H, orig_W = image.shape[1:]

    shape_transform = LetterBox(new_shape=new_shape)
    image_np = image.permute(1, 2, 0).numpy()

    updated_labels = dict()
    updated_labels["img"] = image_np
    updated_labels["cls"] = np.zeros_like(labels)
    ratio = min(new_shape[0] / orig_H, new_shape[1] / orig_W)
    updated_labels["ratio_pad"] = ((ratio, ratio), 0, 0)

    updated_labels["instances"] = Instances(
        bboxes=np.array(
            [
                (
                    label[bbox_key]
                    if bbox_key in label
                    else [
                        label["box2d"]["x1"],
                        label["box2d"]["y1"],
                        label["box2d"]["x2"],
                        label["box2d"]["y2"],
                    ]
                )
                for label in labels
            ]
        ),
        # providing segments to avoid ultralytics bug in instance.py:258
        segments=np.zeros(shape=[len(labels), int(new_shape[1] * 3 / 4), 2]),
    )

    updated_labels = shape_transform(updated_labels)
    left, top = updated_labels["ratio_pad"][1]
    image = torch.tensor(updated_labels["img"]).permute(2, 0, 1)
    image = image.to(torch.float32) / 255.0

    for label in labels:
        x1, y1, x2, y2 = (
            label[bbox_key]
            if bbox_key in label
            else [
                label["box2d"]["x1"],
                label["box2d"]["y1"],
                label["box2d"]["x2"],
                label["box2d"]["y2"],
            ]
        )
        label[bbox_key] = [
            x1 * ratio + left,
            y1 * ratio + top,
            x2 * ratio + left,
            y2 * ratio + top,
        ]

    return image, labels


def yolo_bdd_transform(
    image: torch.Tensor, labels: List[dict], new_shape: Tuple[int, int]
):
    return yolo_transform(image, labels, new_shape, "box2d")


def yolo_nuscene_transform(
    image: torch.Tensor, labels: List[dict], new_shape: Tuple[int, int]
):
    return yolo_transform(image, labels, new_shape, "bbox")

