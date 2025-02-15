from pathlib import Path
<<<<<<< HEAD
from torchvision.io import decode_image
from typing import Iterator, List, Tuple
=======
from typing import Any, Dict, Iterator, List, Tuple

>>>>>>> main
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

# def yolo_waymo_transform(image, labels, stride=32):
#     orig_H, orig_W = image.shape[1:]

#     C, H, W = image.shape
#     new_H = (H + stride - 1) // stride * stride
#     new_W = (W + stride - 1) // stride * stride
#     image = image.permute(1, 2, 0).cpu().numpy()
#     resized_image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
#     resized_image = torch.from_numpy(resized_image).permute(2, 0, 1).float()

#     scale_x = new_W / orig_W
#     scale_y = new_H / orig_H
#     for label in labels:
#         x1, y1, x2, y2 = label["bbox"]
#         label["bbox"] = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

#     return resized_image, labels


def _get_bbox(label: Dict[str, Any], box_key: str) -> List[float]:
    """
    Retrieve bounding box coordinates from a label. The box may be stored
    either as a dict (e.g., label['box2d']['x1'..'y2']) or as a list (e.g., label['bbox']).
    """
    box_data = label[box_key]
    if isinstance(box_data, dict):
        # e.g., { 'x1': val, 'y1': val, 'x2': val, 'y2': val }
        return [box_data["x1"], box_data["y1"], box_data["x2"], box_data["y2"]]
    else:
        # e.g., [x1, y1, x2, y2]
        return box_data


def _set_bbox(label: Dict[str, Any], box_key: str, coords: List[float]) -> None:
    """
    Update bounding box coordinates in a label.
    If stored as dict, update dict fields. If stored as list, replace the list.
    """
    box_data = label[box_key]
    x1, y1, x2, y2 = coords
    if isinstance(box_data, dict):
        box_data["x1"] = x1
        box_data["y1"] = y1
        box_data["x2"] = x2
        box_data["y2"] = y2
    else:
        label[box_key] = [x1, y1, x2, y2]


def yolo_transform(
    image: torch.Tensor,
    labels: List[Dict[str, Any]],
    new_shape: Tuple[int, int],
    box_key: str,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    A unified transform function that applies a letterbox transform to the image
    and rescales bounding boxes according to the new shape. The bounding box
    field is determined by 'box_key'.
    """
    # Example: shape_transform is a letterbox transform that expects
    #   updated_labels["img"], updated_labels["instances"].bboxes, etc.
    shape_transform = LetterBox(new_shape=new_shape, scaleup=False)

    # Original image dimensions
    orig_H, orig_W = image.shape[1:]
    ratio = min(new_shape[0] / orig_H, new_shape[1] / orig_W)

    # Prepare data for shape_transform
    image_np = image.permute(1, 2, 0).numpy()
    updated_labels = {
        "img": image_np,
        "cls": np.zeros_like(labels),
        "ratio_pad": ((ratio, ratio), 0, 0),  # ((ratio, ratio), left, top)
        "instances": Instances(
            bboxes=np.array([_get_bbox(label, box_key) for label in labels]),
            # Provide 'segments' to avoid certain ultralytics issues
            segments=np.zeros(shape=[len(labels), int(new_shape[1] * 3 / 4), 2]),
        ),
    }

    # Perform the letterbox transform
    updated_labels = shape_transform(updated_labels)
    # After shape_transform, ratio_pad might include new left, top pad values
    left, top = updated_labels["ratio_pad"][1]

    # Convert back to torch, scale to [0,1] range
    image_out = (
        torch.tensor(updated_labels["img"]).permute(2, 0, 1).to(torch.float32) / scale
    )

    # Update label bounding boxes based on the new ratio and padding
    for label in labels:
        x1, y1, x2, y2 = _get_bbox(label, box_key)
        new_coords = [
            x1 * ratio + left,
            y1 * ratio + top,
            x2 * ratio + left,
            y2 * ratio + top,
        ]
        _set_bbox(label, box_key, new_coords)

    return image_out, labels


def yolo_bdd_transform(
    image: torch.Tensor, labels: List[dict], new_shape: Tuple[int, int]
):
    return yolo_transform(image, labels, new_shape, "box2d", scale=255.0)


def yolo_nuscene_transform(
    image: torch.Tensor, labels: List[dict], new_shape: Tuple[int, int]
):
    return yolo_transform(image, labels, new_shape, "bbox", scale=255.0)


def yolo_waymo_transform(
    image: torch.Tensor, labels: List[dict], new_shape: Tuple[int, int]
):
    return yolo_transform(image, labels, new_shape, "bbox", scale=1.0)
