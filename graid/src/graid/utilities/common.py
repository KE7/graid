import io
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from torchvision.io import decode_image
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
    return yolo_transform(image, labels, new_shape, "box2d", scale=1.0)


def yolo_nuscene_transform(
    image: torch.Tensor, labels: List[dict], new_shape: Tuple[int, int]
):
    return yolo_transform(image, labels, new_shape, "bbox", scale=1.0)


def yolo_waymo_transform(
    image: torch.Tensor, labels: List[dict], new_shape: Tuple[int, int]
):
    return yolo_transform(image, labels, new_shape, "bbox", scale=(1.0 / 255.0))


def _decode_to_bool(
    mask: Union[dict, List, Tuple, np.ndarray, torch.Tensor]
) -> np.ndarray:
    """Decode various mask representations to a boolean numpy array."""
    if isinstance(mask, dict):
        # COCO RLE dict
        decoded = mask_util.decode(mask)
        return decoded.astype(bool)
    elif isinstance(mask, (list, tuple)):
        # RLE in list format (as returned by some APIs)
        rle = (
            mask_util.frPyObjects(mask, *mask[0]["size"])
            if isinstance(mask[0], dict)
            else mask[0]
        )
        decoded = mask_util.decode(rle)
        return decoded.astype(bool)
    elif isinstance(mask, (bytes, bytearray)):
        # Assume PNG-encoded binary mask (Waymo dataset). Decode with PIL
        img = Image.open(io.BytesIO(mask)).convert("L")
        return np.array(img) > 0
    elif isinstance(mask, torch.Tensor):
        return mask.squeeze().cpu().numpy().astype(bool)
    elif isinstance(mask, np.ndarray):
        return mask.astype(bool)
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")


def _encode_from_bool(boolean_mask: np.ndarray, original_type_example):
    """Encode boolean numpy mask back to the original type (RLE dict or tensor)."""
    if isinstance(original_type_example, dict):
        rle = mask_util.encode(np.asfortranarray(boolean_mask.astype(np.uint8)))
        # pycocotools returns bytes for counts; convert to str for JSON friendliness
        rle["counts"] = (
            rle["counts"].decode("ascii")
            if isinstance(rle["counts"], bytes)
            else rle["counts"]
        )
        rle["size"] = list(boolean_mask.shape)
        return rle
    elif isinstance(original_type_example, torch.Tensor):
        return torch.from_numpy(boolean_mask).unsqueeze(0)
    elif isinstance(original_type_example, (bytes, bytearray)):
        # Encode back to PNG bytes so downstream Waymo logic still works
        from PIL import Image

        img = Image.fromarray((boolean_mask.astype(np.uint8)) * 255)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    elif isinstance(original_type_example, np.ndarray):
        return boolean_mask
    else:
        # Fallback to numpy array
        return boolean_mask


def _resize_and_letterbox_mask(
    mask_bool: np.ndarray, new_shape: Tuple[int, int]
) -> np.ndarray:
    """Resize a boolean mask using letterbox logic (maintaining aspect ratio with padding)."""
    orig_H, orig_W = mask_bool.shape
    ratio = min(new_shape[0] / orig_H, new_shape[1] / orig_W)
    new_w, new_h = int(round(orig_W * ratio)), int(round(orig_H * ratio))

    # Resize mask
    resized = cv2.resize(
        mask_bool.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # Compute padding
    pad_w, pad_h = new_shape[1] - new_w, new_shape[0] - new_h
    left, top = pad_w // 2, pad_h // 2

    # Pad to desired shape
    padded_mask = np.zeros(new_shape, dtype=bool)
    padded_mask[top : top + new_h, left : left + new_w] = resized
    return padded_mask


def yolo_segmentation_transform(
    image: torch.Tensor,
    labels: List[Dict[str, Any]],
    new_shape: Tuple[int, int],
    box_key: str,
    mask_key: str = "mask",
    scale: float = 1.0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Apply letter-box style resizing to both image and masks.

    1. If the labels contain a bounding-box field (``box_key`` exists in at least one
       label), we delegate to the generic ``yolo_transform`` to keep bbox logic.
    2. Otherwise (e.g. pure segmentation datasets without bboxes) we perform a
       lightweight letter-box operation that only rescales the image and masks.
    """

    can_use_bbox = len(labels) > 0 and all(box_key in l for l in labels)

    if can_use_bbox:
        # Use full bbox-aware pipeline
        image_out, labels = yolo_transform(image, labels, new_shape, box_key, scale)
    else:
        # Manual letter-box without bounding boxes ----------------------------------
        orig_C, orig_H, orig_W = image.shape
        ratio = min(new_shape[0] / orig_H, new_shape[1] / orig_W)
        new_w, new_h = int(round(orig_W * ratio)), int(round(orig_H * ratio))

        # Resize image
        image_np = image.permute(1, 2, 0).cpu().numpy()
        resized_img = cv2.resize(
            image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        # Pad image to target shape
        pad_w, pad_h = new_shape[1] - new_w, new_shape[0] - new_h
        left, top = pad_w // 2, pad_h // 2
        # 114 is Ultralytics' default pad value
        padded_img = np.full(
            (new_shape[0], new_shape[1], orig_C), 114, dtype=resized_img.dtype
        )
        padded_img[top : top + new_h, left : left + new_w] = resized_img

        image_out = (
            torch.from_numpy(padded_img).permute(2, 0, 1).to(torch.float32) / scale
        )

    # -----------------------------------------------------------------------------
    # Resize & pad masks (works for both branches above)
    for label in labels:
        if mask_key in label and label[mask_key] is not None:
            original_mask_repr = label[mask_key]
            mask_bool = _decode_to_bool(original_mask_repr)
            mask_resized = _resize_and_letterbox_mask(mask_bool, new_shape)
            label[mask_key] = _encode_from_bool(mask_resized, original_mask_repr)

    return image_out, labels


# Convenience wrappers for specific datasets -----------------------------------------------------------


def yolo_bdd_seg_transform(
    image: torch.Tensor, labels: List[dict], new_shape: Tuple[int, int]
):
    # BDD segmentation masks stored under 'rle'
    return yolo_segmentation_transform(
        image, labels, new_shape, box_key="box2d", mask_key="rle", scale=1.0
    )


def yolo_nuscene_seg_transform(
    image: torch.Tensor, labels: List[dict], new_shape: Tuple[int, int]
):
    return yolo_segmentation_transform(
        image, labels, new_shape, box_key="bbox", mask_key="mask", scale=1.0
    )


def yolo_waymo_seg_transform(
    image: torch.Tensor, labels: List[dict], new_shape: Tuple[int, int]
):
    return yolo_segmentation_transform(
        image, labels, new_shape, box_key="bbox", mask_key="masks", scale=(1.0 / 255.0)
    )


def persistent_cache(filepath: str):
    def decorator(func):
        cache = {}
        if Path(filepath).exists():
            with open(filepath, "rb") as f:
                try:
                    cache = pickle.load(f)
                except Exception as e:
                    cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            if key not in cache:
                cache[key] = func(*args, **kwargs)
                with open(filepath, "wb") as f:
                    pickle.dump(cache, f)
            return cache[key]

        return wrapper

    return decorator


def convert_image_to_numpy(
    image: Union[str, np.ndarray, torch.Tensor, Image.Image]
) -> np.ndarray:
    """Convert various image formats to numpy array."""
    if isinstance(image, str):
        # File path
        pil_image = Image.open(image)
        return np.array(pil_image)
    elif isinstance(image, np.ndarray):
        return image
    elif isinstance(image, torch.Tensor):
        # Convert tensor to numpy
        if image.dim() == 3 and image.shape[0] in [1, 3]:
            # CHW format
            return image.permute(1, 2, 0).cpu().numpy()
        else:
            # HWC format or other
            return image.cpu().numpy()
    elif isinstance(image, Image.Image):
        return np.array(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def convert_batch_to_numpy(
    batch: List[Union[str, np.ndarray, torch.Tensor, Image.Image]]
) -> List[np.ndarray]:
    """Convert a batch of images to numpy arrays."""
    return [convert_image_to_numpy(image) for image in batch]
