from pathlib import Path
from typing import Iterator, List

import cv2
import torch
from PIL import Image
from ultralytics.data.augment import LetterBox


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


def yolo_waymo_transform(image, stride=32):
    C, H, W = image.shape
    new_H = (H + stride - 1) // stride * stride
    new_W = (W + stride - 1) // stride * stride
    image = image.permute(1, 2, 0).cpu().numpy()
    resized_image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    resized_image = torch.from_numpy(resized_image).permute(2, 0, 1).float()
    return resized_image


def yolo_bdd_transform(image: torch.Tensor):
    shape_transform = LetterBox(new_shape=(768, 1280))
    image_np = image.permute(1, 2, 0).numpy()
    # 2) resize to 768x1280
    image_np = shape_transform(image=image_np)
    # 3) convert back to tensor
    image = torch.tensor(image_np).permute(2, 0, 1)
    # 4) normalize to 0-1
    image = image.to(torch.float32) / 255.0

    return image


def yolo_nuscene_transform(image):
    return yolo_bdd_transform(image)
