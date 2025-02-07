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
        x1, y1, x2, y2 = label['bbox']
        label['bbox'] = (
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
        )


    return resized_image, labels


def yolo_bdd_transform(image: torch.Tensor, labels):
    orig_H, orig_W = image.shape[1:]
    shape_transform = LetterBox(new_shape=(768, 1280))
    image_np = image.permute(1, 2, 0).numpy()
    image_np = shape_transform(image=image_np)
    image = torch.tensor(image_np).permute(2, 0, 1)
    image = image.to(torch.float32) / 255.0
    
    new_H, new_W = 768, 1280
    scale_x = new_W / orig_W
    scale_y = new_H / orig_H
    pad_x = (orig_W - new_W) / 2
    pad_y = (orig_H - new_H) / 2

    for label in labels:
        bbox = label['box2d']
        label['box2d'] = {
            'x1': bbox['x1'] * scale_x,
            'y1': bbox['y1'] * scale_y,
            'x2': bbox['x2'] * scale_x,
            'y2': bbox['y2'] * scale_y,
        }
    
    return image, labels


def yolo_nuscene_transform(image, labels):
    orig_H, orig_W = image.shape[1:]

    shape_transform = LetterBox(new_shape=(768, 1280))
    image_np = image.permute(1, 2, 0).numpy()
    image_np = shape_transform(image=image_np)
    image = torch.tensor(image_np).permute(2, 0, 1)
    image = image.to(torch.float32) / 255.0
    
    new_H, new_W = 768, 1280
    scale_x = new_W / orig_W
    scale_y = new_H / orig_H
    pad_x = (orig_W - new_W) / 2
    pad_y = (orig_H - new_H) / 2

    for label in labels:
        x1, y1, x2, y2 = label['bbox']
        label['bbox'] = [
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
            ]
        
    return image, labels
