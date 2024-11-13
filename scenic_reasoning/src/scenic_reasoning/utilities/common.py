from pathlib import Path
from typing import Iterator, List

import cv2
import torch
from PIL import Image


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
