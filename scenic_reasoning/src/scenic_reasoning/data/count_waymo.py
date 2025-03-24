import base64
import io
import json
import logging
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from scenic_reasoning.utilities.common import convert_to_xyxy, project_root_dir
from tqdm import tqdm

logger = logging.getLogger(__name__)


def metric(data_per_scene):
    """
    Computes a score that evenly weights the max number of bounding boxes
    and the largest area of a bounding box.

    :param bboxes: List of bounding boxes, each represented as (x, y, w, h)
    :return: Weighted score
    """

    N_max = 0
    A_max = 0

    for image in data_per_scene:
        N_max = max(N_max, image["N"])
        A_max = max(A_max, image["A"])

    best_score = 0
    best_idx = 0

    for i, image in enumerate(data_per_scene):

        N = image["N"]
        A = image["A"]
        N_score = min(N / N_max, 1)
        A_score = min(A / A_max, 1)
        curr_score = 0.5 * (N_score + A_score)
        if curr_score > best_score:
            best_score = curr_score
            best_idx = i

    return best_score, best_idx


def choose_best(camera_image_files, split):

    if not camera_image_files:
        raise FileNotFoundError(f"No parquet image files found in {camera_img_dir}")

    data = {}
    for i, image_file in enumerate(tqdm(camera_image_files, desc="Processing images")):
        box_file = image_file.replace("camera_image", "camera_box")
        image_path = camera_img_dir / image_file
        box_path = camera_box_dir / box_file

        image_df = pd.read_parquet(image_path)
        box_df = pd.read_parquet(box_path)
        merged_df = pd.merge(
            image_df,
            box_df,
            on=[
                "key.segment_context_name",
                "key.frame_timestamp_micros",
                "key.camera_name",
            ],
            how="inner",
        )

        if merged_df.empty:
            logger.warning(f"No matches found for {image_file} and {box_file}.")
            continue

        grouped_df = merged_df.groupby(
            [
                "key.segment_context_name",
                "key.frame_timestamp_micros",
                "key.camera_name",
            ]
        )

        data_per_scene = []
        for group_name, group_data in grouped_df:
            if group_name[2] != 1:  # Only consider camera 1 (front)
                continue
            bboxes = []
            for _, row in group_data.iterrows():
                bbox = convert_to_xyxy(
                    row["[CameraBoxComponent].box.center.x"],
                    row["[CameraBoxComponent].box.center.y"],
                    row["[CameraBoxComponent].box.size.x"],
                    row["[CameraBoxComponent].box.size.y"],
                )

                bboxes.append(bbox)

            image_data = group_data.iloc[0]
            img_bytes = image_data["[CameraImageComponent].image"]
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]
            mean_area = sum(areas) / len(areas)
            frame_timestamp_micros = group_name[1]
            camera_name = group_name[2]

            data_per_scene.append(
                {
                    "key.frame_timestamp_micros": frame_timestamp_micros,
                    "key.camera_name": camera_name,
                    "A": mean_area,
                    "N": len(bboxes),
                    "image": img_bytes,
                    "bboxes": bboxes,
                }
            )

        best_score, idx = metric(data_per_scene)
        best_time, best_camera, image, bboxes = (
            data_per_scene[idx]["key.frame_timestamp_micros"],
            data_per_scene[idx]["key.camera_name"],
            data_per_scene[idx]["image"],
            data_per_scene[idx]["bboxes"],
        )

        print(
            f"Best score: {best_score}, Best time: {best_time}, Best camera: {best_camera}"
        )
        data[image_file] = {
            "key.frame_timestamp_micros": int(best_time),
            "score": best_score,
            "image": base64.b64encode(image).decode("utf-8"),
            "bboxes": bboxes,
            "index": i + idx
        }

    output_file = f"{split}_best_frames.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Data saved to {output_file}")

    return data


if __name__ == "__main__":

    split = "training"
    # root_dir = Path("/work/ke-public/graid_data/waymo")
    root_dir = project_root_dir() / "data" / "waymo"
    camera_img_dir = root_dir / f"{split}" / "camera_image"
    camera_box_dir = root_dir / f"{split}" / "camera_box"

    if not os.path.exists(camera_img_dir) or not os.path.exists(camera_box_dir):
        raise FileNotFoundError(
            f"Directories not found: {camera_img_dir}, {camera_box_dir}"
        )

    camera_image_files = [
        f for f in os.listdir(camera_img_dir) if f.endswith(".parquet")
    ]

    input_file = f"{split}_best_frames.json"

    if os.path.exists(input_file):
        with open(input_file, "r") as f:
            data = json.load(f)
    else:
        data = choose_best(camera_image_files, split)

    import matplotlib.pyplot as plt

    for image_file, image_data in data.items():
        img_bytes = base64.b64decode(image_data["image"])
        img = Image.open(io.BytesIO(img_bytes))
        plt.imshow(img)
        plt.title(f"Score: {image_data['score']}")
        plt.show()


# add bounding boxes to the image
