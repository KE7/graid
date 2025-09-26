"""
Waymo best-frame generation (metric-based) – end-to-end steps we use

Overview
  This script selects ONE "best" front-camera (camera 1) frame per Waymo segment
  (one segment ≈ one camera_image parquet file). The score favors frames with
  more/larger bounding boxes: metric = 0.5 * (N/Nmax + A/Amax).

What we run in practice (training + validation)
  1) Build base PKL caches from raw Waymo (all hours; no time filter):
       python - <<'PY'
       from graid.data.ImageLoader import WaymoDataset
       WaymoDataset(split='training',   rebuild=True, use_time_filtered=False)
       WaymoDataset(split='validation', rebuild=True, use_time_filtered=False)
       PY

  2) Generate best-frame JSONs (one entry per scene/parquet):
       # writes training_best_frames.json and validation_best_frames.json
       python graid/src/graid/data/count_waymo.py --split both

  3) Copy those exact frames into data/waymo_{split}_interesting/ (identity-based):
       # selection matches by (segment_context_name, key.frame_timestamp_micros, camera=1)
       python graid/src/graid/data/select_waymo.py --split training
       python graid/src/graid/data/select_waymo.py --split validation

  4) Verify counts match number of scenes:
       # training: 798   validation: 202 (Waymo v1 split sizes)
       ls data/waymo_training_interesting/*.pkl   | wc -l
       ls data/waymo_validation_interesting/*.pkl | wc -l

Notes and pitfalls
  - Do NOT select by numeric index alone. If caches are rebuilt/filtered, PKL
    numbering changes. Our selector now matches by identity (segment+timestamp+camera)
    so counts stay in sync with the JSON.
  - Timestamps are UTC. Local “night” may correspond to hours 05–09 UTC in US
    timezones. To find true night scenes, prefer a luminance/solar-elevation
    heuristic rather than raw UTC hour.
"""

import base64
import argparse
import io
import json
import logging
import os

import pandas as pd
from graid.utilities.common import convert_to_xyxy, project_root_dir
from PIL import Image
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
            "index": i + idx,
        }

    output_file = f"{split}_best_frames.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Data saved to {output_file}")

    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute best Waymo frames per file and write <split>_best_frames.json"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["training", "validation", "both"],
        default="training",
        help="Waymo split to process (or 'both')",
    )
    args = parser.parse_args()
    splits = ["training", "validation"] if args.split == "both" else [args.split]

    for split in splits:
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
            plt.title(f"{split}: Score {image_data['score']}")
            plt.show()


# add bounding boxes to the image
