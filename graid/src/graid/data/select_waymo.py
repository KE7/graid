"""
Waymo best-frame selection – identity-based copy

This script materializes the frames picked by count_waymo.py into
data/waymo_{split}_interesting/ by matching identities, not indices.

End-to-end pipeline we run
  1) Build base caches (no time filter):
       from graid.data.ImageLoader import WaymoDataset
       WaymoDataset(split='training',   rebuild=True, use_time_filtered=False)
       WaymoDataset(split='validation', rebuild=True, use_time_filtered=False)

  2) Generate per-scene best-frame JSONs:
       python graid/src/graid/data/count_waymo.py --split both

  3) Select exact frames by identity (segment, timestamp, camera=1):
       python graid/src/graid/data/select_waymo.py --split training
       python graid/src/graid/data/select_waymo.py --split validation

Rationale
  Index-based selection breaks whenever caches are rebuilt or filtered, causing
  dropped frames. Identity-based selection keeps counts aligned with the JSON
  (e.g., training=798, validation=202) and preserves night scenes.
"""

import argparse
import json
import os
import subprocess
import pickle
from pathlib import Path

from graid.utilities.common import project_root_dir
from tqdm import tqdm


def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def _identity_from_json(entry_key: str, entry: dict[str, any]) -> tuple[str, int, int]:
    """Return (segment_context_name, timestamp_micros) from JSON entry."""
    # entry_key is the parquet filename e.g. "abc123.parquet"
    segment = Path(entry_key).stem  # strip extension -> segment_context_name
    ts = int(entry["key.frame_timestamp_micros"])
    cam = 1  # count_waymo chooses only camera 1 (front)
    return segment, ts, cam


def _identity_from_pkl(pkl_path: Path) -> tuple[str, int, int]:
    """Return (segment_context_name, timestamp_micros) from saved PKL."""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    segment = str(data["name"])
    ts = int(str(data["timestamp"]))
    try:
        cam = int(str(data["path"]).rsplit("_", 1)[-1])
    except Exception:
        cam = -1
    return segment, ts, cam


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy interesting files based on validation json."
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["validation", "training"],
        required=True,
        help="Choose between validation or training.",
    )
    args = parser.parse_args()
    split = args.split

    file_path = f"{split}_best_frames.json"
    data = load_json(file_path)
    # Build identity set from JSON
    interesting_identities = {
        _identity_from_json(k, v) for k, v in data.items()
    }

    directory_path = str(project_root_dir() / f"data/waymo_{split}")
    interesting_directory_path = str(
        project_root_dir() / f"data/waymo_{split}_interesting"
    )

    os.makedirs(interesting_directory_path, exist_ok=True)

    dir_path = Path(directory_path)
    for pkl_file in tqdm(dir_path.glob("*.pkl"), desc="Processing files"):
        try:
            identity = _identity_from_pkl(pkl_file)
        except Exception:
            continue
        if identity in interesting_identities:
            subprocess.run(["cp", str(pkl_file), interesting_directory_path])

    print("renaming...")
    files = [f for f in os.listdir(interesting_directory_path) if f.endswith(".pkl")]
    files.sort()

    for i, file_name in enumerate(files):
        old_file = os.path.join(interesting_directory_path, file_name)
        temp_file = os.path.join(interesting_directory_path, f"temp_{i}.pkl")
        os.rename(old_file, temp_file)

    temp_files = sorted(
        f for f in os.listdir(interesting_directory_path) if f.startswith("temp_") and f.endswith(".pkl")
    )

    for i, file_name in enumerate(temp_files):
        old_file = os.path.join(interesting_directory_path, file_name)
        new_file = os.path.join(interesting_directory_path, f"{i}.pkl")
        os.rename(old_file, new_file)
