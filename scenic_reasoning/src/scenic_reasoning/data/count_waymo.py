import logging
import os
from scenic_reasoning.utilities.common import project_root_dir
import pandas as pd

# waymo = WaymoDataset(split="validation", transform=lambda i, l: yolo_waymo_transform(i, l, (768, 1280)))
logger = logging.getLogger(__name__)


split = "validation"
root_dir = project_root_dir() / "data" / "waymo"
camera_img_dir = root_dir / f"{split}" / "camera_image"
camera_box_dir = root_dir / f"{split}" / "camera_box"

if not os.path.exists(camera_img_dir) or not os.path.exists(
    camera_box_dir
):
    raise FileNotFoundError(
        f"Directories not found: {camera_img_dir}, {camera_box_dir}"
    )

# Initialize img_labels
img_labels = []

# Get the camera image files in the directory
camera_image_files = [
    f for f in os.listdir(camera_img_dir) if f.endswith(".parquet")
]

# camera_image_files = camera_image_files[
#     :10
# ]  # TODO: doing this because using the entire validation gives us memory issue. Need to change later.

# Check if image files are found
if not camera_image_files:
    raise FileNotFoundError(
        f"No parquet image files found in {camera_img_dir}"
    )

merged_dfs = []
context_set = set()
for image_file in camera_image_files:
    box_file = image_file.replace("camera_image", "camera_box")
    image_path = camera_img_dir / image_file
    box_path = camera_box_dir / box_file

    # Load the dataframes
    image_df = pd.read_parquet(image_path)
    box_df = pd.read_parquet(box_path)

    unique_images_df = box_df.groupby(
        [
            "key.segment_context_name",
            "key.frame_timestamp_micros",
            "key.camera_name",
        ]
    )

    # Merge image and box data
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
    else:
        logger.debug(f"Merged DataFrame for {image_file}: {merged_df.shape}\n")
        # merged_dfs.append(merged_df)
        unique_frame = merged_df["key.segment_context_name"].unique().tolist()
        print(unique_frame)
        context_set.update(unique_frame)
        # if not frames:
        #     continue
        # for frame in frames:
        #     if frame not in context_set:
        #         context_set.add(frame)


print(len(context_set))
    
    

