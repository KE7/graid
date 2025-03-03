import logging
import os
from scenic_reasoning.utilities.common import project_root_dir
import pandas as pd
from scenic_reasoning.utilities.common import convert_to_xyxy
import json
from tqdm import tqdm


logger = logging.getLogger(__name__)


def metric(data_per_scene):
    """
    Computes a score that evenly weights the max number of bounding boxes 
    and the largest area of a bounding box.
    
    :param bboxes: List of bounding boxes, each represented as (x, y, w, h)
    :return: Weighted score
    """

    N_max = 0  # Track the max mean area found
    A_max = 0  # Track the max number of boxes found

    for image in data_per_scene:
        N_max = max(N_max, image['N'])
        A_max = max(A_max, image['A'])
    

    best_score = 0
    best_time = None
    best_camera = None

    for image in data_per_scene:
        if not bboxes:
            continue
        
        N = image['N']
        A = image['A']
        N_score = min(N / N_max, 1)
        A_score = min(A / A_max, 1)
        curr_score = 0.5 * (N_score + A_score)
        if curr_score > best_score:
            best_score = curr_score
            best_time = image["key.frame_timestamp_micros"]
            best_camera = image["key.camera_name"]
    
    return best_score, best_time, best_camera



def main(camera_img_dir, camera_box_dir, split):

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

    data = {}
    for image_file in tqdm(camera_image_files, desc="Processing images"):
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
            # Each group has one unique image frame, in which all the detected objects belong to
            image_data = group_data.iloc[0]
            img_bytes = image_data["[CameraImageComponent].image"]

            labels = []
            bboxes = []
            for _, row in group_data.iterrows():
                bbox = convert_to_xyxy(
                            row["[CameraBoxComponent].box.center.x"],
                            row["[CameraBoxComponent].box.center.y"],
                            row["[CameraBoxComponent].box.size.x"],
                            row["[CameraBoxComponent].box.size.y"],
                        )
            
                bboxes.append(bbox)

            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]
            mean_area = sum(areas) / len(areas)
            data_per_scene.append({
                "key.frame_timestamp_micros": group_name[1], 
                "key.camera_name": group_name[2], 
                "A": mean_area,
                "N": len(bboxes)
                })
            
        best_score, best_time, best_camera = metric(data_per_scene)
        
        print(f"Best score: {best_score}, Best time: {best_time}, Best camera: {best_camera}")
        data[image_file] = {
            "key.frame_timestamp_micros": int(best_time), 
            "key.camera_name": int(best_camera), 
            "score": best_score
            }
        
    output_file = f"{split}_best_frames.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Data saved to {output_file}")

    return data




if __name__ == "__main__":
    
    split = "validation"
    root_dir = project_root_dir() / "data" / "waymo"
    camera_img_dir = root_dir / f"{split}" / "camera_image"
    camera_box_dir = root_dir / f"{split}" / "camera_box"

        # Read in the JSON file as data
    input_file = f"{split}_best_frames.json"

    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            data = json.load(f)
    else:
        data = main(camera_img_dir, camera_box_dir, split)

    camera_name_counts = {}
    for image_file, details in data.items():
        camera_name = details["key.camera_name"]
        if camera_name not in camera_name_counts:
            camera_name_counts[camera_name] = 0
        camera_name_counts[camera_name] += 1

    for camera_name, count in camera_name_counts.items():
        print(f"Camera {camera_name}: {count}")
