from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from graid.data.ImageLoader import Bdd100kDataset
from graid.interfaces.ObjectDetectionI import ObjectDetectionResultI


def convert_bdd_to_yolo():
    """
    Converts BDD100K dataset annotations to YOLO format.

    This script processes the 'train', 'val', and 'test' splits of the
    BDD100K dataset. For each image containing objects, it generates a
    corresponding '.txt' file in YOLO format.

    The YOLO format consists of one line per object, with each line
    containing the class ID and the normalized bounding box coordinates
    (center_x, center_y, width, height).

    The class IDs are derived from the original BDD100K categories,
    mapped to a zero-indexed integer.

    Generated label files are stored in a 'yolo_labels' directory,
    organized by dataset split.
    """
    root_output_dir = Path("data/bdd100k/yolo_labels")
    print(f"Output directory: {root_output_dir.resolve()}")

    # Get categories from the Bdd100kDataset class
    category_map = {v: k for k, v in Bdd100kDataset._CATEGORIES.items()}
    print("BDD100K Class to ID mapping (from Bdd100kDataset):")
    for class_id, name in sorted(category_map.items()):
        print(f"  {class_id}: {name}")

    for split in ["train", "val"]:
        print(f"\nProcessing '{split}' split...")

        dataset = Bdd100kDataset(
            split=split,
            use_original_categories=True,
            use_time_filtered=False
        )

        output_dir = root_output_dir / split
        output_dir.mkdir(parents=True, exist_ok=True)

        labels_generated = 0

        for i in tqdm(range(len(dataset)), desc=f"Exporting {split}"):
            item = dataset[i]
            image_name = item["name"]
            # The 'labels' are a list of ObjectDetectionResultI objects
            detections: list[ObjectDetectionResultI] = item["labels"]

            if not detections:
                continue

            yolo_lines = []
            for det in detections:
                # The class ID is directly available in the detection object
                class_id = det.cls

                # as_xywhn() provides normalized [x_center, y_center, width, height]
                # It returns a tensor, so we get the first (and only) row
                xywhn = det.as_xywhn()[0]
                x_center, y_center, width, height = xywhn

                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

            if yolo_lines:
                label_path = output_dir / f"{Path(image_name).stem}.txt"
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_lines))
                labels_generated += 1

        if labels_generated == 0:
            print(f"\nWARNING: No label files were generated for the '{split}' split.")
            print("This could be because the dataset split is empty, contains no annotations,")
            print("or all annotations were filtered out.")
            print("This can cause errors during training/validation if the framework expects labels.")
        else:
            print(f"\nSuccessfully generated {labels_generated} label files for the '{split}' split.")

    print("\nConversion to YOLO format complete.")


if __name__ == "__main__":
    convert_bdd_to_yolo() 