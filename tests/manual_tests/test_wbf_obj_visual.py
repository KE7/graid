import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
from pathlib import Path
from typing import Any

from graid.data.ImageLoader import Bdd100kDataset, NuImagesDataset, WaymoDataset
from graid.interfaces.ObjectDetectionI import ObjectDetectionResultI
from graid.models.MMDetection import MMdetection_obj
from graid.models.Ultralytics import RT_DETR, Yolo
from graid.models.WBF import WBF
from graid.utilities.common import get_default_device, project_root_dir, yolo_nuscene_transform, yolo_waymo_transform
from graid.models.Detectron import Detectron_obj

from PIL import Image, ImageDraw


def filter_detections_by_score(
    detections: list[ObjectDetectionResultI], min_score: float = 0.4
) -> list[ObjectDetectionResultI]:
    """Filter out detections with scores below the minimum threshold."""
    return [det for det in detections if det.score >= min_score]


def draw_boxes(
    image: np.ndarray[Any, np.dtype[np.uint8]], detections: list[ObjectDetectionResultI], alpha: float = 1.0
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Overlay detections on an RGB image and return the visualised image."""

    colours: list[tuple[int, int, int]] = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (255, 192, 203),
        (128, 128, 0),
    ]

    # Pillow path ---------------------------------------------------
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img, "RGBA")
    for i, det in enumerate(detections):
        colour = colours[i % len(colours)] + (255,)
        x1, y1, x2, y2 = map(int, det.as_xyxy().squeeze()[:4].tolist())
        label = f"{det.label}: {det.score:.1%}"
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
        text_size = draw.textlength(label)
        draw.rectangle([x1, y1 - 15, x1 + text_size + 4, y1], fill=colour)
        draw.text((x1 + 2, y1 - 14), label, fill=(0, 0, 0, int(255 * alpha)))
        print(
            f"Found {det.label}: {det.score:.1%} at {x1/image.shape[1]}, {y1/image.shape[0]}, {x2/image.shape[1]}, {y2/image.shape[0]}")
    return np.array(pil_img)


# ----------------------------------------------------------------------------
# Model loading helpers
# ----------------------------------------------------------------------------

def load_dino() -> MMdetection_obj:
    mmdet = project_root_dir() / "install" / "mmdetection"
    cfg = str(mmdet / "configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py")
    ckpt = (
        "https://download.openmmlab.com/mmdetection/v3.0/dino/"
        "dino-5scale_swin-l_8xb2-12e_coco/"
        "dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth"
    )
    return MMdetection_obj(cfg, ckpt, device=get_default_device())


def load_codetr() -> MMdetection_obj:
    mmdet = project_root_dir() / "install" / "mmdetection"
    cfg = str(
        mmdet
        / "projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py"
    )
    ckpt = (
        "https://download.openmmlab.com/mmdetection/v3.0/codetr/"
        "co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth"
    )
    return MMdetection_obj(cfg, ckpt, device=get_default_device())


def load_rtdetr() -> RT_DETR:
    return RT_DETR("rtdetr-x.pt")


def load_yolo_v10x() -> Yolo:
    return Yolo("yolov10x.pt")


def load_mask_rcnn_detectron() -> Detectron_obj:
    """Load Detectron2 Mask R-CNN R-50 FPN model."""
    # Use a simpler, well-supported model config
    cfg = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ckpt = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    return Detectron_obj(cfg, ckpt, device=get_default_device())


def main():
    print("=== WBF Ensemble Bounding-Box Visualisation ===")

    NUM_IMAGES = 3
    SAVE = True
    SHOW = True
    SCORE_THRESHOLD = 0.25  # Minimum confidence score for detections

    datasets = []

    # BDD100K dataset
    print("Loading BDD100K validation images …")
    bdd_dataset = Bdd100kDataset(split="val")
    datasets.append(("BDD100K", "val", bdd_dataset))
    print(f"✓ BDD100K loaded successfully ({len(bdd_dataset)} images)")

    # NuImages dataset
    print("Loading NuImages validation images …")
    nuimages_dataset = NuImagesDataset(
        split="val",
        transform=lambda i, l: yolo_nuscene_transform(
            i, l, new_shape=(896, 1600))
    )
    datasets.append(("NuImages", "val", nuimages_dataset))
    print(
        f"✓ NuImages loaded successfully ({len(nuimages_dataset)} images)")

    # Waymo dataset
    print("Loading Waymo validation images …")
    waymo_dataset = WaymoDataset(
        split="validation",
        transform=lambda i, l: yolo_waymo_transform(i, l, (1280, 1920))
    )
    datasets.append(("Waymo", "validation", waymo_dataset))
    print(f"✓ Waymo loaded successfully ({len(waymo_dataset)} images)")

    if not datasets:
        raise RuntimeError("No datasets could be loaded successfully!")

    print(f"\nSuccessfully loaded {len(datasets)} dataset(s)")

    print("Initialising base models …")
    dino = load_dino()
    codetr = load_codetr()
    rtdetr = load_rtdetr()
    yolo10x = load_yolo_v10x()
    mask_rcnn = load_mask_rcnn_detectron()

    # Assemble WBF
    ensemble = WBF(
        detectron2_models=[mask_rcnn],
        mmdet_models=[dino, codetr],
        ultralytics_models=[rtdetr, yolo10x],
        model_weights=[0.8, 0.8, 1.0, 0.9, 0.8],
        iou_threshold=0.55,
        skip_box_threshold=0.01,
    )
    print("WBF ensemble ready!")
    print(f"Using score threshold: {SCORE_THRESHOLD}")

    out_dir = Path("wbf_results")
    if SAVE:
        out_dir.mkdir(exist_ok=True)
        print(f"Saving results to {out_dir.resolve()}")

    # Process images from each dataset
    for dataset_name, split, dataset in datasets:
        print(f"\n--- Processing {dataset_name} dataset ---")

        for idx, data in enumerate(islice(dataset, NUM_IMAGES)):
            img_tensor = data["image"]
            filename = data["name"]

            # Sanitize and shorten filename
            filename = filename.replace("/", "_").replace("\\", "_")
            short_filename = Path(filename).stem

            # Create helpful filename: dataset_split_shortname_wbf.png
            output_filename = f"{dataset_name.lower()}_{split}_{short_filename}.png"

            # Convert CHW tensor → HWC numpy in the correct value range
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

            # Dataset may already be in [0,255] – mimic logic from
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            print(f"[{idx+1}/{NUM_IMAGES}] {dataset_name} - {filename}")
            detections = ensemble.identify_for_image(img_tensor)
            print(f"  → {len(detections)} raw detections")

            # Apply post-processing filter
            filtered_detections = filter_detections_by_score(
                detections, SCORE_THRESHOLD)
            print(
                f"  → {len(filtered_detections)} detections after filtering (score >= {SCORE_THRESHOLD})")

            if len(filtered_detections) == 0:
                print("  → No detections remain after filtering, skipping visualization")
                continue

            vis = draw_boxes(img_np, filtered_detections)

            if SHOW:
                plt.figure(figsize=(10, 6))
                plt.imshow(vis)
                plt.title(
                    f"WBF fused detections (score >= {SCORE_THRESHOLD}) – {dataset_name} - {filename}")
                plt.axis("off")
                plt.show()

            if SAVE:
                save_path = out_dir / output_filename
                Image.fromarray(vis).save(save_path)
                print(f"  Saved → {save_path.relative_to(out_dir.parent)}")

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
