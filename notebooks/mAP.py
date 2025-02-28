from detectron2.data import build_detection_test_loader, MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import json
from scenic_reasoning.data.ImageLoader import Bdd100kDataset
from scenic_reasoning.utilities.common import get_default_device, yolo_bdd_transform, yolo_nuscene_transform
from scenic_reasoning.utilities.coco import coco_label
from torch.utils.data import DataLoader

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from scenic_reasoning.utilities.common import get_default_device


'''
j = [{
    "file_name": "demo/1.jpg",
    "height": 120,
    "width": 120,
    "image_id": 1,
    "annotations": [
        {
            "bbox": [876.8, 447.2, 1003.2, 520.8],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 2,
            "iscrowd": 0
        },
        {
            "bbox": [776.0, 445.6, 824.0, 479.2],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 1,
            "iscrowd": 0
        },
        {
            "bbox": [1008.0, 420.0, 1232.8, 560.8],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 1,
            "iscrowd": 0
        }
    ],
} ]

'''

# Script for generating the annotation.json file
# data = []
# bdd = Bdd100kDataset(
#     split="val",
#     # transform=lambda i, l: yolo_bdd_transform(i, l, new_shape=(768, 1280)),
#     use_original_categories=False,
#     use_extended_annotations=False,
# )

# data_loader = DataLoader(bdd, batch_size=1, shuffle=False, collate_fn=lambda x: x)

# count = 0
# for batch in data_loader:
#     if count == 10:
#         break
#     labels = batch[0]["labels"]
#     annotations = []
#     for l in labels:
#         annotations.append({
#             "bbox": l.as_xyxy().tolist()[0],
#             "bbox_mode": BoxMode.XYXY_ABS,
#             "category_id": l._class,
#             "iscrowd": 0
#         })
#     # labels = [l.as_xyxy().tolist()[0] for l in labels]
    
#     # import pdb
#     # pdb.set_trace()
#     h, w = batch[0]["image"].shape[1:]

#     data.append({
#         "file_name": batch[0]["path"],
#         "height": h,
#         "width": w,
#         "image_id": 1,
#         "annotations": annotations
#     })
#     count += 1

#     with open("/Users/harry/Desktop/Nothing/sky/scenic-reasoning/notebooks/annotation.json", "w") as f:
#         json.dump(data, f, indent=4)

def my_dataset_function():
    with open("/Users/harry/Desktop/Nothing/sky/scenic-reasoning/notebooks/annotation.json") as f:
        annotations = json.load(f)

    return annotations

# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset", {}, "/Users/harry/Desktop/Nothing/sky/scenic-reasoning/notebooks/annotation.json", "/Users/harry/Desktop/Nothing/sky/scenic-reasoning/demo")


if __name__ == "__main__":

    DatasetCatalog.register("my_dataset", my_dataset_function)
    MetadataCatalog.get("my_dataset").set(thing_classes=list(coco_label.values()))
    d = DatasetCatalog.get("my_dataset")

    print(d)

    config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    weights_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.MODEL.DEVICE = str(get_default_device())
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("my_dataset", output_dir="../output")

    val_loader = build_detection_test_loader(cfg, "my_dataset", batch_size = 1, num_workers = 0)

    print("loaded up")

    print(inference_on_dataset(predictor.model, val_loader, evaluator))


# reference notebook: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=h9tECBQCvMv3