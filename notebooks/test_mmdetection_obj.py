from itertools import islice
from scenic_reasoning.data.ImageLoader import Bdd100kDataset
from scenic_reasoning.measurements.ObjectDetection import ObjectDetectionMeasurements
from scenic_reasoning.utilities.common import get_default_device
from scenic_reasoning.models.MMdetection import MMDetection_obj
import torch
from ultralytics.data.augment import LetterBox
from PIL import Image
import numpy as np
import pdb

from mmdet.apis import DetInferencer, init_detector, inference_detector

NUM_EXAMPLES_TO_SHOW = 1
BATCH_SIZE = 1

shape_transform = LetterBox(new_shape=(768, 1280))
bdd = Bdd100kDataset(
    split="val", 
    # YOLO requires images to be 640x640 or 768x1280, 
    # but BDD100K images are 720x1280 so we need to resize
    use_original_categories=False,
    use_extended_annotations=False,
)


threshold = 0.5


# Below is my attempt of using batch prediction. There's no obvious speed gain unfortunately.
# from detectron2 import model_zoo
# from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog
# from detectron2.engine import DefaultPredictor
# from detectron2.structures import BitMasks
# from detectron2.utils.visualizer import Visualizer
# from detectron2.modeling import build_model
# from detectron2.checkpoint import DetectionCheckpointer

# print("!!!!!!!!!!!", model_zoo.get_checkpoint_url(weights_file))
# weight_url = model_zoo.get_checkpoint_url(weights_file)

# cfg = get_cfg()
# cfg.merge_from_file("../install/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set threshold for this model
# cfg.MODEL.DEVICE = str(get_default_device())

# model = build_model(cfg) # returns a torch.nn.Module
# DetectionCheckpointer(model).load(weight_url) # must load weights this way, can't use cfg.MODEL.WEIGHTS = "..."
# model.train(False) # inference mode

sample_img_path = "/Users/harry/Desktop/Nothing/sky/scenic-reasoning/data/bdd100k/images/100k/val/b1c9c847-3bda4659.jpg"
sample_img = Image.open(sample_img_path)
sample_img = np.array(sample_img)
# sample_img_tensor = decode_image(sample_img_path)

# inputs = model([{"image": sample_img_tensor, "image": sample_img_tensor}])

# outputs = model(inputs)
# print(outputs)

# exit()

config_file = "/Users/harry/Desktop/Nothing/sky/scenic-reasoning/install/mmdetection/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py"
checkpoint_file = "/Users/harry/Desktop/Nothing/sky/scenic-reasoning/install/mmdetection/checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"

# inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')

model = MMDetection_obj(config_file, checkpoint_file, device=torch.device("cpu"))
# print(model.identify_for_image(sample_img))

# exit()
measurements = ObjectDetectionMeasurements(model, bdd, batch_size=BATCH_SIZE, collate_fn=lambda x: x) # hacky way to avoid RuntimeError: each element in list of batch should be of equal size

# WARNING ⚠️ imgsz=[720, 1280] must be multiple of max stride 64, updating to [768, 1280]
from pprint import pprint
for (results, ims) in islice(measurements.iter_measurements(
        device=get_default_device(), 
        imgsz=[720, 1280],
        bbox_offset=24,
        debug=True,
        conf=0.1,
        class_metrics=True,
        extended_summary=True,
        ), 
    NUM_EXAMPLES_TO_SHOW):
    pprint(results)
    [im.show() for im in ims]

