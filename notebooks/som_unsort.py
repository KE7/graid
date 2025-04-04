import os

import cv2
import supervision as sv
from scenic_reasoning.utilities.common import get_default_device
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

CHECKPOINT_PATH = "../evals/sam_vit_h_4b8939.pth"
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

DEVICE = get_default_device()
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

IMAGE_PATH = "../demo/demo2.jpg"
MIN_AREA_PERCENTAGE = 0.005
MAX_AREA_PERCENTAGE = 0.05


# load image
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# segment image
sam_result = mask_generator.generate(image_rgb)
detections = sv.Detections.from_sam(sam_result=sam_result)

# filter masks
height, width, channels = image_bgr.shape
image_area = height * width

min_area_mask = (detections.area / image_area) > MIN_AREA_PERCENTAGE
max_area_mask = (detections.area / image_area) < MAX_AREA_PERCENTAGE
detections = detections[min_area_mask & max_area_mask]


# setup annotators
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.3)
label_annotator = sv.LabelAnnotator(
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER,
    text_scale=0.5,
    text_color=sv.Color.WHITE,
    color=sv.Color.BLACK,
    text_thickness=1,
    text_padding=5,
)

# annotate
labels = [str(i) for i in range(len(detections))]
print(len(labels))

annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels
)

cv2.imwrite("annotated_image_unsort.jpg", annotated_image)
print("done!")
