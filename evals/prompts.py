import difflib
import openai
import re
import os
from scenic_reasoning.utilities.common import get_default_device
import supervision as sv
import cv2

class PromptingStrategy:
    """Base class for different prompting strategies."""

    def generate_prompt(self, query):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Each subclass must implement this method.")

class ZeroShotPrompt(PromptingStrategy):
    """Zero-shot prompting method."""
    def generate_prompt(self, image, question):
        return question

class FewShotPrompt(PromptingStrategy):
    """Few-shot prompting method."""
    def __init__(self, examples):
        """
        Args:
            examples (list): List of (input, output) examples for few-shot prompting.
        """
        self.examples = examples

    def generate_prompt(self, query):
        if not self.examples:
            raise ValueError("Few-shot examples are required but not provided.")
        
        prompt = "Here are some examples:\n"
        for i, (inp, out) in enumerate(self.examples):
            prompt += f"Example {i+1}:\nInput: {inp}\nOutput: {out}\n\n"

        prompt += f"Now, answer the following question:\n{query}"
        return prompt

class SetOfMarkPrompt(PromptingStrategy):
    """Set-of-mark prompting method."""
    def __init__(self):
        """
        Args:
            set_of_mark (list): List of constraints or reference points.
        """
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
        print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

        DEVICE = get_default_device()
        MODEL_TYPE = "vit_h"

        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.MIN_AREA_PERCENTAGE = 0.005
        self.MAX_AREA_PERCENTAGE = 0.05

    def generate_prompt(self, image, question):
        # load image
        image_bgr = cv2.imread(image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # segment image
        sam_result = self.mask_generator.generate(image_rgb)
        detections = sv.Detections.from_sam(sam_result=sam_result)

        # filter masks
        height, width, channels = image_bgr.shape
        image_area = height * width

        min_area_mask = (detections.area / image_area) > self.MIN_AREA_PERCENTAGE
        max_area_mask = (detections.area / image_area) < self.MAX_AREA_PERCENTAGE
        detections = detections[min_area_mask & max_area_mask]

        # setup annotators
        mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            opacity=0.3
        )
        label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            text_position=sv.Position.CENTER,
            text_scale=0.5,
            text_color=sv.Color.WHITE,
            color=sv.Color.BLACK,
            text_thickness=1,
            text_padding=5
        )

        # annotate
        labels = [str(i) for i in range(len(detections))]

        annotated_image = mask_annotator.annotate(
            scene=image_bgr.copy(), detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)
        
        return question, annotated_image

