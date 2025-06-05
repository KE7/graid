import os
from textwrap import dedent

import cv2
import numpy as np
import supervision as sv
import torch
from graid.utilities.common import get_default_device


class PromptingStrategy:
    """Base class for different prompting strategies."""

    def generate_prompt(self, image, question):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Each subclass must implement this method.")


class ZeroShotPrompt(PromptingStrategy):
    """Zero-shot prompting method."""

    def __init__(self, using_cd=False):
        self.using_cd = using_cd
        self.ans_format_str = (
            " Make sure to wrap the answer in triple backticks. '```'"
            if not self.using_cd
            else ""
        )

    def generate_prompt(self, image, question):
        prompt = f"""\
            Answer the following question related to the image. If this question involves object naming, you may only identify objects from the COCO dataset (80 labels).{self.ans_format_str}

            Here's the question: {question}. 
        """

        return image, dedent(prompt)

    def __str__(self):
        return "ZeroShotPrompt"


class ZeroShotPrompt_batch(PromptingStrategy):
    """Zero-shot prompting method."""

    def generate_prompt(self, image, question):
        prompt = f"""\
        Answer the following questions related to the image. Provide your answers to each question, separated by commas. Here are the questions:
        {question}
        """

        return image, dedent(prompt)

    def __str__(self):
        return "ZeroShotPrompt_batch"


class CoT(PromptingStrategy):
    """CoT prompting method."""

    def generate_prompt(self, image, question):
        prompt = f"""\
        Look at the image carefully and think through each question step by step. Use the process below to guide your reasoning and arrive at the correct answer. Here are some examples of how to answer the question:

        Question: Are there any motorcyclists to the right of any pedestrians? 

        Steps:
        1. I see three pedestrians walking on the left sidewalk, roughly in the left third of the image.
        2. I also see a single motorcyclist riding away from the camera, positioned nearer the center of the road and center of the camera frame but clearly to the right of those pedestrians.
        3. Comparing their horizontal positions, the motorcyclist’s x‑coordinate is larger (further to the right) than either pedestrian’s.

        Conclusion: The motorcyclist is to the right of the pedestrians.
        Final_Answer: Yes.


        Question: What group of objects are most clustered together?

        Steps:
        1. Scanning for COCO categories only, I identify the following objects
        2. Person:
            I spot three pedestrians on the left sidewalk: one nearest the foreground, one a few meters behind, and a third just past the white box truck.
            They are spaced roughly 2–3 m apart along the sidewalk.

        3. Motorcycle
            A single motorcyclist is riding down the center of the road, about midway up the frame.
            Only one instance, so no clustering.

        4. Truck
            A single white box truck is parked on the left curb beyond the first two pedestrians.
            Again only one, so no cluster.

        5. Car
            At least six cars parked behind the french on the right and at least four cars in the distance near the center of the image
            Both clusters of cars, especially the parked ones behind the fence occupy a small contiguous area, tightly packed together.


        Conclusion: We can compare the densities of the groups we found.
            The three people, while grouped, are separated by a few meters each.    
            The six-plus cars are parked immediately adjacent in a compact line.

        Final_Answer: The cars are the most clustered together.


        Question: Does the leftmost object in the image appear to be wider than it is tall?

        Steps:
        1. Among the COCO categories present, the object farthest to the left is the bench under the bus‐stop canopy.
        2. That bench’s bounding area is much broader horizontally than it is tall vertically.

        Conclusion: The bench is wider than it is tall.
        Final_Answer: Yes.

        Now that you have seen the examples, answer only the following question in the same step-by-step reasoning format as the examples:
        Question: {question}

        """
        return image, dedent(prompt)

    def __str__(self):
        return "CoT"


class FewShotPrompt(PromptingStrategy):
    """Few-shot prompting method."""

    def __init__(self, examples):
        """
        Args:
            examples (list): List of (input, output) examples for few-shot prompting.
        """
        self.examples = examples

    def generate_prompt(self, image, question):
        if not self.examples:
            raise ValueError("Few-shot examples are required but not provided.")

        prompt = "Here are some examples:\n"
        for i, (inp, out) in enumerate(self.examples):
            prompt += f"Example {i+1}:\nInput: {inp}\nOutput: {out}\n\n"

        prompt += f"Now, answer the following question:\n{question}"
        return prompt

    def __str__(self):
        return "FewShotPrompt"


class SetOfMarkPrompt(PromptingStrategy):
    def __init__(self, gpu=1):
        from segment_anything import (
            SamAutomaticMaskGenerator,
            SamPredictor,
            sam_model_registry,
        )

        CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
        print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

        DEVICE = get_default_device()
        MODEL_TYPE = "vit_h"

        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(
            device=f"cuda:{gpu}"
        )
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.MIN_AREA_PERCENTAGE = 0.005
        self.MAX_AREA_PERCENTAGE = 0.05

    def generate_prompt(self, image, question):
        prompt = f"""Answer the following question related to the image. If this question involves object naming, you may only identify objects from the COCO dataset (80 labels). Make sure to wrap the answer in triple backticks. "```"
        Here's the question: {question}. 
        """

        if isinstance(image, str):
            image_bgr = cv2.imread(image)
        elif isinstance(image, torch.Tensor):
            image_bgr = image.mul(255).permute(1, 2, 0).numpy().astype(np.uint8)
        else:
            raise ValueError(
                f"Input image should be either a numpy array or a tensor of shape (B, C, H, W), but got {image.shape}"
            )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        sam_result = self.mask_generator.generate(image_rgb)
        detections = sv.Detections.from_sam(sam_result=sam_result)

        height, width, channels = image_bgr.shape
        image_area = height * width

        min_area_mask = (detections.area / image_area) > self.MIN_AREA_PERCENTAGE
        max_area_mask = (detections.area / image_area) < self.MAX_AREA_PERCENTAGE
        detections = detections[min_area_mask & max_area_mask]

        def Find_Center(mask: np.ndarray) -> tuple[int, int]:
            mask_8u = mask.astype(np.uint8)

            # Distance transform
            dist = cv2.distanceTransform(mask_8u, distanceType=cv2.DIST_L2, maskSize=3)

            # Find the global maximum in distance map
            _, _, _, max_loc = cv2.minMaxLoc(dist)
            return max_loc

        def Mark_Allocation(masks: list[np.ndarray]) -> list[tuple[int, int]]:
            # 1) Sort all masks by ascending area
            #    (Compute area by summing pixels in each mask.)
            areas = [mask.sum() for mask in masks]
            sort_indices = np.argsort(areas)  # ascending
            sorted_masks = [masks[i] for i in sort_indices]

            # Prepare an "excluded region" mask to carve out overlaps
            h, w = sorted_masks[0][0].shape
            excluded = np.zeros((h, w), dtype=np.uint8)

            centers = []
            for i, mask_ in enumerate(sorted_masks):
                # Convert to 8-bit for bitwise ops if necessary
                mask_8u = mask_[0].astype(np.uint8)

                # Exclude overlapping area with previously processed masks
                # final_mask = mask & NOT(excluded)
                final_mask = cv2.bitwise_and(mask_8u, cv2.bitwise_not(excluded))

                center_xy = Find_Center(final_mask)
                centers.append(center_xy)

                excluded = cv2.bitwise_or(excluded, final_mask)

            return centers

        all_masks = [detections[i].mask for i in range(len(detections))]

        if not all_masks:
            return image, prompt

        centers = Mark_Allocation(all_masks)

        # 6) We need to reorder the Detections as well to match the sorted area order
        sorted_idx = np.argsort(detections.area)
        sorted_detections = detections[sorted_idx]

        mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX, opacity=0.3
        )
        annotated_image = image_bgr.copy()

        annotated_image = mask_annotator.annotate(
            scene=annotated_image, detections=detections
        )

        for idx, (x, y) in enumerate(centers, start=1):
            cv2.circle(annotated_image, (x, y), 11, (0, 0, 0), -1)
            cv2.putText(
                annotated_image,
                str(idx),
                (x - 6, y + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated_image, prompt

    def __str__(self):
        return "SetOfMarkPrompt"
