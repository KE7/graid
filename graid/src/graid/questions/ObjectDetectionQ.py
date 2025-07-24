import logging
import math
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from PIL import Image

from graid.interfaces.ObjectDetectionI import (
    ObjectDetectionResultI,
    ObjectDetectionUtils,
)

logger = logging.getLogger(__name__)


class Question(ABC):
    @abstractmethod
    def __init__(
        self, question: str, variables: list[str], predicates: list[Callable]
    ) -> None:
        self.question = question
        self.variables = variables
        self.predicates = predicates

    def is_applicable(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> bool:
        """
        Check if the question is applicable to the given image and detections.

        Args:
            image: The image to check.
            detections: A list of ObjectDetectionResultI objects corresponding to the image.

        Returns:
            bool: True if all predicates return True, False otherwise.
        """
        return all(predicate(image, detections) for predicate in self.predicates)

    # TODO: cache the results of the predicates to avoid recomputing them.
    #   because we need them when converting the question to a string.

    def _find_extremes(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        # for every kind (label) of object in the image, find the right most detection
        # label -> (center of bbox (x, y), bounding box (x1, y1, x2, y2))
        right_most_detections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # also the left most
        left_most_detections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # also the top most
        top_most_detections: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # also the lowest
        bottom_most_detections: dict[str,
                                     tuple[torch.Tensor, torch.Tensor]] = {}

        for detection in detections:
            class_name = detection.label
            center_box = detection.get_center()  # shape == (# of boxes, 2)
            bbox = detection.as_xyxy()  # shape == (# of boxes, 4)
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # find the right most bbox using the center of the bbox
                n = class_name.shape[0]
                for i in range(n):
                    curr_class_name = class_name[i]
                    curr_center_box = center_box[i]
                    curr_bbox = bbox[i]

                    # right most
                    if curr_class_name not in right_most_detections:
                        right_most_detections[curr_class_name] = (
                            curr_center_box,
                            curr_bbox,
                        )
                    else:
                        if (
                            curr_center_box[0]
                            > right_most_detections[curr_class_name][0]
                        ):
                            right_most_detections[curr_class_name] = (
                                curr_center_box,
                                curr_bbox,
                            )

                    # left most
                    if curr_class_name not in left_most_detections:
                        left_most_detections[curr_class_name] = (
                            curr_center_box,
                            curr_bbox,
                        )
                    else:
                        if (
                            curr_center_box[0]
                            < left_most_detections[curr_class_name][0]
                        ):
                            left_most_detections[curr_class_name] = (
                                curr_center_box,
                                curr_bbox,
                            )

                    # top most
                    if curr_class_name not in top_most_detections:
                        top_most_detections[curr_class_name] = (
                            curr_center_box,
                            curr_bbox,
                        )
                    else:
                        if curr_center_box[1] < top_most_detections[curr_class_name][1]:
                            top_most_detections[curr_class_name] = (
                                curr_center_box,
                                curr_bbox,
                            )

                    # bottom most
                    if curr_class_name not in bottom_most_detections:
                        bottom_most_detections[curr_class_name] = (
                            curr_center_box,
                            curr_bbox,
                        )
                    else:
                        if (
                            curr_center_box[1]
                            > bottom_most_detections[curr_class_name][1]
                        ):
                            bottom_most_detections[curr_class_name] = (
                                curr_center_box,
                                curr_bbox,
                            )

            else:  # type(class_name) == str
                # bbox would be shape (1, 4) so let's just grab the only element
                # right most
                if class_name not in right_most_detections:
                    right_most_detections[class_name] = (center_box[0], bbox[0])
                else:
                    if center_box[0][0] > right_most_detections[class_name][0][0]:
                        right_most_detections[class_name] = (
                            center_box[0], bbox[0])

                # left most
                if class_name not in left_most_detections:
                    left_most_detections[class_name] = (center_box[0], bbox[0])
                else:
                    if center_box[0][0] < left_most_detections[class_name][0][0]:
                        left_most_detections[class_name] = (
                            center_box[0], bbox[0])

                # top most
                if class_name not in top_most_detections:
                    top_most_detections[class_name] = (center_box[0], bbox[0])
                else:
                    if center_box[0][1] < top_most_detections[class_name][0][1]:
                        top_most_detections[class_name] = (
                            center_box[0], bbox[0])

                # bottom most
                if class_name not in bottom_most_detections:
                    bottom_most_detections[class_name] = (
                        center_box[0], bbox[0])
                else:
                    if center_box[0][1] > bottom_most_detections[class_name][0][1]:
                        bottom_most_detections[class_name] = (
                            center_box[0], bbox[0])

        return [
            left_most_detections,
            right_most_detections,
            top_most_detections,
            bottom_most_detections,
        ]

    @abstractmethod
    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        """
        Apply the question to the image and detections.

        @precondition: is_applicable(image, detections) == True
        Args:
            image: The image to apply the question to.
            detections: A list of ObjectDetectionResultI objects corresponding to the image.

        Returns:
            A list of question-answer pairs where each pair with the substituted appropriate
            classes and the answer to that question.

            For example:
            Image: A person is sitting on a chair.
            Question: How many <object_class> are there in this image?
            apply() -> [
                ("How many person(s) are there in this image?", "1"),
                ("How many chair(s) are there in this image?", "1"),
            ]
        """
        pass

    def __repr__(self):
        representation = f"Question: {self.question}"
        # Safely check if 'other_question' is defined and not None
        if getattr(self, "other_question", None) is not None:
            representation += f"\nOther Question: {self.other_question}"

        return representation


class ObjectDetectionPredicates:
    @staticmethod
    def at_least_one_single_detection(
        image: Image, detections: list[ObjectDetectionResultI]
    ) -> bool:
        if len(detections) == 0:
            return False
        if len(detections) == 1:
            # if there is only one detection, we can just return True
            return True

        # check if there is at least one detection with a single class
        counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for class_name in class_name:
                    counts[class_name] = counts.get(class_name, 0) + 1
            else:
                counts[class_name] = counts.get(class_name, 0) + 1

        return any(count == 1 for count in counts.values())

    @staticmethod
    def at_least_x_many_class_detections(
        image: Image, detections: list[ObjectDetectionResultI], x: int
    ) -> bool:
        counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for single_class_name in class_name:
                    counts[single_class_name] = counts.get(
                        single_class_name, 0) + 1
            else:
                counts[class_name] = counts.get(class_name, 0) + 1

        return len(counts) >= x

    @staticmethod
    def at_least_x_detections(
        image: Image, detections: list[ObjectDetectionResultI], x: int
    ) -> bool:
        return len(detections) >= 3

    @staticmethod
    def at_least_x_detections(
        image: Image, detections: list[ObjectDetectionResultI], x: int
    ) -> bool:
        return len(detections) >= 3

    @staticmethod
    def exists_non_overlapping_detections(
        image: Image, detections: list[ObjectDetectionResultI]
    ) -> bool:
        for i, detection1 in enumerate(detections):
            for j in range(i + 1, len(detections)):
                detection2 = detections[j]

                if detection1.label != detection2.label:
                    iou: torch.Tensor = ObjectDetectionUtils.pairwise_iou(
                        detection1, detection2
                    )
                    if iou.max() == 0:
                        return True

        return False

    @staticmethod
    def has_clusters(
        image: Image, detections: list[ObjectDetectionResultI], threshold=50
    ) -> bool:
        import numpy as np
        from scipy.spatial.distance import pdist, squareform

        # Get centers of all detections
        centers = []
        for detection in detections:
            bbox = detection.as_xyxy().squeeze(0)
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            centers.append((x_center, y_center))

        centers = np.array(centers)

        # Compute pairwise distances
        dists = squareform(pdist(centers))

        # Simple clustering by distance threshold (e.g., 50 pixels)
        visited = set()
        clusters = []

        for i in range(len(centers)):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(len(centers)):
                if j not in visited and dists[i][j] < threshold:
                    cluster.append(j)
                    visited.add(j)
            if len(cluster) >= 2:
                clusters.append(cluster)

        if not clusters:
            return False
        else:
            return True


class IsObjectCentered(Question):
    def __init__(self, buffer_ratio: float = 0.05) -> None:
        """Create an *Is-Object-Centered* question.

        Args:
            buffer_ratio: Fraction of the image width to treat as a no-ask buffer
                around the one-third and two-third vertical lines. A value such as
                ``0.05`` means 5 % of the image width on either side of the grid
                boundary will be treated as *ambiguous* – if any side of the
                bounding box falls in that zone, the question is skipped for
                that object.
        """
        super().__init__(
            question=(
                "Divide the image into thirds. In which third does the "
                "{object_1} primarily appear? Respond with the letter only: "
                "A) left third, B) middle third, C) right third."
            ),
            variables=["object_1"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )
        if buffer_ratio < 0 or buffer_ratio > 0.5:
            raise ValueError(
                "Buffer ratio provided does not make sense. Must be between 0 (no buffer) and 0.5 (half the image width)")
        self.buffer_ratio: float = buffer_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True

        # get all the classes that have only one detection
        detection_counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for class_name in class_name:
                    detection_counts[class_name] = (
                        detection_counts.get(class_name, 0) + 1
                    )
            else:
                detection_counts[class_name] = detection_counts.get(
                    class_name, 0) + 1

        single_detections = [
            class_name for class_name, count in detection_counts.items() if count == 1
        ]

        image_width, image_height = image.size

        object_positions = []
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for single_class_name in class_name:
                    if single_class_name in single_detections:
                        object_positions.append(
                            (
                                single_class_name,
                                detection.as_xyxy()[0][0],
                                detection.as_xyxy()[0][2],
                            )
                        )
            else:
                if class_name in single_detections:
                    object_positions.append(
                        (
                            class_name,
                            detection.as_xyxy()[0][0],
                            detection.as_xyxy()[0][2],
                        )
                    )

        question_answer_pairs = []
        for class_name, x_min, x_max in object_positions:
            question = self.question.format(object_1=class_name)

            left_line = image_width / 3
            right_line = 2 * image_width / 3
            buffer = self.buffer_ratio * image_width

            # Discard if bbox is too close to a boundary (ambiguous)
            if (
                abs(x_min - left_line) < buffer
                or abs(x_max - left_line) < buffer
                or abs(x_min - right_line) < buffer
                or abs(x_max - right_line) < buffer
            ):
                logger.debug("IsObjectCentered skipped due to ambiguity buffer")
                continue

            # Determine third based on buffered grid
            if x_max < left_line - buffer:
                answer = "A"
            elif x_min > left_line + buffer and x_max < right_line - buffer:
                answer = "B"
            elif x_min > right_line + buffer:
                answer = "C"
            else:
                # Large object spans multiple thirds – ambiguous
                continue
            question_answer_pairs.append((question, answer))

        return question_answer_pairs


class WidthVsHeight(Question):
    def __init__(
        self,
        threshold: float = 0.75,
        non_articulated_classes: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            question="Is the width of the {object_1} appear to be larger than the height?",
            variables=["object_1"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )
        # ask recall. if object is detected, then ask for unique description
        if len(non_articulated_classes) == 0:
            raise ValueError(
                "non_articulated_classes must be a non-empty list of class names")
        self.non_articulated_classes: list[str] = non_articulated_classes
        self.threshold: float = threshold
        self.other_question: str = "Is the height of the {object_1} larger than the width?"

    def __repr__(self):
        return f"Question: {self.question} (threshold: {self.threshold})"

    def _question_answer(
        self, class_name: str, detection: ObjectDetectionResultI, reverse: bool = False
    ) -> Optional[tuple[str, str]]:
        width = detection.as_xywh().squeeze()[2].item()
        height = detection.as_xywh().squeeze()[3].item()
        # TODO: should we check for a minimum width or height?
        if abs(width - height) / width < self.threshold:
            logger.debug(
                "Width and height are roughly equal (within threshold) so can't ask WidthVsHeight"
            )
            return None

        if width > height:
            answer = "yes"
            other_answer = "no"
        else:
            answer = "no"
            other_answer = "yes"

        if reverse:
            question = self.other_question.format(object_1=class_name)
            answer = other_answer
        else:
            question = self.question.format(object_1=class_name)

        return (question, answer)

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        reverse: bool = False,
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True

        # get all the classes that have only one detection
        detection_counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for single_class_name in class_name:
                    detection_counts[single_class_name] = (
                        detection_counts.get(single_class_name, 0) + 1
                    )
            else:
                detection_counts[class_name] = detection_counts.get(
                    class_name, 0) + 1

        single_detections = [
            class_name for class_name, count in detection_counts.items() if count == 1
        ]

        question_answer_pairs = []
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for single_class_name in class_name:
                    if (
                        single_class_name in single_detections
                        and single_class_name in self.non_articulated_classes
                    ):
                        question_answer_pair = self._question_answer(
                            single_class_name, detection, reverse=reverse
                        )
                        if question_answer_pair is not None:
                            question_answer_pairs.append(question_answer_pair)
            else:
                if (
                    class_name in single_detections
                    and class_name in self.non_articulated_classes
                ):
                    question_answer_pair = self._question_answer(
                        class_name, detection, reverse=reverse
                    )
                    if question_answer_pair is not None:
                        question_answer_pairs.append(question_answer_pair)

        return question_answer_pairs


class Quadrants(Question):
    def __init__(self, N: int, M: int, margin_ratio: float = 0.1) -> None:
        if N <= 0 or M <= 0:
            raise ValueError("N and M must be positive integers")
        if N * M > 12:
            raise ValueError("N * M must be less than or equal to 12")
        if margin_ratio < 0 or margin_ratio > 0.5:
            raise ValueError(
                "Margin ratio must be between 0 (no margin) and 0.5 (half the quadrant width/height)")
        self.rows: int = N
        self.cols: int = M
        self.margin_ratio: float = margin_ratio
        super().__init__(
            question="Divide the image into a {N} x {M} grid. Number the quadrants from left to right, top to bottom, starting with 1. In what quadrant does the {object_1} appear?",
            variables=["object_1", "N", "M"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def _question_answer(
        self, image: Image.Image, class_name: str, detection: ObjectDetectionResultI
    ) -> Optional[tuple[str, str]]:
        x_min, y_min, x_max, y_max = detection.as_xyxy()[0]
        detection_width = x_max - x_min
        detection_height = y_max - y_min

        image_width, image_height = image.size

        quadrant_width = image_width / self.cols
        quadrant_height = image_height / self.rows

        # Margin inside each quadrant that the bbox must fully respect
        margin_x = self.margin_ratio * quadrant_width
        margin_y = self.margin_ratio * quadrant_height

        # Require bbox to fit wholly inside a quadrant with the margin buffer
        if not (
            detection_width < quadrant_width - 2 * margin_x
            and detection_height < quadrant_height - 2 * margin_y
        ):
            return None

        # calculate the quadrant the object is in
        # if it is in multiple quadrants, ignore that object
        row = math.floor(y_min / quadrant_height)
        if row != math.floor(y_max / quadrant_height):
            logger.debug("Object spans multiple rows")
            return None
        col = math.floor(x_min / quadrant_width)
        if col != math.floor(x_max / quadrant_width):
            logger.debug("Object spans multiple columns")
            return None

        # Ensure bbox respects margin inside the identified quadrant
        if not (
            x_min >= col * quadrant_width + margin_x
            and x_max <= (col + 1) * quadrant_width - margin_x
            and y_min >= row * quadrant_height + margin_y
            and y_max <= (row + 1) * quadrant_height - margin_y
        ):
            logger.debug("Quadrants skipped due to margin ambiguity")
            return None

        quadrant = row * self.cols + col + 1

        question = self.question.format(
            object_1=class_name,
            N=self.rows,
            M=self.cols,
        )
        answer = str(quadrant)
        return (question, answer)

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True

        # get all the classes that have only one detection
        detection_counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for single_class_name in class_name:
                    detection_counts[single_class_name] = (
                        detection_counts.get(single_class_name, 0) + 1
                    )
            else:
                detection_counts[class_name] = detection_counts.get(
                    class_name, 0) + 1

        single_detections = [
            class_name for class_name, count in detection_counts.items() if count == 1
        ]

        question_answer_pairs = []
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for single_class_name in class_name:
                    if single_class_name in single_detections:
                        question_answer_pair = self._question_answer(
                            image, single_class_name, detection
                        )
                        if question_answer_pair is not None:
                            question_answer_pairs.append(question_answer_pair)
            else:
                if class_name in single_detections:
                    question_answer_pair = self._question_answer(
                        image, class_name, detection
                    )
                    if question_answer_pair is not None:
                        question_answer_pairs.append(question_answer_pair)

        return question_answer_pairs


class LargestAppearance(Question):
    def __init__(self, threshold: float = 0.3) -> None:
        super().__init__(
            question="Which kind of object appears the largest in the image?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )
        # in the R.O.S. verifier, black out every single box then ask 
        self.threshold = threshold

    def __repr__(self):
        return f"Question: {self.question} (threshold: {self.threshold})"

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 2) == True

        if len(detections) == 0:
            logger.debug("No detections given to LargestAppearance question")
            return []

        # TODO: verify if this works
        # the same logic should apply here regardless of detections being a tensor or not
        areas = [detection.get_area() for detection in detections]
        largest_detection = detections[torch.argmax(torch.stack(areas))]
        second_largest_detection = detections[
            torch.argsort(torch.stack(areas).squeeze())[-2]
        ]

        # check if the largest detection is at least 30% larger than the second largest
        if not (
            largest_detection.get_area().item()
            > (1 + self.threshold) * second_largest_detection.get_area().item()
        ):
            logger.debug(
                f"Largest detection is not at least {self.threshold:.2%} larger than the second largest"
            )
            return []

        question = self.question
        answer = str(largest_detection.label)
        return [(question, answer)]


class RankLargestK(Question):
    """Rank the *k* object classes that have the largest single-instance area.

    Example question (for k=3):

        "Rank the 3 kinds of objects that appear the largest in the image from
        largest to smallest. Provide your answer as a comma-separated list of
        object names only."
    """

    def __init__(self, k: int, margin_ratio: float = 0.3) -> None:
        """Create a RankLargestK question.

        Args:
            k: number of classes to rank.
            margin_ratio: required multiplicative margin between consecutive
                ranked areas. For class *i* to be considered larger than class
                *i+1*, its area must be at least ``(1 + margin_ratio)`` times
                larger. If any consecutive pair fails this criterion, the
                question will be skipped for that image.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if margin_ratio < 0:
            raise ValueError("margin_ratio must be non-negative")

        self.k: int = k
        self.margin_ratio: float = margin_ratio
        super().__init__(
            question=(
                "Rank the {k} kinds of objects that appear the largest (by pixel area) in the "
                "image from largest to smallest. Provide your answer as a "
                "comma-separated list of object names only."
            ),
            variables=["k"],
            predicates=[
                # Need at least k different classes detected
                lambda image, detections, k=k: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, k
                ),
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        if len(detections) == 0:
            logger.debug("No detections for RankLargestK question")
            return []

        # Build max-area per class dictionary
        class_max_area: dict[str, float] = {}
        for detection in detections:
            label = detection.label
            area_val = detection.get_area().item()

            if isinstance(label, torch.Tensor):
                # Iterate through tensor labels (multiple boxes per detection)
                for idx in range(label.shape[0]):
                    cls_name = str(label[idx])
                    area_single = area_val if label.shape[0] == 1 else detection.get_area()[
                        idx].item()
                    class_max_area[cls_name] = max(
                        class_max_area.get(cls_name, 0.0), area_single
                    )
            else:
                cls_name = str(label)
                class_max_area[cls_name] = max(
                    class_max_area.get(cls_name, 0.0), area_val
                )

        if len(class_max_area) < self.k:
            logger.debug("Not enough unique classes for RankLargestK question")
            return []

        # Sort classes by their largest instance area
        sorted_classes = sorted(
            class_max_area.items(), key=lambda item: item[1], reverse=True
        )

        # Verify margin criterion among top-k areas
        top_k = sorted_classes[: self.k]
        for i in range(len(top_k) - 1):
            area_i = top_k[i][1]
            area_next = top_k[i + 1][1]
            if area_i < (1 + self.margin_ratio) * area_next:
                logger.debug(
                    "RankLargestK margin threshold not met between %s and %s", top_k[i][0], top_k[i + 1][0])
                return []

        top_k_labels = [cls for cls, _ in top_k]

        question = self.question.format(k=self.k)
        answer = ", ".join(map(str, top_k_labels))
        return [(question, answer)]


class MostAppearance(Question):
    def __init__(self, margin_ratio: float = 0.2) -> None:
        super().__init__(
            question="What kind of object appears the most frequently in the image?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )
        if margin_ratio < 0 or margin_ratio >= 1:
            raise ValueError(
                "The margin ratio between the classes that appear most frequently must be non-negative and less than 1")
        self.margin_ratio: float = margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 2) == True

        if len(detections) == 0:
            logger.debug("No detections given to MostAppearance question")
            return []

        detections_counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for single_class_name in class_name:
                    detections_counts[single_class_name] = (
                        detections_counts.get(single_class_name, 0) + 1
                    )
            else:
                detections_counts[class_name] = detections_counts.get(
                    class_name, 0) + 1

        sorted_detections = sorted(
            detections_counts.items(), key=lambda x: x[1], reverse=True
        )
        top_count = sorted_detections[0][1]
        second_count = sorted_detections[1][1]

        # Require top_count to be sufficiently greater than second_count
        if top_count < (1 + self.margin_ratio) * second_count:
            logger.debug("MostAppearance margin threshold not met")
            return []

        most_detections = sorted_detections[0][0]

        question = self.question
        answer = str(most_detections)
        return [(question, answer)]


class LeastAppearance(Question):
    def __init__(self, margin_ratio: float = 0.2) -> None:
        super().__init__(
            question="What kind of object appears the least frequently in the image?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )
        if margin_ratio < 0 or margin_ratio >= 1:
            raise ValueError(
                "The margin ratio between the classes that appear least frequently must be non-negative and less than 1")
        self.margin_ratio: float = margin_ratio

    def apply(
        self, image: Image.Image, detections: list[ObjectDetectionResultI]
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 2) == True

        if len(detections) == 0:
            logger.debug("No detections given to LeastAppearance question")
            return []

        detections_counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for single_class_name in class_name:
                    detections_counts[single_class_name] = (
                        detections_counts.get(single_class_name, 0) + 1
                    )
            else:
                detections_counts[class_name] = detections_counts.get(
                    class_name, 0) + 1

        sorted_detections = sorted(
            detections_counts.items(), key=lambda x: x[1])

        lowest_count = sorted_detections[0][1]
        second_lowest_count = sorted_detections[1][1]

        if second_lowest_count < (1 + self.margin_ratio) * lowest_count:
            logger.debug("LeastAppearance margin threshold not met")
            return []

        least_detections = sorted_detections[0][0]

        question = self.question
        answer = str(least_detections)
        return [(question, answer)]


class LeftOf(Question):
    def __init__(self) -> None:
        super().__init__(
            question="Is there at least one {object_1} to the left of any {object_2}?",
            variables=["object_1", "object_2"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
                ObjectDetectionPredicates.exists_non_overlapping_detections,
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 2) == True
        # @precondition: exists_non_overlapping_detections(image, detections) == True

        left_most_detections, right_most_detections, _, _ = self._find_extremes(
            image, detections
        )

        # iterate over the right most detections and check if there is a different class
        # that is to the left and non-overlapping of the instances we found above
        question_answer_pairs = []
        for obj_2_class_name, (_, right_most_bbox) in right_most_detections.items():
            for obj_1_class_name, (_, left_most_bbox) in left_most_detections.items():
                if obj_2_class_name == obj_1_class_name:
                    continue

                # check if the left most detection of obj_1 is to the left
                # of the right most detection of obj_2
                if not (left_most_bbox[2] < right_most_bbox[0]):  # not (x2 < x1)
                    continue

                # and non-overlapping
                x1_inter = max(left_most_bbox[0], right_most_bbox[0])
                x2_inter = min(left_most_bbox[2], right_most_bbox[2])
                y1_inter = max(left_most_bbox[1], right_most_bbox[1])
                y2_inter = min(left_most_bbox[3], right_most_bbox[3])

                inter_width = max(0, x2_inter - x1_inter + 1)
                inter_height = max(0, y2_inter - y1_inter + 1)
                inter_area = inter_width * inter_height

                if inter_area > 0:
                    continue

                question = self.question.format(
                    object_1=obj_1_class_name,
                    object_2=obj_2_class_name,
                )
                answer = "Yes"
                question_answer_pairs.append((question, answer))

        return question_answer_pairs


class RightOf(Question):
    def __init__(self) -> None:
        super().__init__(
            question="Is there at least one {object_1} to the right of any {object_2}?",
            variables=["object_1", "object_2"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
                ObjectDetectionPredicates.exists_non_overlapping_detections,
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 2) == True
        # @precondition: exists_non_overlapping_detections(image, detections) == True

        left_most_detections, right_most_detections, _, _ = self._find_extremes(
            image, detections
        )

        # iterate over the left most detections and check if there is a different class
        # that is to the right and non-overlapping of the instances we found above
        question_answer_pairs = []
        for obj_1_class_name, (_, left_most_bbox) in right_most_detections.items():
            for obj_2_class_name, (_, right_most_bbox) in left_most_detections.items():
                if obj_1_class_name == obj_2_class_name:
                    continue

                # check if the right most detection of obj_1 is to the right
                # of the left most detection of obj_2
                if not (right_most_bbox[2] < left_most_bbox[0]):  # not (x2 < x1)
                    continue

                # and non-overlapping
                x1_inter = max(left_most_bbox[0], right_most_bbox[0])
                x2_inter = min(left_most_bbox[2], right_most_bbox[2])
                y1_inter = max(left_most_bbox[1], right_most_bbox[1])
                y2_inter = min(left_most_bbox[3], right_most_bbox[3])

                inter_width = max(0, x2_inter - x1_inter + 1)
                inter_height = max(0, y2_inter - y1_inter + 1)
                inter_area = inter_width * inter_height

                if inter_area > 0:
                    continue

                question = self.question.format(
                    object_1=obj_1_class_name,
                    object_2=obj_2_class_name,
                )
                answer = "Yes"
                question_answer_pairs.append((question, answer))

        return question_answer_pairs


# One can image an AboveOf and BelowOf question as well
# However, these are actually not a good idea
# When you look at an image, what appears as a higher or lower
# y-coordinate may not necessarily translate to a higher or lower object
# This is especially true of perspective images (i.e. images taken from a distance)
# An object that is further away from the camera may appear at a higher
# y-coordinate than an object that is closer to the camera but they are
# in fact on the same plane


class LeftMost(Question):
    def __init__(self) -> None:
        super().__init__(
            question="What is the leftmost object in the image?",
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True

        # TODO: Asking this question heavily depends on the accuracy of the object detection model.
        # It's possible that the model is not able to detect some objects because it was not trained on them.
        # For example, if the model was trained on COCO, it might not be able to detect objects that
        # are not in the COCO dataset, whereas a model trained on Imagenet-1k might be able to do so.
        #
        # One way to address this, would be to implement set-of-mark prompting (highlight what we can
        # detect via bounding boxes) and then ask the model to answer the question based on that.

        if len(detections) == 0:
            return []

        if len(detections) == 1:
            image_width, _ = image.size
            # logic to check if the bbox is actually on the left side of the image
            if (
                detections[0].as_xyxy()[0][0] < image_width / 2
                and detections[0].as_xyxy()[0][2] < image_width / 2
            ):
                return [(self.question, detections[0].label)]
            else:
                return []

        flattened_detections = []
        for detection in detections:
            curr_bbox = detection.as_xyxy().squeeze(0)
            if type(detection.label) is torch.Tensor:
                for i in range(detection.label.shape[0]):
                    label = detection.label[i]
                    curr_bbox = curr_bbox[i]
                    flattened_detections.append((label, curr_bbox))
            else:
                flattened_detections.append((detection.label, curr_bbox))

        sorted_detections = sorted(
            flattened_detections, key=lambda x: x[1][0]
        )  # sort by x1 coordinate
        leftmost_detection = sorted_detections[0]
        second_leftmost_detection = sorted_detections[1]

        x1_inter = max(leftmost_detection[1][0],
                       second_leftmost_detection[1][0])
        x2_inter = min(leftmost_detection[1][2],
                       second_leftmost_detection[1][2])
        y1_inter = max(leftmost_detection[1][1],
                       second_leftmost_detection[1][1])
        y2_inter = min(leftmost_detection[1][3],
                       second_leftmost_detection[1][3])

        inter_width = max(0, x2_inter - x1_inter + 1)
        inter_height = max(0, y2_inter - y1_inter + 1)
        inter_area = inter_width * inter_height

        if inter_area > 0:  # overlapping
            logger.debug(
                "LeftMost question not ask-able due to overlapping detections")
            return []

        image_width, _ = image.size
        # logic to check if the bbox is actually on the left side of the image
        if not (
            leftmost_detection[1][0] < image_width / 2
            and leftmost_detection[1][2] < image_width / 2
        ):
            logger.debug(
                "LeftMost question not ask-able due to not being on the left side of the image"
            )
            return []

        return [(self.question, leftmost_detection[0])]


class RightMost(Question):
    def __init__(self) -> None:
        super().__init__(
            question="What is the rightmost object in the image?",
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True

        # TODO: Asking this question heavily depends on the accuracy of the object detection model.
        # It's possible that the model is not able to detect some objects because it was not trained on them.
        # For example, if the model was trained on COCO, it might not be able to detect objects that
        # are not in the COCO dataset, whereas a model trained on Imagenet-1k might be able to do so.
        #
        # One way to address this, would be to implement set-of-mark prompting (highlight what we can
        # detect via bounding boxes) and then ask the model to answer the question based on that.

        if len(detections) == 0:
            return []

        if len(detections) == 1:
            image_width, _ = image.size
            # logic to check if the bbox is actually on the right side of the image
            if (
                detections[0].as_xyxy()[0][0] > image_width / 2
                and detections[0].as_xyxy()[0][2] > image_width / 2
            ):
                return [(self.question, detections[0].label)]
            else:
                return []

        flattened_detections = []
        for detection in detections:
            curr_bbox = detection.as_xyxy().squeeze(0)
            if type(detection.label) is torch.Tensor:
                for i in range(detection.label.shape[0]):
                    label = detection.label[i]
                    curr_bbox = curr_bbox[i]
                    flattened_detections.append((label, curr_bbox))
            else:
                flattened_detections.append((detection.label, curr_bbox))

        sorted_detections = sorted(
            flattened_detections, key=lambda x: x[1][2], reverse=True
        )  # sort by x2 coordinate
        rightmost_detection = sorted_detections[0]
        second_rightmost_detection = sorted_detections[1]

        x1_inter = max(rightmost_detection[1]
                       [0], second_rightmost_detection[1][0])
        x2_inter = min(rightmost_detection[1]
                       [2], second_rightmost_detection[1][2])
        y1_inter = max(rightmost_detection[1]
                       [1], second_rightmost_detection[1][1])
        y2_inter = min(rightmost_detection[1]
                       [3], second_rightmost_detection[1][3])

        inter_width = max(0, x2_inter - x1_inter + 1)
        inter_height = max(0, y2_inter - y1_inter + 1)
        inter_area = inter_width * inter_height

        if inter_area > 0:  # overlapping
            logger.debug(
                "RightMost question not ask-able due to overlapping detections"
            )
            return []

        image_width, _ = image.size
        # logic to check if the bbox is actually on the right side of the image
        if not (
            rightmost_detection[1][0] > image_width / 2
            and rightmost_detection[1][2] > image_width / 2
        ):
            logger.debug(
                "RightMost question not ask-able due to not being on the right side of the image"
            )
            return []

        return [(self.question, rightmost_detection[0])]


class HowMany(Question):
    # TODO: Create a version of this question that is multiple choice
    def __init__(self) -> None:
        super().__init__(
            question="How many {object_1}(s) are there in this image?",
            variables=["object_1"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 1
                ),
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 1) == True

        detection_counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for single_class_name in class_name:
                    detection_counts[single_class_name] = (
                        detection_counts.get(single_class_name, 0) + 1
                    )
            else:
                detection_counts[class_name] = detection_counts.get(
                    class_name, 0) + 1

        question_answer_pairs = []
        for class_name, count in detection_counts.items():
            question_answer_pairs.append(
                (self.question.format(object_1=class_name), str(count))
            )

        return question_answer_pairs


class AreMore(Question):
    # TODO: Create a version of this question that is multiple choice
    def __init__(self, margin_ratio: float = 0.2) -> None:
        """AreMore question with margin-based count filtering.
        
        Args:
            margin_ratio: Required margin between counts. Only asks question if
                the larger count exceeds the smaller by at least this ratio.
                E.g., margin_ratio=0.2 means count_1 must be ≥ 1.2 * count_2.
        """
        super().__init__(
            question="Are there more {object_1}(s) than {object_2}(s) in this image?",
            variables=["object_1", "object_2"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )
        if margin_ratio < 0 or margin_ratio > 1:
            raise ValueError("margin_ratio must be between 0 and 1")
        self.margin_ratio = margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:

        detection_counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:
                for single_class_name in class_name:
                    detection_counts[single_class_name] = (
                        detection_counts.get(single_class_name, 0) + 1
                    )
            else:
                detection_counts[class_name] = detection_counts.get(
                    class_name, 0) + 1
        question_answer_pairs = []
        detected_classes = list(detection_counts.keys())

        for i in range(len(detected_classes)):
            for j in range(i + 1, len(detected_classes)):
                object_1, object_2 = detected_classes[i], detected_classes[j]
                count_1, count_2 = (
                    detection_counts[object_1],
                    detection_counts[object_2],
                )

                if count_1 > count_2:
                    # Check if count_1 is significantly greater than count_2
                    if count_1 >= (1 + self.margin_ratio) * count_2:
                        answer = "Yes"
                    else:
                        # Difference not significant enough - skip question
                        continue
                elif count_2 > count_1:
                    # Check if count_2 is significantly greater than count_1
                    if count_2 >= (1 + self.margin_ratio) * count_1:
                        answer = "No"
                    else:
                        # Difference not significant enough - skip question
                        continue
                else:
                    continue

                question_answer_pairs.append(
                    (self.question.format(object_1=object_1, object_2=object_2), answer)
                )

        return question_answer_pairs


class WhichMore(Question):
    def __init__(self, margin_ratio: float = 0.2) -> None:
        """WhichMore question with margin-based count filtering.
        
        Args:
            margin_ratio: Required margin for clear winner. Only asks question if
                the winning count exceeds the second-highest by at least this ratio.
        """
        super().__init__(
            question="What appears the most in this image: {object_1}s, {object_2}s, or {object_3}s?",
            variables=["object_1", "object_2", "objejct_3"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )
        if margin_ratio < 0 or margin_ratio > 1:
            raise ValueError("margin_ratio must be between 0 and 1")
        self.margin_ratio = margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:

        detection_counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:
                for single_class_name in class_name:
                    detection_counts[single_class_name] = (
                        detection_counts.get(single_class_name, 0) + 1
                    )
            else:
                detection_counts[class_name] = detection_counts.get(
                    class_name, 0) + 1
        question_answer_pairs = []
        detected_classes = list(detection_counts.keys())

        for i in range(len(detected_classes)):
            for j in range(i + 1, len(detected_classes)):
                for k in range(j + 1, len(detected_classes)):
                    object_1, object_2, object_3 = (
                        detected_classes[i],
                        detected_classes[j],
                        detected_classes[k],
                    )
                    count_1, count_2, count_3 = (
                        detection_counts[object_1],
                        detection_counts[object_2],
                        detection_counts[object_3],
                    )

                    max_count = max(count_1, count_2, count_3)
                    # Sort counts to find second highest
                    sorted_counts = sorted([count_1, count_2, count_3], reverse=True)
                    second_highest_count = sorted_counts[1]
                    
                    # Check if winner has significant margin over second place
                    if max_count < (1 + self.margin_ratio) * second_highest_count:
                        # Winner not clear enough - skip question
                        continue
                    
                    max_objects = []
                    if count_1 == max_count:
                        max_objects.append(object_1)
                    if count_2 == max_count:
                        max_objects.append(object_2)
                    if count_3 == max_count:
                        max_objects.append(object_3)

                    if len(max_objects) == 1:
                        answer = max_objects[0]
                        question_answer_pairs.append(
                            (
                                self.question.format(
                                    object_1=object_1,
                                    object_2=object_2,
                                    object_3=object_3,
                                ),
                                answer + "s",
                            )
                        )
        return question_answer_pairs


class LeftMostWidthVsHeight(WidthVsHeight):
    def __init__(self, threshold: float = 0.75, spatial_margin_ratio: float = 0.05) -> None:
        """LeftMostWidthVsHeight with spatial stability checks.
        
        Args:
            threshold: Aspect ratio threshold
            spatial_margin_ratio: Required spatial separation as fraction of image width.
                The leftmost object must be separated from the second-leftmost by at least
                this margin to ensure stable positioning.
        """
        super().__init__(threshold=threshold)
        self.question = (
            "Does the leftmost object in the image appear to be wider than it is tall?"
        )
        self.other_question = (
            "Does the leftmost object in the image appear to be taller than it is wide?"
        )
        if spatial_margin_ratio < 0 or spatial_margin_ratio > 1:
            raise ValueError("spatial_margin_ratio must be between 0 and 1")
        self.spatial_margin_ratio = spatial_margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        reverse: bool = False,
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True
        im_width, im_height = image.size

        if len(detections) == 0:
            return []

        flattened_detections = [
            box for detection in detections for box in detection.flatten()
        ]
        detection_counts = {}
        for detection in flattened_detections:
            class_name = detection.label
            detection_counts[class_name] = detection_counts.get(
                class_name, 0) + 1

        single_detections = [
            class_name for class_name, count in detection_counts.items() if count == 1
        ]
        if len(single_detections) == 0:
            return []

        sorted_detections = sorted(
            flattened_detections, key=lambda x: x.as_xyxy()[0][0]
        )  # sort by x1 coordinate
        leftmost_detection = None
        second_leftmost_detection = None
        for i, detection in enumerate(sorted_detections):
            if detection.label in single_detections:
                is_on_left = (
                    detection.as_xyxy()[0][0] < im_width / 2
                    and detection.as_xyxy()[0][2] < im_width / 2
                )
                if not is_on_left:
                    # no point in continuing if the leftmost detection is not on the left side of the image
                    logger.debug(
                        "LeftMostWidthVsHeight question not ask-able due to not being on the left side of the image"
                    )
                    return []
                leftmost_detection = detection
                if i + 1 < len(sorted_detections):
                    second_leftmost_detection = sorted_detections[i + 1]
                break

        if leftmost_detection is None:
            logger.debug("No leftmost detection found")
            return []
        if second_leftmost_detection is not None:
            # Check spatial stability: leftmost object must be clearly separated
            leftmost_x_max = leftmost_detection.as_xyxy()[0][2]
            second_leftmost_x_min = second_leftmost_detection.as_xyxy()[0][0]
            
            # Calculate required spatial margin
            required_margin = self.spatial_margin_ratio * im_width
            actual_gap = second_leftmost_x_min - leftmost_x_max
            
            if actual_gap < required_margin:
                logger.debug(
                    f"LeftMostWidthVsHeight question not ask-able due to insufficient spatial separation: "
                    f"gap={actual_gap:.1f}px < required={required_margin:.1f}px"
                )
                return []
            
            # Additional check: ensure no overlap (legacy check kept for safety)
            x1_inter = max(
                leftmost_detection.as_xyxy()[0][0],
                second_leftmost_detection.as_xyxy()[0][0],
            )
            x2_inter = min(
                leftmost_detection.as_xyxy()[0][2],
                second_leftmost_detection.as_xyxy()[0][2],
            )
            y1_inter = max(
                leftmost_detection.as_xyxy()[0][1],
                second_leftmost_detection.as_xyxy()[0][1],
            )
            y2_inter = min(
                leftmost_detection.as_xyxy()[0][3],
                second_leftmost_detection.as_xyxy()[0][3],
            )
            inter_width = max(0, x2_inter - x1_inter + 1)
            inter_height = max(0, y2_inter - y1_inter + 1)
            inter_area = inter_width * inter_height

            if inter_area > 0:  # overlapping
                logger.debug(
                    "LeftMostWidthVsHeight question not ask-able due to overlapping detections"
                )
                return []

        # check if the leftmost detection is at least threshold % larger than the second largest
        question_answer_pair = self._question_answer(
            leftmost_detection.label,
            leftmost_detection,
            reverse=reverse,
        )
        if question_answer_pair is None:
            logger.debug(
                "LeftMostWidthVsHeight question not ask-able due to width and height being roughly equal"
            )
            return []
        return question_answer_pair


class RightMostWidthVsHeight(WidthVsHeight):
    def __init__(self, threshold: float = 0.75, spatial_margin_ratio: float = 0.05) -> None:
        """RightMostWidthVsHeight with spatial stability checks.
        
        Args:
            threshold: Aspect ratio threshold (inherited from WidthVsHeight)
            spatial_margin_ratio: Required spatial separation as fraction of image width.
                The rightmost object must be separated from the second-rightmost by at least
                this margin to ensure stable positioning.
        """
        super().__init__(threshold=threshold)
        self.question = (
            "Does the rightmost object in the image appear to be wider than it is tall?"
        )
        self.other_question = "Does the rightmost object in the image appear to be taller than it is wide?"
        if spatial_margin_ratio < 0 or spatial_margin_ratio > 1:
            raise ValueError("spatial_margin_ratio must be between 0 and 1")
        self.spatial_margin_ratio = spatial_margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        reverse: bool = False,
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True
        im_width, im_height = image.size

        if len(detections) == 0:
            return []

        flattened_detections = [
            box for detection in detections for box in detection.flatten()
        ]
        detection_counts = {}
        for detection in flattened_detections:
            class_name = detection.label
            detection_counts[class_name] = detection_counts.get(
                class_name, 0) + 1

        single_detections = [
            class_name for class_name, count in detection_counts.items() if count == 1
        ]
        if len(single_detections) == 0:
            return []

        sorted_detections = sorted(
            flattened_detections, key=lambda x: x.as_xyxy()[0][2], reverse=True
        )  # sort by x2 coordinate
        rightmost_detection = None
        second_rightmost_detection = None
        for i, detection in enumerate(sorted_detections):
            if detection.label in single_detections:
                is_on_right = (
                    detection.as_xyxy()[0][0] > im_width / 2
                    and detection.as_xyxy()[0][2] > im_width / 2
                )
                if not is_on_right:
                    # no point in continuing if the rightmost detection is not on the right side of the image
                    logger.debug(
                        "RightMostWidthVsHeight question not ask-able due to not being on the right side of the image"
                    )
                    return []
                rightmost_detection = detection
                if i + 1 < len(sorted_detections):
                    second_rightmost_detection = sorted_detections[i + 1]
                break

        if rightmost_detection is None:
            logger.debug("No rightmost detection found")
            return []

        if second_rightmost_detection is not None:
            # Check spatial stability: rightmost object must be clearly separated
            rightmost_x_min = rightmost_detection.as_xyxy()[0][0]
            second_rightmost_x_max = second_rightmost_detection.as_xyxy()[0][2]
            
            # Calculate required spatial margin
            required_margin = self.spatial_margin_ratio * im_width
            actual_gap = rightmost_x_min - second_rightmost_x_max
            
            if actual_gap < required_margin:
                logger.debug(
                    f"RightMostWidthVsHeight question not ask-able due to insufficient spatial separation: "
                    f"gap={actual_gap:.1f}px < required={required_margin:.1f}px"
                )
                return []
            
            # Additional check: ensure no overlap (legacy check kept for safety)
            x1_inter = max(
                rightmost_detection.as_xyxy()[0][0],
                second_rightmost_detection.as_xyxy()[0][0],
            )
            x2_inter = min(
                rightmost_detection.as_xyxy()[0][2],
                second_rightmost_detection.as_xyxy()[0][2],
            )
            y1_inter = max(
                rightmost_detection.as_xyxy()[0][1],
                second_rightmost_detection.as_xyxy()[0][1],
            )
            y2_inter = min(
                rightmost_detection.as_xyxy()[0][3],
                second_rightmost_detection.as_xyxy()[0][3],
            )
            inter_width = max(0, x2_inter - x1_inter + 1)
            inter_height = max(0, y2_inter - y1_inter + 1)
            inter_area = inter_width * inter_height

            if inter_area > 0:  # overlapping
                logger.debug(
                    "RightMostWidthVsHeight question not ask-able due to overlapping detections"
                )
                return []
        # check if the rightmost detection is at least threshold % larger than the second largest
        question_answer_pair = self._question_answer(
            rightmost_detection.label,
            rightmost_detection,
            reverse=reverse,
        )
        if question_answer_pair is None:
            logger.debug(
                "RightMostWidthVsHeight question not ask-able due to width and height being roughly equal"
            )
            return []
        return question_answer_pair


# drop this question
class ObjectsInRow(Question):
    def __init__(self, variance_threshold: float = 0.1) -> None:
        """Linear regression-based row detection.

        Args:
            variance_threshold: Maximum normalized variance for y-centers to be 
                considered in a row. Lower values = stricter row detection.
        """
        super().__init__(
            question="Are there any objects arranged in a row?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_detections(
                    image, detections, 3
                ),
            ],
        )
        self.variance_threshold = variance_threshold

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        from sklearn.linear_model import LinearRegression

        if len(detections) < 3:
            return [(self.question, "No")]

        # Get center points
        centers = []
        for detection in detections:
            bbox = detection.as_xyxy().squeeze(0)
            x_center = float((bbox[0] + bbox[2]) / 2)
            y_center = float((bbox[1] + bbox[3]) / 2)
            centers.append((x_center, y_center))

        # Sort by x-coordinate
        centers = sorted(centers, key=lambda p: p[0])

        # Try sliding windows of 3+ objects
        image_height = image.size[1]

        for window_size in range(3, len(centers) + 1):
            for start in range(len(centers) - window_size + 1):
                window = centers[start:start + window_size]

                # Extract x and y coordinates
                x_coords = np.array([p[0] for p in window]).reshape(-1, 1)
                y_coords = np.array([p[1] for p in window])

                # Fit linear regression
                reg = LinearRegression().fit(x_coords, y_coords)
                y_pred = reg.predict(x_coords)

                # Calculate normalized variance (by image height)
                variance = np.var(y_coords - y_pred)
                normalized_variance = variance / (image_height ** 2)

                if normalized_variance < self.variance_threshold:
                    return [(self.question, "Yes")]

        return [(self.question, "No")]


class ObjectsInLine(Question):
    def __init__(self, variance_threshold: float = 0.1) -> None:
        """Multiple choice question about which objects are in a row.

        Args:
            variance_threshold: Same as ObjectsInRow for consistency.
        """
        super().__init__(
            question="Which objects appear to be arranged in a row? A) {option_a}, B) {option_b}, C) {option_c}, D) No clear row arrangement. Respond with the letter only.",
            variables=["option_a", "option_b", "option_c"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_detections(
                    image, detections, 3
                ),
            ],
        )
        self.variance_threshold = variance_threshold

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        from sklearn.linear_model import LinearRegression

        if len(detections) < 3:
            return []

        # Get centers with labels
        centers_with_labels = []
        for detection in detections:
            bbox = detection.as_xyxy().squeeze(0)
            x_center = float((bbox[0] + bbox[2]) / 2)
            y_center = float((bbox[1] + bbox[3]) / 2)
            label = str(detection.label)
            centers_with_labels.append((x_center, y_center, label))

        # Sort by x-coordinate
        centers_with_labels = sorted(centers_with_labels, key=lambda p: p[0])

        # Find best row arrangement
        best_row = None
        best_variance = float('inf')
        image_height = image.size[1]

        for window_size in range(3, len(centers_with_labels) + 1):
            for start in range(len(centers_with_labels) - window_size + 1):
                window = centers_with_labels[start:start + window_size]

                x_coords = np.array([p[0] for p in window]).reshape(-1, 1)
                y_coords = np.array([p[1] for p in window])

                reg = LinearRegression().fit(x_coords, y_coords)
                y_pred = reg.predict(x_coords)

                variance = np.var(y_coords - y_pred)
                normalized_variance = variance / (image_height ** 2)

                if normalized_variance < self.variance_threshold and normalized_variance < best_variance:
                    best_variance = normalized_variance
                    best_row = [p[2] for p in window]  # Extract labels

        if best_row is None:
            return []

        # Create multiple choice options
        row_text = ", ".join(best_row)

        # Generate plausible distractors that are different from correct answer
        all_labels = list(set(str(d.label) for d in detections))
        random.shuffle(all_labels)

        # Create distractors ensuring they're different from correct answer
        distractor1 = ", ".join(all_labels[:min(3, len(all_labels))])
        distractor2 = ", ".join(all_labels[-min(3, len(all_labels)):])

        # Ensure distractors are different from correct answer
        max_attempts = 10
        attempt = 0
        while (distractor1 == row_text or distractor2 == row_text or distractor1 == distractor2) and attempt < max_attempts:
            random.shuffle(all_labels)
            distractor1 = ", ".join(all_labels[:min(3, len(all_labels))])
            # Use different slice
            distractor2 = ", ".join(all_labels[-min(2, len(all_labels)):])
            attempt += 1

        # If still duplicates after attempts, skip this question
        if distractor1 == row_text or distractor2 == row_text or distractor1 == distractor2:
            return []

        # Randomly assign correct answer to A/B/C
        options = [row_text, distractor1, distractor2]
        random.shuffle(options)
        correct_letter = ["A", "B", "C"][options.index(row_text)]

        q = self.question.format(
            option_a=options[0],
            option_b=options[1],
            option_c=options[2]
        )

        return [(q, correct_letter)]


# drop this question
class MostClusteredObjects(Question):
    def __init__(self, eps_ratio: float = 0.05, min_samples: int = 3) -> None:
        """DBSCAN-based clustering with multiple choice answers.

        Args:
            eps_ratio: Maximum distance between points in a cluster as a fraction 
                of the image diagonal. Default 0.05 means 5% of image diagonal.
            min_samples: Minimum points required to form a cluster.
        """
        super().__init__(
            question="Which group of objects appears most tightly clustered? A) {option_a}, B) {option_b}, C) {option_c}, D) No clear clusters. Respond with the letter only.",
            variables=["option_a", "option_b", "option_c"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_detections(
                    image, detections, 9  # Need at least 3 clusters × 3 objects each
                ),
            ],
        )
        self.eps_ratio = eps_ratio
        self.min_samples = min_samples

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        from sklearn.cluster import DBSCAN

        if len(detections) < 9:
            return []

        # Get centers and labels
        centers = []
        labels = []
        for detection in detections:
            bbox = detection.as_xyxy().squeeze(0)
            x_center = float((bbox[0] + bbox[2]) / 2)
            y_center = float((bbox[1] + bbox[3]) / 2)
            centers.append([x_center, y_center])
            labels.append(str(detection.label))

        centers = np.array(centers)

        # Calculate eps as a fraction of image diagonal
        image_width, image_height = image.size
        image_diagonal = math.sqrt(image_width**2 + image_height**2)
        eps = self.eps_ratio * image_diagonal

        # Apply DBSCAN
        clustering = DBSCAN(
            eps=eps, min_samples=self.min_samples).fit(centers)
        cluster_labels = clustering.labels_

        # Group objects by cluster (ignore noise points with label -1)
        clusters = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id != -1:  # Not noise
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(labels[i])

        if len(clusters) < 2:
            return []  # Need at least 2 clusters to compare

        # Find most compact cluster
        def cluster_compactness(cluster_id):
            cluster_points = centers[cluster_labels == cluster_id]
            if len(cluster_points) < 2:
                return float('inf')
            return np.mean(np.var(cluster_points, axis=0))

        most_compact_id = min(clusters.keys(), key=cluster_compactness)
        most_compact_objects = list(
            set(clusters[most_compact_id]))  # Remove duplicates

        # Create multiple choice options
        correct_text = ", ".join(sorted(most_compact_objects))

        # Generate distractors from other clusters or random combinations
        all_unique_labels = list(set(labels))
        random.shuffle(all_unique_labels)

        # Create distractors ensuring they're different from correct answer
        distractor1 = ", ".join(
            all_unique_labels[:min(3, len(all_unique_labels))])
        distractor2 = ", ".join(
            all_unique_labels[-min(2, len(all_unique_labels)):])

        # Ensure distractors are different from correct answer
        max_attempts = 10
        attempt = 0
        while (distractor1 == correct_text or distractor2 == correct_text or distractor1 == distractor2) and attempt < max_attempts:
            random.shuffle(all_unique_labels)
            distractor1 = ", ".join(
                all_unique_labels[:min(3, len(all_unique_labels))])
            distractor2 = ", ".join(
                all_unique_labels[-min(2, len(all_unique_labels)):])
            attempt += 1

        # If still duplicates after attempts, skip this question
        if distractor1 == correct_text or distractor2 == correct_text or distractor1 == distractor2:
            return []

        # Randomly assign correct answer
        options = [correct_text, distractor1, distractor2]
        random.shuffle(options)
        correct_letter = ["A", "B", "C"][options.index(correct_text)]

        q = self.question.format(
            option_a=options[0],
            option_b=options[1],
            option_c=options[2]
        )

        return [(q, correct_letter)]


class MoreThanThresholdHowMany(Question):
    """More-than count question with built-in Yes/No balance.

    For each detected object class with count *N* we generate two prompts:

    1. *Yes case*   – target = ⌊N / threshold⌋.
       The detector's count is safely above the target, so the correct answer is **Yes**.

    2. *No case*    – target = ⌈N × threshold⌉.
       The detector's count is well below the target, so the correct answer is **No**.

    The gap created by the multiplicative buffer acts as a hedge against recall / precision noise
    while keeping the overall Yes/No distribution roughly balanced.
    """

    def __init__(self, threshold: float = 2.0):
        if threshold <= 1.0:
            raise ValueError(
                "threshold should be > 1.0 for 'more than' questions")

        self.threshold: float = threshold
        super().__init__(
            question="Are there more than {target} {object_1}(s) in this image? Respond Yes/No.",
            variables=["object_1", "target"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 1
                ),
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:

        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    counts[str(l)] = counts.get(str(l), 0) + 1
            else:
                counts[str(lbl)] = counts.get(str(lbl), 0) + 1

        qa_pairs: list[tuple[str, str]] = []
        for cls, n in counts.items():
            if n == 0:
                continue

            # Question that should be answered "Yes" (target below n)
            target_yes = max(1, math.floor(n / self.threshold))
            if target_yes == n:
                target_yes = max(1, target_yes - 1)

            q_yes = self.question.format(object_1=cls, target=target_yes)
            qa_pairs.append((q_yes, "Yes"))

            # Question that should be answered "No" (target well above n)
            target_no = math.ceil(n * self.threshold)
            if target_no == n:
                target_no += 1

            q_no = self.question.format(object_1=cls, target=target_no)
            qa_pairs.append((q_no, "No"))

        return qa_pairs


class LessThanThresholdHowMany(Question):
    """Less-than count question with symmetric Yes/No balance.

    For detected count *N* we generate:

    1. *Yes case* – target = ⌈N / threshold⌉ (> N), so the answer **Yes** is correct.
    2. *No case*  – target = ⌊N × threshold⌋ (< N), so **No** is correct.

    This mirrors the more-than version and maintains balanced answer keys while
    providing a tolerance band for detector errors.
    """

    def __init__(self, threshold: float = 0.5):
        if not (0.0 < threshold < 1.0):
            raise ValueError(
                "threshold must be between 0 and 1 for 'less than'")

        self.threshold: float = threshold
        super().__init__(
            question="Are there less than {target} {object_1}(s) in this image? Respond Yes/No.",
            variables=["object_1", "target"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 1
                ),
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:

        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    counts[str(l)] = counts.get(str(l), 0) + 1
            else:
                counts[str(lbl)] = counts.get(str(lbl), 0) + 1

        qa_pairs: list[tuple[str, str]] = []
        for cls, n in counts.items():
            if n == 0:
                continue

            # Question that should be answered "Yes" (target above n)
            target_yes = math.ceil(n / self.threshold)
            if target_yes == n:
                target_yes += 1

            q_yes = self.question.format(object_1=cls, target=target_yes)
            qa_pairs.append((q_yes, "Yes"))

            # Question that should be answered "No" (target well below n)
            target_no = max(1, math.floor(n * self.threshold))
            if target_no == n:
                target_no = max(1, target_no - 1)

            q_no = self.question.format(object_1=cls, target=target_no)
            qa_pairs.append((q_no, "No"))

        return qa_pairs


class MultiChoiceHowMany(Question):
    """Noise-tolerant *How Many* as a 3-way multiple-choice question.

    Workflow per detected object class with count *N*:

    1.  Build **contiguous** numeric buckets based on *N* (and confidence variance):
        • *low*  :   `0 – ⌊α · N⌋`
        • *mid*  :   `⌈α · N⌉ – ⌊β · N⌋`
        • *high* :   `⌈β · N⌉ – ⌈β · N⌋+w`  (finite width so all three look alike)
       where `(α, β) = (0.5, 1.5)` by default or `(0.4, 1.8)` when per-class
       confidence variance > 0.05, and *w* equals the width of the mid bucket.

    2.  Randomly **shuffle** which bucket is labelled A, B, or C.  This removes
        the positional/letter bias while the LLM still sees all ranges.

    3.  The correct answer letter is determined after the shuffle so that the
        dataset remains balanced across A/B/C over time.

    4.  A fourth option **D) Unsure / Not Visible** is always listed to allow a
        graceful fallback when the model feels uncertain.

    Questions are only generated when `N ≥ 4`; for very small counts, the
    buckets become too narrow to be useful.
    """

    def __init__(self):
        super().__init__(
            question="How many {object_1}(s) are in the image? Choose one: "
            "A) {range_a}, B) {range_b}, C) {range_c}, D) Unsure / Not Visible. "
            "Respond with the letter only.",
            variables=["object_1", "range_a", "range_b", "range_c"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 1
                ),
            ],
        )

    def _bucket_ranges(self, n: int, var: float) -> tuple[dict[str, str], str]:
        """Return bucket description dict and the *semantic* correct bucket key.

        Keys: "low", "mid", "high" → string description "x–y" (inclusive).
        Also returns which *bucket key* contains ``n`` so we can map it to the
        shuffled letter later.
        """

        # Variance-based adjustment of coefficients
        low_coef, mid_high_coef = (0.4, 1.8) if var > 0.05 else (0.5, 1.5)

        # Bucket boundaries (inclusive)
        low_max = max(0, int((low_coef * n) - 1e-6))
        mid_min = low_max + 1
        mid_max = int(mid_high_coef * n)
        high_min = mid_max + 1

        # Make the high bucket a finite *range* with similar width to mid bucket
        mid_width = mid_max - mid_min
        high_max = high_min + max(2, mid_width)  # ensure non-zero width

        buckets: dict[str, str] = {
            "low": f"0-{low_max}" if low_max > 0 else "0-{mid_min-1}",
            "mid": f"{mid_min}-{mid_max}",
            "high": f"{high_min}-{high_max}",
        }

        # With fixed α/β the detected count N always lands in the mid bucket,
        # so we can simply hard-code it instead of checking.
        correct_bucket = "mid"

        return buckets, correct_bucket

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:

        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    counts[str(l)] = counts.get(str(l), 0) + 1
            else:
                counts[str(lbl)] = counts.get(str(lbl), 0) + 1

        qa_pairs: list[tuple[str, str]] = []
        for cls, n in counts.items():
            if n < 4:
                continue
            # extract per-detection confidences for this class
            scores: list[float] = []
            for det in detections:
                lbl = det.label
                conf = getattr(det, "score", getattr(det, "confidence", 1.0))
                if isinstance(lbl, torch.Tensor):
                    for idx in range(lbl.shape[0]):
                        if str(lbl[idx]) == cls:
                            scores.append(float(conf[idx]) if isinstance(
                                conf, torch.Tensor) else float(conf))
                else:
                    if str(lbl) == cls:
                        scores.append(float(conf))

            var = float(np.var(scores)) if len(scores) > 1 else 0.0

            buckets, correct_bucket = self._bucket_ranges(n, var)

            # Randomly permute letter → bucket mapping to avoid letter bias
            letters = ["A", "B", "C"]
            random.shuffle(letters)
            bucket_keys = ["low", "mid", "high"]

            letter_to_bucket = {letter: bucket for letter,
                                bucket in zip(letters, bucket_keys)}

            # Build question text in A/B/C order after permutation
            q = self.question.format(
                object_1=cls,
                range_a=buckets[letter_to_bucket["A"].lower()],
                range_b=buckets[letter_to_bucket["B"].lower()],
                range_c=buckets[letter_to_bucket["C"].lower()],
            )

            # Identify the letter assigned to the mid bucket (the correct answer)
            correct_letter = {bkey: ltr for ltr, bkey in letter_to_bucket.items()}[
                "mid"]

            qa_pairs.append((q, correct_letter))

        return qa_pairs


ALL_QUESTIONS = [
    IsObjectCentered(),
    WidthVsHeight(),
    LargestAppearance(),
    MostAppearance(),
    LeastAppearance(),
    LeftOf(),
    RightOf(),
    LeftMost(),
    RightMost(),
    HowMany(),
    MostClusteredObjects(),
    WhichMore(),
    AreMore(),
    Quadrants(2, 2),
    Quadrants(2, 3),
    Quadrants(3, 2),
    Quadrants(3, 3),
    LeftMostWidthVsHeight(),
    RightMostWidthVsHeight(),
]
