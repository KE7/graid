import logging
import math
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any

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
        self, question: str, variables: list[str], predicates: list[Callable[..., bool]]
    ) -> None:
        self.question = question
        self.variables = variables
        self.predicates = predicates
        self.other_question: Optional[str] = None

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

    # New optional hook for questions that can benefit from a shared cache of expensive
    # per-image objects (e.g. depth map, SAM predictor).  By default this simply calls
    # the original `apply` implementation so subclasses are not required to override it.
    # Sub-classes that need the cache should implement their own version.
    def apply_with_cache(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        cache: dict[str, Any],
    ) -> list[tuple[str, str]]:  # noqa: D401 – simple wrapper
        return self.apply(image, detections)


class ObjectDetectionPredicates:
    @staticmethod
    def at_least_one_single_detection(
        image: Image.Image, detections: list[ObjectDetectionResultI]
    ) -> bool:
        if len(detections) <= 1:
            return len(detections) == 1
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1
        return any(c == 1 for c in counts.values())

    @staticmethod
    def at_least_x_many_class_detections(
        image: Image.Image, detections: list[ObjectDetectionResultI], x: int
    ) -> bool:
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1
        return len(counts) >= x

    @staticmethod
    def at_least_x_detections(
        image: Image.Image, detections: list[ObjectDetectionResultI], x: int
    ) -> bool:
        return len(detections) >= x

    @staticmethod
    def exists_non_overlapping_detections(
        image: Image.Image, detections: list[ObjectDetectionResultI]
    ) -> bool:
        for i, d1 in enumerate(detections):
            for j in range(i + 1, len(detections)):
                d2 = detections[j]
                if str(d1.label) != str(d2.label):
                    iou = ObjectDetectionUtils.pairwise_iou(d1, d2)
                    if iou.max() == 0:
                        return True
        return False

    @staticmethod
    def has_clusters(
        image: Image.Image, detections: list[ObjectDetectionResultI], threshold=50
    ) -> bool:
        import numpy as np

        if len(detections) < 2:
            return False

        # Compute centers from detections
        centers = []
        for det in detections:
            bbox = det.as_xyxy()[0]  # Get first bbox
            center_x = float((bbox[0] + bbox[2]) / 2.0)
            center_y = float((bbox[1] + bbox[3]) / 2.0)
            centers.append([center_x, center_y])

        centers = np.array(centers)
        # Simple O(n^2) proximity check; no heavy scipy
        n = centers.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                if (dx * dx + dy * dy) ** 0.5 < threshold:
                    return True
        return False


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
                "Buffer ratio provided does not make sense. Must be between 0 (no buffer) and 0.5 (half the image width)"
            )
        self.buffer_ratio: float = buffer_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_one_single_detection(image, detections) == True
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        # classes with single instance
        single_classes = {k for k, v in counts.items() if v == 1}

        image_width, _ = image.size

        question_answer_pairs = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if key not in single_classes:
                        continue
                    bbox = det.as_xyxy()[i]
                    x_min, x_max = float(bbox[0]), float(bbox[2])
                    self._process_single_detection(
                        key, x_min, x_max, image_width, question_answer_pairs
                    )
            else:
                key = str(lbl)
                if key not in single_classes:
                    continue
                bbox = det.as_xyxy()[0]
                x_min, x_max = float(bbox[0]), float(bbox[2])
                self._process_single_detection(
                    key, x_min, x_max, image_width, question_answer_pairs
                )

        return question_answer_pairs

    def _process_single_detection(
        self,
        class_name: str,
        x_min: float,
        x_max: float,
        image_width: int,
        question_answer_pairs: list,
    ):
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
            return

        # Determine third based on buffered grid
        if x_max < left_line - buffer:
            answer = "A"
        elif x_min > left_line + buffer and x_max < right_line - buffer:
            answer = "B"
        elif x_min > right_line + buffer:
            answer = "C"
        else:
            # Large object spans multiple thirds – ambiguous
            return
        question_answer_pairs.append((question, answer))


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
        if non_articulated_classes is not None and len(non_articulated_classes) == 0:
            raise ValueError(
                "non_articulated_classes must be a non-empty list of class names"
            )
        self.non_articulated_classes: Optional[list[str]] = non_articulated_classes
        self.threshold: float = threshold
        self.other_question: Optional[str] = (
            "Is the height of the {object_1} larger than the width?"
        )

    def __repr__(self):
        return f"Question: {self.question} (threshold: {self.threshold})"

    def _question_answer_ratio(
        self, class_name: str, ratio_wh: float, reverse: bool = False
    ) -> Optional[tuple[str, str]]:
        # Skip if near-square within threshold band
        if abs(ratio_wh - 1.0) < self.threshold:
            return None
        answer = "Yes" if ratio_wh > 1.0 else "No"
        if reverse:
            if self.other_question is not None:
                question = self.other_question.format(object_1=class_name)
                answer = "No" if answer == "Yes" else "Yes"
            else:
                return None
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
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        qa: list[tuple[str, str]] = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if counts.get(key, 0) != 1:
                        continue
                    if (
                        self.non_articulated_classes is not None
                        and key not in self.non_articulated_classes
                    ):
                        continue
                    bbox = det.as_xyxy()[i]
                    w = float(bbox[2] - bbox[0])
                    h = float(bbox[3] - bbox[1])
                    ratio = w / max(h, 1e-6)
                    qa_pair = self._question_answer_ratio(key, ratio, reverse=reverse)
                    if qa_pair is not None:
                        qa.append(qa_pair)
            else:
                key = str(lbl)
                if counts.get(key, 0) != 1:
                    continue
                if (
                    self.non_articulated_classes is not None
                    and key not in self.non_articulated_classes
                ):
                    continue
                bbox = det.as_xyxy()[0]
                w = float(bbox[2] - bbox[0])
                h = float(bbox[3] - bbox[1])
                ratio = w / max(h, 1e-6)
                qa_pair = self._question_answer_ratio(key, ratio, reverse=reverse)
                if qa_pair is not None:
                    qa.append(qa_pair)
        return qa


class Quadrants(Question):
    def __init__(self, N: int, M: int, margin_ratio: float = 0.1) -> None:
        if N <= 0 or M <= 0:
            raise ValueError("N and M must be positive integers")
        if N * M > 12:
            raise ValueError("N * M must be less than or equal to 12")
        if margin_ratio < 0 or margin_ratio > 0.5:
            raise ValueError(
                "Margin ratio must be between 0 (no margin) and 0.5 (half the quadrant width/height)"
            )
        self.rows: int = N
        self.cols: int = M
        self.margin_ratio: float = margin_ratio
        super().__init__(
            question="Divide the image into a grid of {N} rows x {M} columns. Number the cells from left to right, then top to bottom, starting with 1. In what cell does the {object_1} appear?",
            variables=["object_1", "N", "M"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def _question_answer(
        self, image: Image.Image, class_name: str, bbox: torch.Tensor
    ) -> Optional[tuple[str, str]]:
        x_min, y_min, x_max, y_max = bbox
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
        row = math.floor(float(y_min) / quadrant_height)
        if row != math.floor(float(y_max) / quadrant_height):
            logger.debug("Object spans multiple rows")
            return None
        col = math.floor(float(x_min) / quadrant_width)
        if col != math.floor(float(x_max) / quadrant_width):
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
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        single_classes = {k for k, v in counts.items() if v == 1}

        question_answer_pairs = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if key not in single_classes:
                        continue
                    bbox = det.as_xyxy()[i]
                    qa = self._question_answer(image, key, bbox)
                    if qa is not None:
                        question_answer_pairs.append(qa)
            else:
                key = str(lbl)
                if key not in single_classes:
                    continue
                bbox = det.as_xyxy()[0]
                qa = self._question_answer(image, key, bbox)
                if qa is not None:
                    question_answer_pairs.append(qa)

        return question_answer_pairs


class LargestAppearance(Question):
    def __init__(self, threshold: float = 0.3) -> None:
        super().__init__(
            question="If you were to draw a tight box around each object in the image, which type of object would have the biggest box?",
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
        # Calculate areas for all detections
        areas = []
        labels = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    bbox = det.as_xyxy()[i]
                    area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    areas.append(area)
                    labels.append(str(l))
            else:
                bbox = det.as_xyxy()[0]
                area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                areas.append(area)
                labels.append(str(lbl))

        if len(areas) == 0:
            return []

        # Sort by area descending
        area_label_pairs = list(zip(areas, labels))
        area_label_pairs.sort(key=lambda x: x[0], reverse=True)

        if len(area_label_pairs) < 2:
            return []

        largest_area, largest_label = area_label_pairs[0]
        second_area, _ = area_label_pairs[1]

        if largest_area <= (1 + self.threshold) * second_area:
            return []

        return [(self.question, largest_label)]


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
        # Calculate max area per class
        class_max_area: dict[str, float] = {}

        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    bbox = det.as_xyxy()[i]
                    area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    class_max_area[key] = max(class_max_area.get(key, 0), area)
            else:
                key = str(lbl)
                bbox = det.as_xyxy()[0]
                area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                class_max_area[key] = max(class_max_area.get(key, 0), area)

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
                    "RankLargestK margin threshold not met between %s and %s",
                    top_k[i][0],
                    top_k[i + 1][0],
                )
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
                "The margin ratio between the classes that appear most frequently must be non-negative and less than 1"
            )
        self.margin_ratio: float = margin_ratio

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 2) == True
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        if len(counts) < 2:
            return []
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_count = sorted_counts[0][1]
        second_count = sorted_counts[1][1]
        if top_count < (1 + self.margin_ratio) * second_count:
            return []
        most = sorted_counts[0][0]
        return [(self.question, str(most))]


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
                "The margin ratio between the classes that appear least frequently must be non-negative and less than 1"
            )
        self.margin_ratio: float = margin_ratio

    def apply(
        self, image: Image.Image, detections: list[ObjectDetectionResultI]
    ) -> list[tuple[str, str]]:
        # @precondition: at_least_x_many_class_detections(image, detections, 2) == True
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        if len(counts) < 2:
            return []
        sorted_counts = sorted(counts.items(), key=lambda x: x[1])
        lowest = sorted_counts[0][1]
        second_lowest = sorted_counts[1][1]
        if second_lowest < (1 + self.margin_ratio) * lowest:
            return []
        least = sorted_counts[0][0]
        return [(self.question, str(least))]


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

        # Group detections by class
        class_detections: dict[str, list[tuple[ObjectDetectionResultI, int]]] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if key not in class_detections:
                        class_detections[key] = []
                    class_detections[key].append((det, i))
            else:
                key = str(lbl)
                if key not in class_detections:
                    class_detections[key] = []
                class_detections[key].append((det, 0))

        qa = []
        classes = list(class_detections.keys())
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i == j:
                    continue
                c1, c2 = classes[i], classes[j]
                found_yes = False

                for det1, idx1 in class_detections[c1]:
                    bbox1 = det1.as_xyxy()[idx1]
                    x2_1 = float(bbox1[2])  # right edge of object 1

                    for det2, idx2 in class_detections[c2]:
                        bbox2 = det2.as_xyxy()[idx2]
                        x1_2 = float(bbox2[0])  # left edge of object 2

                        if x2_1 < x1_2:  # object 1 is to the left of object 2
                            # Check non-overlap via IOU
                            # Create single detection objects for IOU check
                            single_det1 = ObjectDetectionResultI(
                                score=det1.score,
                                cls=det1.cls,
                                label=c1,
                                bbox=bbox1.unsqueeze(0),
                                image_hw=(image.height, image.width),
                            )
                            single_det2 = ObjectDetectionResultI(
                                score=det2.score,
                                cls=det2.cls,
                                label=c2,
                                bbox=bbox2.unsqueeze(0),
                                image_hw=(image.height, image.width),
                            )
                            if (
                                ObjectDetectionUtils.pairwise_iou(
                                    single_det1, single_det2
                                ).max()
                                == 0
                            ):
                                qa.append(
                                    (
                                        self.question.format(object_1=c1, object_2=c2),
                                        "Yes",
                                    )
                                )
                                found_yes = True
                                break
                    if found_yes:
                        break

                if not found_yes:
                    qa.append((self.question.format(object_1=c1, object_2=c2), "No"))
        return qa


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

        # Group detections by class
        class_detections: dict[str, list[tuple[ObjectDetectionResultI, int]]] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if key not in class_detections:
                        class_detections[key] = []
                    class_detections[key].append((det, i))
            else:
                key = str(lbl)
                if key not in class_detections:
                    class_detections[key] = []
                class_detections[key].append((det, 0))

        qa = []
        classes = list(class_detections.keys())
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i == j:
                    continue
                c1, c2 = classes[i], classes[j]
                found_yes = False

                for det1, idx1 in class_detections[c1]:
                    bbox1 = det1.as_xyxy()[idx1]
                    x1_1 = float(bbox1[0])  # left edge of object 1

                    for det2, idx2 in class_detections[c2]:
                        bbox2 = det2.as_xyxy()[idx2]
                        x2_2 = float(bbox2[2])  # right edge of object 2

                        if x1_1 > x2_2:  # object 1 is to the right of object 2
                            # Check non-overlap via IOU
                            # Create single detection objects for IOU check
                            single_det1 = ObjectDetectionResultI(
                                score=det1.score,
                                cls=det1.cls,
                                label=c1,
                                bbox=bbox1.unsqueeze(0),
                                image_hw=(image.height, image.width),
                            )
                            single_det2 = ObjectDetectionResultI(
                                score=det2.score,
                                cls=det2.cls,
                                label=c2,
                                bbox=bbox2.unsqueeze(0),
                                image_hw=(image.height, image.width),
                            )
                            if (
                                ObjectDetectionUtils.pairwise_iou(
                                    single_det1, single_det2
                                ).max()
                                == 0
                            ):
                                qa.append(
                                    (
                                        self.question.format(object_1=c1, object_2=c2),
                                        "Yes",
                                    )
                                )
                                found_yes = True
                                break
                    if found_yes:
                        break

                if not found_yes:
                    qa.append((self.question.format(object_1=c1, object_2=c2), "No"))
        return qa


# One can image an AboveOf and BelowOf question as well
# However, these are actually not a good idea
# When you look at an image, what appears as a higher or lower
# y-coordinate may not necessarily translate to a higher or lower object
# This is especially true of perspective images (i.e. images taken from a distance)
# An object that is further away from the camera may appear at a higher
# y-coordinate than an object that is closer to the camera but they are
# in fact on the same plane


class LeftMost(Question):
    def __init__(self, margin_ratio: float = 0.05) -> None:
        super().__init__(
            question="What is the leftmost object in the image?",
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
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
        # @precondition: at_least_one_single_detection(image, detections) == True
        # Create list of all individual detections with their positions
        all_detections = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    bbox = det.as_xyxy()[i]
                    all_detections.append((bbox, str(l), det, i))
            else:
                bbox = det.as_xyxy()[0]
                all_detections.append((bbox, str(lbl), det, 0))

        if len(all_detections) == 0:
            return []

        # Sort by left edge (x1)
        all_detections.sort(key=lambda x: float(x[0][0]))

        if len(all_detections) < 2:
            # Single detection case: ensure it's on the left half fully
            bbox, label, _, _ = all_detections[0]
            x1, x2 = float(bbox[0]), float(bbox[2])
            if x1 < image.size[0] / 2 and x2 < image.size[0] / 2:
                return [(self.question, label)]
            return []

        # Check overlap between first two leftmost
        bbox1, label1, det1, idx1 = all_detections[0]
        bbox2, label2, det2, idx2 = all_detections[1]

        im_width = image.size[0]
        margin = self.margin_ratio * im_width
        right_edge_of_left_most = float(bbox1[2])
        left_edge_of_second_left_most = float(bbox2[0])
        overlap = right_edge_of_left_most + margin > left_edge_of_second_left_most
        if overlap:
            # Not enough horizontal gap – ambiguous
            return []

        # Ensure leftmost is on left half fully
        left_edge_of_left_most, right_edge_of_left_most = float(bbox1[0]), float(
            bbox1[2]
        )
        if not (
            left_edge_of_left_most < image.size[0] / 2
            and right_edge_of_left_most < image.size[0] / 2
        ):
            return []
        return [(self.question, label1)]


class RightMost(Question):
    def __init__(self, margin_ratio: float = 0.05) -> None:
        super().__init__(
            question="What is the rightmost object in the image?",
            variables=[],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
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
        # @precondition: at_least_one_single_detection(image, detections) == True
        # Create list of all individual detections with their positions
        all_detections = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    bbox = det.as_xyxy()[i]
                    all_detections.append((bbox, str(l), det, i))
            else:
                bbox = det.as_xyxy()[0]
                all_detections.append((bbox, str(lbl), det, 0))

        if len(all_detections) == 0:
            return []

        # Sort by right edge (x2) descending
        all_detections.sort(key=lambda x: float(x[0][2]), reverse=True)

        if len(all_detections) < 2:
            # Single detection case: ensure it's on the right half fully
            bbox, label, _, _ = all_detections[0]
            left_edge_of_right_most, right_edge_of_right_most = float(bbox[0]), float(
                bbox[2]
            )
            if (
                left_edge_of_right_most > image.size[0] / 2
                and right_edge_of_right_most > image.size[0] / 2
            ):
                return [(self.question, label)]
            return []

        # Check overlap between first two rightmost
        bbox1, label1, det1, idx1 = all_detections[0]
        bbox2, label2, det2, idx2 = all_detections[1]

        im_width = image.size[0]
        margin = self.margin_ratio * im_width
        left_edge_of_right_most = float(bbox1[0])
        right_edge_of_second_right_most = float(bbox2[2])
        overlap = left_edge_of_right_most - margin < right_edge_of_second_right_most
        if overlap:
            # Not enough horizontal gap – ambiguous
            return []

        # Ensure rightmost is on right half fully
        left_edge_of_right_most, right_edge_of_right_most = float(bbox1[0]), float(
            bbox1[2]
        )
        if not (
            left_edge_of_right_most > image.size[0] / 2
            and right_edge_of_right_most > image.size[0] / 2
        ):
            return []
        return [(self.question, label1)]


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
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        return [
            (self.question.format(object_1=cls), str(cnt))
            for cls, cnt in counts.items()
        ]


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
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        classes = list(counts.keys())
        qa: list[tuple[str, str]] = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                o1, o2 = classes[i], classes[j]
                c1, c2 = counts[o1], counts[o2]
                if c1 > c2:
                    if c1 >= (1 + self.margin_ratio) * c2:
                        qa.append(
                            (self.question.format(object_1=o1, object_2=o2), "Yes")
                        )
                elif c2 > c1:
                    if c2 >= (1 + self.margin_ratio) * c1:
                        qa.append(
                            (self.question.format(object_1=o1, object_2=o2), "No")
                        )
        return qa


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
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
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
    def __init__(
        self, threshold: float = 0.75, spatial_margin_ratio: float = 0.05
    ) -> None:
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
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        # Create list of all individual detections with their positions, filtered for single instances
        all_detections = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if counts.get(key, 0) == 1:
                        bbox = det.as_xyxy()[i]
                        all_detections.append((bbox, key, det, i))
            else:
                key = str(lbl)
                if counts.get(key, 0) == 1:
                    bbox = det.as_xyxy()[0]
                    all_detections.append((bbox, key, det, 0))

        # Sort by left edge (x1)
        all_detections.sort(key=lambda x: float(x[0][0]))

        im_width, _ = image.size

        for pos, (bbox, label, det, idx) in enumerate(all_detections):
            x1, x2 = float(bbox[0]), float(bbox[2])
            if not (x1 < im_width / 2 and x2 < im_width / 2):
                continue  # Must be in left half

            # Check spatial separation if there's a second leftmost
            if pos + 1 < len(all_detections):
                second_bbox, second_label, second_det, second_idx = all_detections[
                    pos + 1
                ]
                second_x1 = float(second_bbox[0])
                required_margin = self.spatial_margin_ratio * im_width
                if (second_x1 - x2) < required_margin:
                    continue  # Not enough separation

                # Check for overlap
                single_det1 = ObjectDetectionResultI(
                    score=(
                        det.score[idx]
                        if isinstance(det.score, torch.Tensor)
                        else det.score
                    ),
                    cls=det.cls[idx] if isinstance(det.cls, torch.Tensor) else det.cls,
                    label=label,
                    bbox=bbox.unsqueeze(0),  # Ensure 2D shape (1, 4)
                    image_hw=(image.height, image.width),
                )
                single_det2 = ObjectDetectionResultI(
                    score=(
                        second_det.score[second_idx]
                        if isinstance(second_det.score, torch.Tensor)
                        else second_det.score
                    ),
                    cls=(
                        second_det.cls[second_idx]
                        if isinstance(second_det.cls, torch.Tensor)
                        else second_det.cls
                    ),
                    label=second_label,
                    bbox=second_bbox.unsqueeze(0),  # Ensure 2D shape (1, 4)
                    image_hw=(image.height, image.width),
                )

                if (
                    ObjectDetectionUtils.pairwise_iou(single_det1, single_det2).max()
                    > 0
                ):
                    logger.debug("Leftmost object overlaps with second-leftmost object")
                    continue

            # Calculate aspect ratio
            w = float(bbox[2] - bbox[0])
            h = float(bbox[3] - bbox[1])
            ratio = w / max(h, 1e-6)

            qa = self._question_answer_ratio(label, ratio, reverse=reverse)
            return [qa] if qa is not None else []

        return []


class RightMostWidthVsHeight(WidthVsHeight):
    def __init__(
        self, threshold: float = 0.75, spatial_margin_ratio: float = 0.05
    ) -> None:
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
        # Count detections per class
        counts: dict[str, int] = {}
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for l in lbl:
                    key = str(l)
                    counts[key] = counts.get(key, 0) + 1
            else:
                key = str(lbl)
                counts[key] = counts.get(key, 0) + 1

        # Create list of all individual detections with their positions, filtered for single instances
        all_detections = []
        for det in detections:
            lbl = det.label
            if isinstance(lbl, torch.Tensor):
                for i, l in enumerate(lbl):
                    key = str(l)
                    if counts.get(key, 0) == 1:
                        bbox = det.as_xyxy()[i]
                        all_detections.append((bbox, key, det, i))
            else:
                key = str(lbl)
                if counts.get(key, 0) == 1:
                    bbox = det.as_xyxy()[0]
                    all_detections.append((bbox, key, det, 0))

        # Sort by right edge (x2) descending
        all_detections.sort(key=lambda x: float(x[0][2]), reverse=True)

        im_width, _ = image.size

        for pos, (bbox, label, det, idx) in enumerate(all_detections):
            x1, x2 = float(bbox[0]), float(bbox[2])
            if not (x1 > im_width / 2 and x2 > im_width / 2):
                continue  # Must be in right half

            # Check spatial separation if there's a second rightmost
            if pos + 1 < len(all_detections):
                second_bbox, second_label, second_det, second_idx = all_detections[
                    pos + 1
                ]
                second_x2 = float(second_bbox[2])
                required_margin = self.spatial_margin_ratio * im_width
                if (x1 - second_x2) < required_margin:
                    continue  # Not enough separation

                # Check for overlap
                single_det1 = ObjectDetectionResultI(
                    score=(
                        det.score[idx]
                        if isinstance(det.score, torch.Tensor)
                        else det.score
                    ),
                    cls=det.cls[idx] if isinstance(det.cls, torch.Tensor) else det.cls,
                    label=label,
                    bbox=bbox.unsqueeze(0),  # Ensure 2D shape (1, 4)
                    image_hw=(image.height, image.width),
                )
                single_det2 = ObjectDetectionResultI(
                    score=(
                        second_det.score[second_idx]
                        if isinstance(second_det.score, torch.Tensor)
                        else second_det.score
                    ),
                    cls=(
                        second_det.cls[second_idx]
                        if isinstance(second_det.cls, torch.Tensor)
                        else second_det.cls
                    ),
                    label=second_label,
                    bbox=second_bbox.unsqueeze(0),  # Ensure 2D shape (1, 4)
                    image_hw=(image.height, image.width),
                )

                if (
                    ObjectDetectionUtils.pairwise_iou(single_det1, single_det2).max()
                    > 0
                ):
                    logger.debug(
                        "Rightmost object overlaps with second-rightmost object"
                    )
                    continue

            # Calculate aspect ratio
            w = float(bbox[2] - bbox[0])
            h = float(bbox[3] - bbox[1])
            ratio = w / max(h, 1e-6)

            qa = self._question_answer_ratio(label, ratio, reverse=reverse)
            return [qa] if qa is not None else []

        return []


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
            raise ValueError("threshold should be > 1.0 for 'more than' questions")

        self.threshold: float = threshold
        super().__init__(
            question="Are there {target} or more {object_1}(s) in this image? Respond Yes/No.",
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
            raise ValueError("threshold must be between 0 and 1 for 'less than'")

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

            # If target is 1, switch to grammatically correct presence question
            if target_no == 1:
                q_no = (
                    "Are there any {object_1}(s) in this image? Respond Yes/No."
                ).format(object_1=cls)
                # Since n > 0 (we skipped n == 0 above), the correct answer is "Yes"
                qa_pairs.append((q_no, "Yes"))
            else:
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
                            scores.append(
                                float(conf[idx])
                                if isinstance(conf, torch.Tensor)
                                else float(conf)
                            )
                else:
                    if str(lbl) == cls:
                        scores.append(float(conf))

            var = float(np.var(scores)) if len(scores) > 1 else 0.0

            buckets, correct_bucket = self._bucket_ranges(n, var)

            # Randomly permute letter → bucket mapping to avoid letter bias
            letters = ["A", "B", "C"]
            random.shuffle(letters)
            bucket_keys = ["low", "mid", "high"]

            letter_to_bucket = {
                letter: bucket for letter, bucket in zip(letters, bucket_keys)
            }

            # Build question text in A/B/C order after permutation
            q = self.question.format(
                object_1=cls,
                range_a=buckets[letter_to_bucket["A"].lower()],
                range_b=buckets[letter_to_bucket["B"].lower()],
                range_c=buckets[letter_to_bucket["C"].lower()],
            )

            # Identify the letter assigned to the correct bucket
            correct_letter = {bkey: ltr for ltr, bkey in letter_to_bucket.items()}[
                correct_bucket
            ]

            qa_pairs.append((q, correct_letter))

        return qa_pairs


class Closer(Question):
    def __init__(self, margin_ratio: float = 0.1) -> None:
        """
        Closer question using depth perception and SAM segmentation.

        Args:
            margin_ratio: Required relative depth difference for reliable comparison.
                Objects must differ by at least this fraction of the closer object's depth.
        """
        super().__init__(
            question="Is there at least one {object_1} that appears closer to the camera than any {object_2}?",
            variables=["object_1", "object_2"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
                ObjectDetectionPredicates.exists_non_overlapping_detections,
            ],
        )
        if margin_ratio <= 0 or margin_ratio >= 1:
            raise ValueError("margin_ratio must be between 0 and 1")
        self.margin_ratio = margin_ratio

        # Initialize SAM and DepthPro models lazily
        self._sam_predictor = None
        self._depth_model = None

    def _get_sam_predictor(self, cache: Optional[dict[str, Any]] = None):
        """Get SAM predictor, preferring shared cache if available."""
        if cache and "sam_predictor" in cache:
            return cache["sam_predictor"]   
        if self._sam_predictor is None:
            from graid.utilities.sam_utils import SAMPredictor
            from graid.utilities.common import get_default_device            
            device = get_default_device()
            self._sam_predictor = SAMPredictor(device=device)
        return self._sam_predictor

    def _get_depth_model(self):
        """Lazy initialization of DepthPro model."""
        if self._depth_model is None:
            from graid.models.DepthPro import DepthPro
            from graid.utilities.common import get_default_device
            device = get_default_device()
            self._depth_model = DepthPro(device=device)
        return self._depth_model

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        return self.apply_with_cache(image, detections, {})

    # New method using shared cache to reuse precomputed depth maps
    def apply_with_cache(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        cache: dict,
    ) -> list[tuple[str, str]]:
        import time
        from graid.utilities.sam_utils import compare_object_depths

        start_time = time.time()
        depth_model = cache.get("depth_model") if cache else None
        if "depth_map" not in cache:
            if depth_model is None:
                depth_model = self._get_depth_model()
            try:
                dm = depth_model.predict_depth(image).depth_prediction
                cache["depth_map"] = dm
            except Exception as e:
                logger.debug(f"Closer depth calculation failed: {e}")
                return []
        depth_map = cache["depth_map"]
        sam_predictor = self._get_sam_predictor(cache)

        # 1. Group detections by class and build a flat list
        class_detections: dict[str, list[ObjectDetectionResultI]] = {}
        for det in detections:
            key = str(det.label)
            class_detections.setdefault(key, []).append(det)

        # 2. Build a mask cache using the batched SAM helper
        mask_cache: dict[int, Optional[torch.Tensor]] = cache.get("sam_masks", {})
        missing = [d for d in detections if id(d) not in mask_cache]
        if missing:
            try:
                new_masks = sam_predictor.get_masks_from_bboxes(image, detections)
            except Exception:
                return []
            if len(new_masks) != len(detections):
                return []
            for d, m in new_masks:
                mask_cache[id(d)] = m
            cache["sam_masks"] = mask_cache

        qa_pairs = []
        classes = list(class_detections.keys())
        main_loop_start_time = time.time()

        for i in range(len(classes)):
            for j in range(len(classes)):
                if i == j:
                                continue
                c1, c2 = classes[i], classes[j]

                found_yes = False
                for det1 in class_detections[c1]:
                    for det2 in class_detections[c2]:
                        if ObjectDetectionUtils.pairwise_iou(det1, det2).max() > 0:
                            continue

                        mask1 = mask_cache.get(id(det1))
                        mask2 = mask_cache.get(id(det2))
                        if mask1 is None or mask2 is None:
                            continue

                        comparison, _, _ = compare_object_depths(
                            depth_map, det1, mask1, det2, mask2, self.margin_ratio
                        )

                        if comparison == "object1_front":
                            qa_pairs.append(
                                (self.question.format(object_1=c1, object_2=c2), "Yes")
                            )
                            found_yes = True
                            break
                    if found_yes:
                        break

                if not found_yes:
                    qa_pairs.append(
                        (self.question.format(object_1=c1, object_2=c2), "No")
                    )

        logger.debug(f"Closer: Main question loop took {time.time() - main_loop_start_time:.4f}s")
        logger.debug(f"Closer: Total apply_with_cache took {time.time() - start_time:.4f}s")
        return qa_pairs


class Farther(Question):
    def __init__(self, margin_ratio: float = 0.1) -> None:
        """
        Farther question using depth perception and SAM segmentation.

        Args:
            margin_ratio: Required relative depth difference for reliable comparison.
        """
        super().__init__(
            question="Is there at least one {object_1} that appears farther from the camera than any {object_2}?",
            variables=["object_1", "object_2"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
                ObjectDetectionPredicates.exists_non_overlapping_detections,
            ],
        )
        if margin_ratio <= 0 or margin_ratio >= 1:
            raise ValueError("margin_ratio must be between 0 and 1")
        self.margin_ratio = margin_ratio

        # Initialize SAM and DepthPro models lazily
        self._sam_predictor = None
        self._depth_model = None

    def _get_sam_predictor(self):
        """Lazy initialization of SAM predictor."""
        if self._sam_predictor is None:
            from graid.utilities.sam_utils import SAMPredictor
            from graid.utilities.common import get_default_device
            device = get_default_device()
            self._sam_predictor = SAMPredictor(device=device)
        return self._sam_predictor

    def _get_depth_model(self):
        """Lazy initialization of DepthPro model."""
        if self._depth_model is None:
            from graid.models.DepthPro import DepthPro
            from graid.utilities.common import get_default_device
            device = get_default_device()
            self._depth_model = DepthPro(device=device)
        return self._depth_model

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        return self.apply_with_cache(image, detections, {})

    def apply_with_cache(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        cache: dict,
    ) -> list[tuple[str, str]]:
        import time
        from graid.utilities.sam_utils import compare_object_depths

        start_time = time.time()
        depth_model = cache.get("depth_model") if cache else None
        if "depth_map" not in cache:
            if depth_model is None:
                depth_model = self._get_depth_model()
            try:
                dm = depth_model.predict_depth(image).depth_prediction
                cache["depth_map"] = dm
            except Exception as e:
                logger.debug(f"Farther depth calculation failed: {e}")
                return []
        depth_map = cache["depth_map"]
        

        sam_predictor = self._get_sam_predictor()
        

        # Group detections by class and flat list
        class_detections: dict[str, list[ObjectDetectionResultI]] = {}
        for det in detections:
            for d_single in det.flatten():
                key = str(d_single.label)
                class_detections.setdefault(key, []).append(d_single)

        # Build mask cache via batched SAM
        # Reuse per-image SAM mask cache if present
        mask_cache: dict[int, Optional[torch.Tensor]] = cache.get("sam_masks", {})
        missing = [d for d in detections if id(d) not in mask_cache]
        if missing:
            try:
                new_masks = sam_predictor.get_masks_from_bboxes(image, detections)
            except Exception:
                return []
            if len(new_masks) != len(detections):
                return []
            for d, m in new_masks:
                mask_cache[id(d)] = m
            cache["sam_masks"] = mask_cache

        qa_pairs = []
        classes = list(class_detections.keys())
        main_loop_start_time = time.time()

        for i in range(len(classes)):
            for j in range(len(classes)):
                if i == j:
                    continue
                c1, c2 = classes[i], classes[j]

                found_yes = False
                for det1 in class_detections[c1]:
                    for det2 in class_detections[c2]:
                        if ObjectDetectionUtils.pairwise_iou(det1, det2).max() > 0:
                            continue

                        mask1 = mask_cache.get(id(det1))
                        mask2 = mask_cache.get(id(det2))
                        if mask1 is None or mask2 is None:
                            continue

                        comparison, _, _ = compare_object_depths(
                            depth_map, det1, mask1, det2, mask2, self.margin_ratio
                        )

                        if comparison == "object2_front":
                            qa_pairs.append(
                                (self.question.format(object_1=c1, object_2=c2), "Yes")
                            )
                            found_yes = True
                            break
                    if found_yes:
                        break

                if not found_yes:
                    qa_pairs.append(
                        (self.question.format(object_1=c1, object_2=c2), "No")
                    )

        logger.debug(f"Farther: Main question loop took {time.time() - main_loop_start_time:.4f}s")
        logger.debug(f"Farther: Total apply_with_cache took {time.time() - start_time:.4f}s")
        return qa_pairs


class DepthRanking(Question):
    """Rank the *k* object classes that are closest to the camera.

    Example question (for k=3):

        "Rank the 3 kinds of objects that appear the closest in the image from
        closest to farthest. Provide your answer as a comma-separated list of
        object names only."
    """

    def __init__(self, k: int, margin_ratio: float = 0.2) -> None:
        """Create a DepthRanking question.

        Args:
            k: number of classes to rank.
            margin_ratio: required multiplicative margin between consecutive
                ranked depths. For class *i* to be considered closer than class
                *i+1*, its depth must be at most `(1 - margin_ratio)` times
                the depth of i+1. If any consecutive pair fails this criterion, the
                question will be skipped for that image.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if not (0 < margin_ratio < 1):
            raise ValueError("margin_ratio must be between 0 and 1")

        self.k: int = k
        self.margin_ratio: float = margin_ratio
        super().__init__(
            question=(
                "Rank the {k} kinds of objects that appear the closest to the camera in the "
                "image from closest to farthest. Provide your answer as a "
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

        # Initialize SAM and DepthPro models lazily
        self._sam_predictor = None
        self._depth_model = None

    def _get_sam_predictor(self):
        """Lazy initialization of SAM predictor."""
        if self._sam_predictor is None:
            from graid.utilities.sam_utils import SAMPredictor
            from graid.utilities.common import get_default_device
            device = get_default_device()
            self._sam_predictor = SAMPredictor(device=device)
        return self._sam_predictor

    def _get_depth_model(self):
        """Lazy initialization of DepthPro model."""
        if self._depth_model is None:
            from graid.models.DepthPro import DepthPro
            from graid.utilities.common import get_default_device
            device = get_default_device()
            self._depth_model = DepthPro(device=device)
        return self._depth_model

    def apply(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
    ) -> list[tuple[str, str]]:
        if len(detections) == 0:
            logger.debug("No detections for DepthRanking question")
            return []
        return self.apply_with_cache(image, detections, {})

    def apply_with_cache(
        self,
        image: Image.Image,
        detections: list[ObjectDetectionResultI],
        cache: dict,
    ) -> list[tuple[str, str]]:
        depth_model = cache.get("depth_model") if cache else None
        if "depth_map" not in cache:
            if depth_model is None:
                depth_model = self._get_depth_model()
            try:
                cache["depth_map"] = depth_model.predict_depth(image).depth_prediction
            except Exception as e:
                logger.debug(f"DepthRanking depth calc failed: {e}")
                return []
        depth_map = cache["depth_map"]

        from graid.utilities.sam_utils import extract_average_depth_from_mask

        sam_predictor = self._get_sam_predictor()

        try:
            if len(detections) < self.k:
                return []

            # Use batched SAM to get masks for all detections
            # Reuse per-image SAM mask cache if available
            mask_cache: dict[int, Optional[torch.Tensor]] = cache.get("sam_masks", {})
            missing = [d for d in detections if id(d) not in mask_cache]
            if missing:
                for d, m in sam_predictor.get_masks_from_bboxes(image, detections):
                    mask_cache[id(d)] = m
                cache["sam_masks"] = mask_cache

            # Compute refined depth per detection
            class_min_depth: dict[str, float] = {}
            for det in detections:
                mask = mask_cache.get(id(det))
                if mask is None:
                    continue
                avg_depth = extract_average_depth_from_mask(depth_map, mask)
                if avg_depth is None:
                    continue
                cls = str(det.label)
                if cls not in class_min_depth or avg_depth < class_min_depth[cls]:
                    class_min_depth[cls] = avg_depth

            if len(class_min_depth) < self.k:
                logger.debug("Not enough classes with valid depth for DepthRanking")
                return []

            sorted_classes = sorted(class_min_depth.items(), key=lambda kv: kv[1])
            top_k = sorted_classes[: self.k]

            # margin check
            for i in range(len(top_k) - 1):
                if top_k[i][1] > (1 - self.margin_ratio) * top_k[i + 1][1]:
                    logger.debug("DepthRanking margin threshold not met between %s and %s", top_k[i][0], top_k[i + 1][0])
                    return []

            labels_ordered = [cls for cls, _ in top_k]
            return [(self.question.format(k=self.k), ", ".join(labels_ordered))]
        except Exception as e:
            logger.debug(f"DepthRanking failed with cache: {e}")
            return []


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

        # Get center points - normalized detections have bbox shape (1, 4)
        centers = []
        for detection in detections:
            bbox = detection.as_xyxy()[0]  # Shape (4,) after indexing
            x_center = float((bbox[0] + bbox[2]) / 2)
            y_center = float((bbox[1] + bbox[3]) / 2)
            centers.append((x_center, y_center))

        # Sort by x-coordinate
        centers = sorted(centers, key=lambda p: p[0])

        # Try sliding windows of 3+ objects
        image_height = image.size[1]

        for window_size in range(3, len(centers) + 1):
            for start in range(len(centers) - window_size + 1):
                window = centers[start : start + window_size]

                # Extract x and y coordinates
                x_coords = np.array([p[0] for p in window]).reshape(-1, 1)
                y_coords = np.array([p[1] for p in window])

                # Fit linear regression
                reg = LinearRegression().fit(x_coords, y_coords)
                y_pred = reg.predict(x_coords)

                # Calculate normalized variance (by image height)
                variance = np.var(y_coords - y_pred)
                normalized_variance = variance / (image_height**2)

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

        # Get centers with labels - normalized detections have bbox shape (1, 4)
        centers_with_labels = []
        for detection in detections:
            bbox = detection.as_xyxy()[0]  # Shape (4,) after indexing
            x_center = float((bbox[0] + bbox[2]) / 2)
            y_center = float((bbox[1] + bbox[3]) / 2)
            label = str(detection.label)
            centers_with_labels.append((x_center, y_center, label))

        # Sort by x-coordinate
        centers_with_labels = sorted(centers_with_labels, key=lambda p: p[0])

        # Find best row arrangement
        best_row = None
        best_variance = float("inf")
        image_height = image.size[1]

        for window_size in range(3, len(centers_with_labels) + 1):
            for start in range(len(centers_with_labels) - window_size + 1):
                window = centers_with_labels[start : start + window_size]

                x_coords = np.array([p[0] for p in window]).reshape(-1, 1)
                y_coords = np.array([p[1] for p in window])

                reg = LinearRegression().fit(x_coords, y_coords)
                y_pred = reg.predict(x_coords)

                variance = np.var(y_coords - y_pred)
                normalized_variance = variance / (image_height**2)

                if (
                    normalized_variance < self.variance_threshold
                    and normalized_variance < best_variance
                ):
                    best_variance = normalized_variance
                    best_row = [p[2] for p in window]  # Extract labels

        if best_row is None:
            return []  # No valid row found

        # Create multiple choice options
        correct_text = ", ".join(sorted(set(best_row)))

        # Generate distractors
        all_labels = [str(d.label) for d in detections]
        unique_labels = list(set(all_labels))
        random.shuffle(unique_labels)

        # Create distinct distractors
        distractor1 = ", ".join(unique_labels[: min(3, len(unique_labels))])
        distractor2 = ", ".join(unique_labels[-min(2, len(unique_labels)) :])

        # Ensure distractors are different from correct answer
        max_attempts = 10
        attempt = 0
        while (
            distractor1 == correct_text
            or distractor2 == correct_text
            or distractor1 == distractor2
        ) and attempt < max_attempts:
            random.shuffle(unique_labels)
            distractor1 = ", ".join(unique_labels[: min(3, len(unique_labels))])
            distractor2 = ", ".join(unique_labels[-min(2, len(unique_labels)) :])
            attempt += 1

        # If still duplicates, skip this question
        if (
            distractor1 == correct_text
            or distractor2 == correct_text
            or distractor1 == distractor2
        ):
            return []

        # Randomly assign correct answer to A, B, or C
        options = [correct_text, distractor1, distractor2]
        random.shuffle(options)
        correct_letter = ["A", "B", "C"][options.index(correct_text)]

        q = self.question.format(
            option_a=options[0], option_b=options[1], option_c=options[2]
        )

        return [(q, correct_letter)]


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

        # Get centers and labels - normalized detections have bbox shape (1, 4)
        centers = []
        labels = []
        for detection in detections:
            bbox = detection.as_xyxy()[0]  # Shape (4,) after indexing
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
        clustering = DBSCAN(eps=eps, min_samples=self.min_samples).fit(centers)
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
                return float("inf")
            return np.mean(np.var(cluster_points, axis=0))

        most_compact_id = min(clusters.keys(), key=cluster_compactness)
        most_compact_objects = list(set(clusters[most_compact_id]))  # Remove duplicates

        # Create multiple choice options
        correct_text = ", ".join(sorted(most_compact_objects))

        # Generate distractors from other clusters or random combinations
        all_unique_labels = list(set(labels))
        random.shuffle(all_unique_labels)

        # Create distractors ensuring they're different from correct answer
        distractor1 = ", ".join(all_unique_labels[: min(3, len(all_unique_labels))])
        distractor2 = ", ".join(all_unique_labels[-min(2, len(all_unique_labels)) :])

        # Ensure distractors are different from correct answer
        max_attempts = 10
        attempt = 0
        while (
            distractor1 == correct_text
            or distractor2 == correct_text
            or distractor1 == distractor2
        ) and attempt < max_attempts:
            random.shuffle(all_unique_labels)
            distractor1 = ", ".join(all_unique_labels[: min(3, len(all_unique_labels))])
            distractor2 = ", ".join(
                all_unique_labels[-min(2, len(all_unique_labels)) :]
            )
            attempt += 1

        # If still duplicates after attempts, skip this question
        if (
            distractor1 == correct_text
            or distractor2 == correct_text
            or distractor1 == distractor2
        ):
            return []

        # Randomly assign correct answer
        options = [correct_text, distractor1, distractor2]
        random.shuffle(options)
        correct_letter = ["A", "B", "C"][options.index(correct_text)]

        q = self.question.format(
            option_a=options[0], option_b=options[1], option_c=options[2]
        )

        return [(q, correct_letter)]


# Dynamically discover all Question classes in this module
import inspect
import sys


def _build_all_questions():
    """Build ALL_QUESTIONS list by discovering all Question subclasses in this module."""
    current_module = sys.modules[__name__]
    question_classes = {}

    # Find all classes that inherit from Question
    for name, obj in inspect.getmembers(current_module, inspect.isclass):
        if (
            issubclass(obj, Question)
            and obj != Question  # Exclude the base class
            and hasattr(obj, "is_applicable")
        ):  # Ensure it's a concrete question class
            question_classes[name] = obj

    return question_classes


# Build the dictionary of available question classes
ALL_QUESTION_CLASSES = _build_all_questions()

# Keep the old ALL_QUESTIONS for backward compatibility, but it's no longer used
ALL_QUESTIONS = []

DEPTH_QUESTIONS = [
    Closer,
    Farther,
    DepthRanking,
]