import logging
import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

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
        self, question: str, variables: List[str], predicates: List[Callable]
    ) -> None:
        self.question = question
        self.variables = variables
        self.predicates = predicates

    def is_applicable(
        self,
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
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
        detections: List[ObjectDetectionResultI],
    ) -> List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        # for every kind (label) of object in the image, find the right most detection
        # label -> (center of bbox (x, y), bounding box (x1, y1, x2, y2))
        right_most_detections: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # also the left most
        left_most_detections: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # also the top most
        top_most_detections: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # also the lowest
        bottom_most_detections: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

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
                        right_most_detections[class_name] = (center_box[0], bbox[0])

                # left most
                if class_name not in left_most_detections:
                    left_most_detections[class_name] = (center_box[0], bbox[0])
                else:
                    if center_box[0][0] < left_most_detections[class_name][0][0]:
                        left_most_detections[class_name] = (center_box[0], bbox[0])

                # top most
                if class_name not in top_most_detections:
                    top_most_detections[class_name] = (center_box[0], bbox[0])
                else:
                    if center_box[0][1] < top_most_detections[class_name][0][1]:
                        top_most_detections[class_name] = (center_box[0], bbox[0])

                # bottom most
                if class_name not in bottom_most_detections:
                    bottom_most_detections[class_name] = (center_box[0], bbox[0])
                else:
                    if center_box[0][1] > bottom_most_detections[class_name][0][1]:
                        bottom_most_detections[class_name] = (center_box[0], bbox[0])

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
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:
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
        image: Image, detections: List[ObjectDetectionResultI]
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
        image: Image, detections: List[ObjectDetectionResultI], x: int
    ) -> bool:
        counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) is torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for single_class_name in class_name:
                    counts[single_class_name] = counts.get(single_class_name, 0) + 1
            else:
                counts[class_name] = counts.get(class_name, 0) + 1

        return len(counts) >= x

    @staticmethod
    def at_least_x_detections(
        image: Image, detections: List[ObjectDetectionResultI], x: int
    ) -> bool:
        return len(detections) >= 3

    @staticmethod
    def at_least_x_detections(
        image: Image, detections: List[ObjectDetectionResultI], x: int
    ) -> bool:
        return len(detections) >= 3

    @staticmethod
    def exists_non_overlapping_detections(
        image: Image, detections: List[ObjectDetectionResultI]
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
        image: Image, detections: List[ObjectDetectionResultI], threshold=50
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
    def __init__(self) -> None:
        super().__init__(
            question="Divide the image into thirds. Is the {object_1} centered in the image, or is it off to the left or right?",
            variables=["object_1"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:
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
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

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

            # TODO: verify this design decision manually
            # edge case: if the object is big enough to cover more than 1/3rd
            # then it's ambiguous so we will not answer
            if x_min < image_width / 3 and x_max < image_width / 3:
                answer = "left"
            elif x_min > image_width / 3 and x_max < 2 * image_width / 3:
                answer = "centered"
            elif x_min > 2 * image_width / 3 and x_max > 2 * image_width / 3:
                answer = "right"
            else:
                # object is too big to be centered so skip
                logger.debug(
                    "Object is too big to be left, right or centered. Skipping question."
                )
                continue
            question_answer_pairs.append((question, answer))

        return question_answer_pairs


class WidthVsHeight(Question):
    # TODO: try a bunch of different thresholds for width vs height
    def __init__(self, threshold: float = 0.30) -> None:
        super().__init__(
            question="Is the width of the {object_1} appear to be larger than the height?",
            variables=["object_1"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )
        self.threshold = threshold
        self.other_question = "Is the height of the {object_1} larger than the width?"

    def __repr__(self):
        return f"Question: {self.question} (threshold: {self.threshold})"

    def _question_answer(
        self, class_name: str, detection: ObjectDetectionResultI, reverse: bool = False
    ) -> Optional[Tuple[str, str]]:
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
        detections: List[ObjectDetectionResultI],
        reverse: bool = False,
    ) -> List[Tuple[str, str]]:
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
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

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
                            single_class_name, detection, reverse=reverse
                        )
                        if question_answer_pair is not None:
                            question_answer_pairs.append(question_answer_pair)
            else:
                if class_name in single_detections:
                    question_answer_pair = self._question_answer(
                        class_name, detection, reverse=reverse
                    )
                    if question_answer_pair is not None:
                        question_answer_pairs.append(question_answer_pair)

        return question_answer_pairs


class Quadrants(Question):
    def __init__(self, N: int, M: int) -> None:
        if N <= 0 or M <= 0:
            raise ValueError("N and M must be positive integers")
        # TODO: verify this design decision manually
        # we will support at most a 3x3 grid
        if N * M > 9:
            raise ValueError("N * M must be less than or equal to 9")
        self.rows = N
        self.cols = M
        super().__init__(
            question="Divide the image into a {N} x {M} grid. Number the quadrants from left to right, top to bottom, starting with 1. In what quadrant does the {object_1} appear?",
            variables=["object_1", "N", "M"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def _question_answer(
        self, image: Image.Image, class_name: str, detection: ObjectDetectionResultI
    ) -> Optional[Tuple[str, str]]:
        x_min, y_min, x_max, y_max = detection.as_xyxy()[0]
        detection_width = x_max - x_min
        detection_height = y_max - y_min

        image_width, image_height = image.size

        quadrant_width = image_width / self.cols
        quadrant_height = image_height / self.rows

        if not (
            detection_width < quadrant_width and detection_height < quadrant_height
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
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:
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
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

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
        self.threshold = threshold

    def __repr__(self):
        return f"Question: {self.question} (threshold: {self.threshold})"

    def apply(
        self,
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:
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


class MostAppearance(Question):
    def __init__(self) -> None:
        super().__init__(
            question="What kind of object appears the most frequently in the image?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:
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
                detections_counts[class_name] = detections_counts.get(class_name, 0) + 1

        sorted_detections = sorted(
            detections_counts.items(), key=lambda x: x[1], reverse=True
        )
        if sorted_detections[0][1] == sorted_detections[1][1]:
            # we will not handle ties so better not to answer
            logger.debug("Tie in MostAppearance question")
            return []

        most_detections = sorted_detections[0][0]

        question = self.question
        answer = str(most_detections)
        return [(question, answer)]


class LeastAppearance(Question):
    def __init__(self) -> None:
        super().__init__(
            question="What kind of object appears the least frequently in the image?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )

    def apply(
        self, image: Image.Image, detections: List[ObjectDetectionResultI]
    ) -> List[Tuple[str, str]]:
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
                detections_counts[class_name] = detections_counts.get(class_name, 0) + 1

        sorted_detections = sorted(detections_counts.items(), key=lambda x: x[1])

        if sorted_detections[0][1] == sorted_detections[1][1]:
            # we will not handle ties so better not to answer
            logger.debug("Tie in LeastAppearance question")
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
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:
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
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:
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
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:
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

        x1_inter = max(leftmost_detection[1][0], second_leftmost_detection[1][0])
        x2_inter = min(leftmost_detection[1][2], second_leftmost_detection[1][2])
        y1_inter = max(leftmost_detection[1][1], second_leftmost_detection[1][1])
        y2_inter = min(leftmost_detection[1][3], second_leftmost_detection[1][3])

        inter_width = max(0, x2_inter - x1_inter + 1)
        inter_height = max(0, y2_inter - y1_inter + 1)
        inter_area = inter_width * inter_height

        if inter_area > 0:  # overlapping
            logger.debug("LeftMost question not ask-able due to overlapping detections")
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
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:
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

        x1_inter = max(rightmost_detection[1][0], second_rightmost_detection[1][0])
        x2_inter = min(rightmost_detection[1][2], second_rightmost_detection[1][2])
        y1_inter = max(rightmost_detection[1][1], second_rightmost_detection[1][1])
        y2_inter = min(rightmost_detection[1][3], second_rightmost_detection[1][3])

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
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:
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
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

        question_answer_pairs = []
        for class_name, count in detection_counts.items():
            question_answer_pairs.append(
                (self.question.format(object_1=class_name), str(count))
            )

        return question_answer_pairs


class AreMore(Question):
    # TODO: Create a version of this question that is multiple choice
    def __init__(self) -> None:
        super().__init__(
            question="Are there more {object_1}(s) than {object_2}(s) in this image?",
            variables=["object_1", "object_2"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:

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
                object_1, object_2 = detected_classes[i], detected_classes[j]
                count_1, count_2 = (
                    detection_counts[object_1],
                    detection_counts[object_2],
                )

                if count_1 > count_2:
                    answer = "Yes"
                elif count_2 > count_1:
                    answer = "No"
                else:
                    continue

                question_answer_pairs.append(
                    (self.question.format(object_1=object_1, object_2=object_2), answer)
                )

        return question_answer_pairs


class WhichMore(Question):
    def __init__(self) -> None:
        super().__init__(
            question="What appears the most in this image: {object_1}s, {object_2}s, or {object_3}s?",
            variables=["object_1", "object_2", "objejct_3"],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:

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
    def __init__(self, threshold: float = 0.3) -> None:
        super().__init__(threshold=threshold)
        self.question = (
            "Does the leftmost object in the image appear to be wider than it is tall?"
        )
        self.other_question = (
            "Does the leftmost object in the image appear to be taller than it is wide?"
        )

    def apply(
        self,
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
        reverse: bool = False,
    ) -> List[Tuple[str, str]]:
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
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

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
            # check if the leftmost detection is overlapping with the second leftmost detection
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
    def __init__(self, threshold: float = 0.3) -> None:
        super().__init__(threshold=threshold)
        self.question = (
            "Does the rightmost object in the image appear to be wider than it is tall?"
        )
        self.other_question = "Does the rightmost object in the image appear to be taller than it is wide?"

    def apply(
        self,
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
        reverse: bool = False,
    ) -> List[Tuple[str, str]]:
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
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

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
            # check if the rightmost detection is overlapping with the second rightmost detection
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

            if inter_area > 0:
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


class ObjectsInRow(Question):
    def __init__(self) -> None:
        super().__init__(
            question="Are there any objects arranged in a row?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 1
                ),
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:
        if len(detections) < 3:
            return [(self.question, "No")]

        bboxes = [detection.as_xyxy().squeeze(0) for detection in detections]

        bboxes_sorted_by_x = sorted(
            bboxes, key=lambda bbox: bbox[0]
        )  # Sorted by left boundary

        def y_overlap(min_y1, max_y1, min_y2, max_y2):
            inter = max(0, min(max_y1, max_y2) - max(min_y1, min_y2))
            len1 = max_y1 - min_y1
            len2 = max_y2 - min_y2
            min_len = min(len1, len2)

            # two objects are considered on the same line only if the y overlap is at least 50% of the smaller object.
            # TODO: add this as a threshold.
            return inter >= 0.5 * min_len

        def check_row_alignment(bboxes_sorted):
            for i in range(len(bboxes_sorted) - 2):
                box1, box2, box3 = (
                    bboxes_sorted[i],
                    bboxes_sorted[i + 1],
                    bboxes_sorted[i + 2],
                )

                # Require >=50% y-overlap for each adjacent pair
                if y_overlap(box1[1], box1[3], box2[1], box2[3]) and y_overlap(
                    box2[1], box2[3], box3[1], box3[3]
                ):
                    return True

            return False

        row_detected = check_row_alignment(bboxes_sorted_by_x)

        answer = "Yes" if row_detected else "No"
        return [(self.question, answer)]


class ObjectsInLine(Question):
    def __init__(self) -> None:
        super().__init__(
            question="What objects are arranged in a row?",
            variables=[],
            predicates=[
                # TODO: at least 3 detections
                lambda image, detections: ObjectDetectionPredicates.at_least_x_detections(
                    image, detections, 3
                ),
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 1
                ),
                lambda image, detections: ObjectsInRow().apply(image, detections)[0][1]
                == "Yes",
            ],
        )

    def apply(
        self,
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:
        bboxes = [detection.as_xyxy().squeeze(0) for detection in detections]

        detections_sorted_by_x = sorted(
            detections, key=lambda detection: detection.as_xyxy().squeeze(0)[0]
        )
        bboxes_sorted_by_x = [
            detection.as_xyxy().squeeze(0) for detection in detections_sorted_by_x
        ]

        def y_overlap(min_y1, max_y1, min_y2, max_y2):
            inter = max(0, min(max_y1, max_y2) - max(min_y1, min_y2))
            len1 = max_y1 - min_y1
            len2 = max_y2 - min_y2
            min_len = min(len1, len2)

            return inter >= 0.5 * min_len

        def find_rows(bboxes_sorted) -> List[List[int]]:
            rows = []
            i = 0
            while i < len(bboxes_sorted) - 2:
                current_row_indices = [i]
                for j in range(i + 1, len(bboxes_sorted)):
                    if y_overlap(
                        bboxes_sorted[j - 1][1],
                        bboxes_sorted[j - 1][3],
                        bboxes_sorted[j][1],
                        bboxes_sorted[j][3],
                    ):
                        current_row_indices.append(j)
                    else:
                        break
                if len(current_row_indices) >= 3:
                    rows.append(current_row_indices)
                    i += len(current_row_indices)
                else:
                    i += 1
            return rows

        rows = find_rows(bboxes_sorted_by_x)

        if not rows:
            return [(self.question, "None")]

        # Collect object names per row
        row_descriptions = []
        for idx, row in enumerate(rows):
            object_names = [detections_sorted_by_x[r]._label for r in row]
            row_descriptions.append(f"Row {idx+1}: {', '.join(object_names)}")

        return [(self.question, " | ".join(row_descriptions))]


class MostClusteredObjects(Question):
    def __init__(self, threshold=100) -> None:
        super().__init__(
            question="What group of objects are most clustered together?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2  # Need at least 2 to form a cluster
                ),
                lambda image, detections: ObjectDetectionPredicates.has_clusters(
                    image, detections, threshold=threshold
                ),
            ],
        )
        self.threshold = threshold

    def apply(
        self,
        image: Image.Image,
        detections: List[ObjectDetectionResultI],
    ) -> List[Tuple[str, str]]:

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
                if j not in visited and dists[i][j] < self.threshold:
                    cluster.append(j)
                    visited.add(j)
            if len(cluster) >= 2:
                clusters.append(cluster)

        def compactness(cluster_indices):
            cluster_centers = centers[cluster_indices]
            if len(cluster_centers) < 2:
                return float("inf")
            return pdist(cluster_centers).mean()

        clusters.sort(key=lambda c: compactness(c))
        most_compact_cluster = clusters[0]

        object_names = [detections[i]._label for i in most_compact_cluster]
        return [(self.question, f"{', '.join(object_names)}")]


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
