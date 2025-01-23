import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from scenic_reasoning.interfaces.ObjectDetectionI import (
    ObjectDetectionResultI,
    ObjectDetectionUtils,
)

logger = logging.getLogger(__name__)


class Question(ABC):
    @abstractmethod
    def __init__(
        self,
        question: str,
        variables: List[str],
        predicates: List[Callable],
        answer: Any,
    ) -> None:
        self.question = question
        self.variables = variables
        self.predicates = predicates
        self.answer = answer

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
            if type(class_name) == torch.Tensor:  # shape == (# of boxes,)
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
                    if center_box[0][0] > right_most_detections[class_name][0]:
                        right_most_detections[class_name] = (center_box[0], bbox[0])

                # left most
                if class_name not in left_most_detections:
                    left_most_detections[class_name] = (center_box[0], bbox[0])
                else:
                    if center_box[0][0] < left_most_detections[class_name][0]:
                        left_most_detections[class_name] = (center_box[0], bbox[0])

                # top most
                if class_name not in top_most_detections:
                    top_most_detections[class_name] = (center_box[0], bbox[0])
                else:
                    if center_box[0][1] < top_most_detections[class_name][1]:
                        top_most_detections[class_name] = (center_box[0], bbox[0])

                # bottom most
                if class_name not in bottom_most_detections:
                    bottom_most_detections[class_name] = (center_box[0], bbox[0])
                else:
                    if center_box[0][1] > bottom_most_detections[class_name][1]:
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


class ObjectDetectionPredicates:
    @staticmethod
    def at_least_one_single_detection(
        image: Image, detections: List[ObjectDetectionResultI]
    ) -> bool:
        counts = {}
        for detection in detections:
            class_name = detection.label
            if type(class_name) == torch.Tensor:  # shape == (# of boxes,)
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
            if type(class_name) == torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for class_name in class_name:
                    counts[class_name] = counts.get(class_name, 0) + 1
            else:
                counts[class_name] = counts.get(class_name, 0) + 1

        return len(counts) >= x

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


class IsObjectCentered(Question):
    def __init__(self) -> None:
        super().__init__(
            question="Is the {object_1} centered in the image, or is it off to the left or right?",
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
            if type(class_name) == torch.Tensor:  # shape == (# of boxes,)
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
            if type(class_name) == torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for class_name in class_name:
                    if class_name in single_detections:
                        object_positions.append(
                            (
                                class_name,
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
            # the answer should be left or right
            if x_min < image_width / 3 and x_max < 2 * image_width / 3:
                answer = "left"
            elif x_min > image_width / 3 and x_max < 2 * image_width / 3:
                answer = "centered"
            elif x_min > image_width / 3 and x_max > 2 * image_width / 3:
                answer = "right"
            else:
                # x_min < image_width / 3 and x_max > 2 * image_width / 3
                answer = "centered"
            question_answer_pairs.append((question, answer))

        return question_answer_pairs


class WidthVsHeight(Question):
    def __init__(self) -> None:
        super().__init__(
            question="Is the width of the {object_1} larger than the height?",
            variables=["object_1"],
            predicates=[
                ObjectDetectionPredicates.at_least_one_single_detection,
            ],
        )

    def _question_answer(
        self, class_name: str, detection: ObjectDetectionResultI
    ) -> Optional[Tuple[str, str]]:
        width = detection.as_xyxy()[0][2] - detection.as_xyxy()[0][0]
        height = detection.as_xyxy()[0][3] - detection.as_xyxy()[0][1]
        question = self.question.format(object_1=class_name)
        # TODO: verify this design decision manually
        # if the image is roughly square (within 10% of each other), return None
        # TODO: should we check for a minimum width or height?
        if abs(width - height) / width < 0.1:
            return None
        answer = "yes" if width > height else "no"
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
            if type(class_name) == torch.Tensor:  # shape == (# of boxes,)
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

        question_answer_pairs = []
        for detection in detections:
            class_name = detection.label
            if type(class_name) == torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for class_name in class_name:
                    if class_name in single_detections:
                        question_answer_pair = self._question_answer(
                            class_name, detection
                        )
                        if question_answer_pair is not None:
                            question_answer_pairs.append(question_answer_pair)
            else:
                if class_name in single_detections:
                    question_answer_pair = self._question_answer(class_name, detection)
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

        self.question = self.question.format(N=N, M=M)

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

        question = self.question.format(object_1=class_name)
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
            if type(class_name) == torch.Tensor:  # shape == (# of boxes,)
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

        question_answer_pairs = []
        for detection in detections:
            class_name = detection.label
            if type(class_name) == torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for class_name in class_name:
                    if class_name in single_detections:
                        question_answer_pair = self._question_answer(
                            image, class_name, detection
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
    def __init__(self) -> None:
        super().__init__(
            question="Which object appears the largest in the image?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(
                    image, detections, 2
                ),
            ],
        )

        self.question = self.question

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

        question = self.question
        answer = "The " + str(largest_detection.label)
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

        self.question = self.question

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
            if type(class_name) == torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for class_name in class_name:
                    detections_counts[class_name] = (
                        detections_counts.get(class_name, 0) + 1
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
        answer = "The " + str(most_detections)
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

        self.question = self.question

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
            if type(class_name) == torch.Tensor:  # shape == (# of boxes,)
                # need to iterate over the tensor to get the class names
                for class_name in class_name:
                    detections_counts[class_name] = (
                        detections_counts.get(class_name, 0) + 1
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
        answer = "The " + str(least_detections)
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

        self.question = self.question

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

        self.question = self.question

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


class LeftMost(Question):
    def __init__(self) -> None:
        super().__init__(
            question="What is the leftmost object in the image?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_one_single_detection(
                    image, detections
                ),
            ],
        )

        self.question = self.question

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

        raise NotImplementedError("LeftMost question not implemented yet")


class RightMost(Question):
    def __init__(self) -> None:
        super().__init__(
            question="What is the rightmost object in the image?",
            variables=[],
            predicates=[
                lambda image, detections: ObjectDetectionPredicates.at_least_one_single_detection(
                    image, detections
                ),
            ],
        )

        self.question = self.question

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

        raise NotImplementedError("RightMost question not implemented yet")
