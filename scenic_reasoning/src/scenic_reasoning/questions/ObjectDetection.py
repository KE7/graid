import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple

import torch
from PIL import Image
from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionResultI

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
    def at_least_one_single_detection(image, detections) -> bool:
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
    def at_least_x_many_class_detections(image, detections, x: int) -> bool:
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
                lambda image, detections: ObjectDetectionPredicates.at_least_x_many_class_detections(image, detections, 2),
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