from abc import ABC, abstractmethod
from PIL import Image
from typing import Any, Callable, List, Tuple

from scenic_reasoning.interfaces.ObjectDetectionI import ObjectDetectionResultI


class Question(ABC):
    @abstractmethod
    def __init__(
        self, 
        question: str,
        variables: List[str],
        predicates: List[Callable],
        answer: Any,
    ) -> None:
        pass

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
        return all(predicate(detections) for predicate in self.predicates)
    
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



