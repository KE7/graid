import ast
import logging
from collections.abc import Sequence
from typing import Callable, Optional

from PIL import Image

from graid.evaluator.prompts import PromptingStrategy

logger = logging.getLogger(__name__)


class RegionVerifier:
    """Orchestrates object detection verification using SetOfMarkPrompt and VLM responses.
    
    This class coordinates the verification process by generating prompts for suspicious
    regions, querying the VLM with annotated images, and parsing the responses to 
    determine if any objects were missed by the original detector.

    Parameters
    ----------
    prompting_strategy : object
        Must implement ``generate_prompt(image, question) -> (annotated_image, prompt)``.
        We expect ``SetOfMarkPrompt`` from ``graid.evaluator.prompts`` but any drop-in
        replacement (e.g. mock for tests) is fine.
    vlm_client : Callable[[Image.Image, str], str]
        Function that takes the *annotated, pre-cropped image* and the prompt string, and
        returns the model's raw answer text.
    """

    def __init__(
        self,
        prompting_strategy: PromptingStrategy,
        vlm_client: Callable[[Image.Image, str], str],
    ) -> None:
        self.ps = prompting_strategy
        self.vlm = vlm_client

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def verify(
        self, image: Image.Image, possible_classes: Optional[Sequence[str]] = None
    ) -> bool:
        """Return **True** if *no* objects are detected in the given image.

        The logic:
        1. Takes a pre-cropped image representing the region of suspicion.
        2. Ask the VLM which of the possible objects are present.
        3. Parse VLM output (expects a Python list literal).
        4. Succeed when the list of found labels is empty.
        """
        question = self._build_question(possible_classes)

        annotated, prompt = self.ps.generate_prompt(image, question)

        answer_text = self.vlm(annotated, prompt)
        found_labels = self._parse_answer(answer_text)

        logger.debug(
            "Possible: %s | Found: %s",
            possible_classes,
            found_labels,
        )
        return len(found_labels) == 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_question(possible_classes: Optional[Sequence[str]]) -> str:
        if possible_classes:
            class_list = ", ".join(possible_classes)
            return (
                "Which of these objects are present in the highlighted regions: "
                f"{class_list}? Provide your answer as a python list. "
                "If none, return empty list []."
            )
        else:
            return (
                "Are there any objects present in the highlighted regions? "
                "Provide your answer as a python list of object names. "
                "If none, return empty list []."
            )

    @staticmethod
    def _parse_answer(answer_text: str) -> list[str]:
        """Extract a Python list from raw answer text.

        The model may wrap the list in triple back-ticks; we strip those out
        and fall back to empty list on any parsing error.
        """
        cleaned = answer_text.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[-2 if cleaned.endswith("```") else -1]
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            # If VLM returned single token instead of list
            return [str(parsed)]
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to parse VLM answer '%s': %s", answer_text, e)
        return [] 