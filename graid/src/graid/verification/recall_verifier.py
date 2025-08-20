import ast
import logging
from collections.abc import Sequence
from typing import Optional

from graid.evaluator.prompts import PromptingStrategy
from graid.evaluator.vlms import VLM
from PIL import Image

logger = logging.getLogger(__name__)


class RecallVerifier:
    """Orchestrates object detection verification using a VLM and a prompting strategy.

    This class coordinates the verification process by generating prompts for
    suspicious regions, querying the VLM with annotated images, and parsing
    the responses to determine if any objects were missed by the original
    detector.

    The prompting behavior can be controlled by providing different strategies
    at instantiation. For example, use ``SetOfMarkPrompt`` to visually highlight
    regions of interest, or ``PassthroughPrompt`` to send the image as-is.

    Parameters
    ----------
    prompting_strategy : PromptingStrategy
        A prompting strategy, e.g., ``SetOfMarkPrompt`` or
        ``PassthroughPrompt`` from ``graid.evaluator.prompts``.
    vlm : VLM
        Instance of a VLM class that adheres to the VLM interface.
    """

    def __init__(
        self,
        prompting_strategy: PromptingStrategy,
        vlm: VLM,
    ) -> None:
        self.ps = prompting_strategy
        self.vlm = vlm

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def verify(
        self,
        image: Image.Image,
        possible_classes: Optional[Sequence[str]] = None,
    ) -> bool:
        """Return **True** if *no* objects are detected in the given region of suspicion.

        Parameters
        ----------
        image : PIL.Image.Image
            *Pre-cropped* region of suspicion.
        possible_classes : Sequence[str] | None, optional
            Candidate classes to ask the VLM about. If ``None`` we let the
            model answer freely.
        """
        # STEP 1: build the textual question adapted to the chosen strategy
        question = self._build_question(possible_classes)

        # STEP 2: Generate the prompt, which may annotate the image
        annotated_image, messages = self.ps.generate_prompt(image, question)

        # STEP 3: query the VLM using the standard interface and parse the answer
        answer_text, _ = self.vlm.generate_answer(annotated_image, messages)
        found_labels = self._parse_answer(answer_text)

        logger.debug(
            "Possible: %s | Found: %s | Prompting: %s",
            possible_classes,
            found_labels,
            self.ps.__class__.__name__,
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
                "Which of these objects are present in the image: "
                f"{class_list}? Provide your answer as a python list. "
                "If none, return empty list []."
            )
        else:
            return (
                "Are there any objects present in the image? "
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
