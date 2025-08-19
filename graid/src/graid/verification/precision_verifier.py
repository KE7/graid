import logging
from typing import Optional

from PIL import Image

from graid.evaluator.prompts import PromptingStrategy, PassthroughPrompt
from graid.evaluator.vlms import VLM

logger = logging.getLogger(__name__)


class PrecisionVerifier:
    """Verify that a *predicted* label matches the object in a cropped image.

    This verifier focuses on **precision**; that is, confirming a detector's
    *positive* prediction is correct. The caller is expected to supply a
    *pre-cropped* image that contains exactly the detected object (usually a
    detection bounding-box crop) along with the predicted class label.

    Parameters
    ----------
    vlm : VLM
        Instance of a VLM class that adheres to the VLM interface.
    prompting_strategy : PromptingStrategy | None, optional
        Strategy used to build the visual prompt passed to the VLM. Defaults
        to a **no-op** strategy that leaves the image unchanged and only adds
        a textual question.
    yes_tokens : tuple[str, ...], default ("yes", "true", "y", "1")
        Tokens considered affirmative when parsing the VLM's response.
    no_tokens : tuple[str, ...], default ("no", "false", "n", "0")
        Tokens considered negative when parsing the VLM's response.
    """

    def __init__(
        self,
        vlm: VLM,
        prompting_strategy: Optional[PromptingStrategy] = None,
        *,
        yes_tokens: tuple[str, ...] = ("yes", "true", "y", "1"),
        no_tokens: tuple[str, ...] = ("no", "false", "n", "0"),
    ) -> None:
        self.vlm = vlm
        self.ps = prompting_strategy or PassthroughPrompt()
        self._yes_tokens = tuple(token.lower() for token in yes_tokens)
        self._no_tokens = tuple(token.lower() for token in no_tokens)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def verify(self, image: Image.Image, label: str) -> bool:
        """Return **True** when the object in the image matches *label*.

        The method:
        1. Builds a yes/no prompt for the provided label.
        2. Queries the VLM using its ``generate_answer`` interface.
        3. Parses the VLM's yes/no answer.
        4. Returns *True* if the answer is affirmative.
        """
        question = self._build_question(label)

        # Generate the prompt, which may annotate the image
        annotated_image, messages = self.ps.generate_prompt(image, question)

        # Call the VLM using its standard interface; we ignore the second
        # element (messages) in the returned tuple.
        answer_text, _ = self.vlm.generate_answer(annotated_image, messages)

        is_match = self._parse_answer(answer_text)
        logger.debug("Label: '%s' | VLM: '%s' | Match: %s", label, answer_text, is_match)
        return is_match

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_question(label: str) -> str:
        # Request a binary response to simplify parsing.
        return (
            f"Is the object in this image a {label}? "
            "Answer with either 'yes' or 'no'."
        )

    def _parse_answer(self, answer_text: str) -> bool:
        """Interpret the VLM response as *yes* or *no*.

        Any unrecognized response defaults to *False* (i.e., the label does
        *not* match), since we only want to confirm positives we are confident
        about.
        """
        cleaned = answer_text.strip().lower()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[-2 if cleaned.endswith("```") else -1].strip()

        # Extract first token (in case of longer sentences)
        first_token = cleaned.split()[0] if cleaned else ""

        if first_token in self._yes_tokens:
            return True
        if first_token in self._no_tokens:
            return False

        logger.warning("Unrecognized VLM precision-verification answer '%s'", answer_text)
        return False 