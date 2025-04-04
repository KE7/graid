import json
import os
import re

import outlines
from dotenv import load_dotenv
from guidance import gen, image, models


class EvaluationMetric:
    """Base class for different evaluation metrics."""

    def evaluate(self, pred, gt):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Each subclass must implement this method.")


class ExactMatch(EvaluationMetric):
    def evaluate(self, pred, gt):
        return 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0


class LLMJudge(EvaluationMetric):
    """LLM-as-a-judge evaluation metric."""

    def __init__(self, llm_model="gpt-4"):
        from openai import OpenAI

        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def evaluate(self, preds, gts):

        prompt = f"""
        Determine if the prediction matches the solution:
        Solution: {gts}
        Prediction: {preds}
        Score each prediction with either 0 (incorrect) or 1 (correct). Give an average score for all predictions in json format as follows:
            {{
            "Solution": ...,
            "Prediction": ...,
            "score": ...
            }}
        
        Don't include any other numbers in your response besides the score.

        Here're some examples for you to follow:

        Example 1:
        Solution: right, left
        Prediction: Off to the right, there's a care on the right
        Score: 1, 0
        Return:
        ```{{
        "Solution": "right, left",
        "Prediction": "Off to the right, there's a care on the right",
        "score": [1, 0]
        }}```

        Example 2:
        Solution: left, centered
        Prediction: centered, No car is detected in the image
        Score: 0, 0
        Return:
        ```{{
        "Solution": left, centered
        "Prediction": centered, No car is detected in the image
        "score": [0, 0]
        }}```
        

        Example 3:
        Solution: centered, right
        Prediction: looks like it's centered, it's on the right of the image
        Score: 1, 1
        Return:
        ```{{
        "Solution": centered, right
        "Prediction": looks like it's centered, it's on the right of the image
        "score": [1, 1]
        }}```

        Example 4:
        Solution: left
        Prediction: I don't know the answer
        Score: 0
        Return:
        ```{{
        "Solution": left
        "Prediction": I don't know the answer
        "score": [0]
        }}```
        """

        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "developer",
                            "content": "You are a helpful assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )

                response = completion.choices[0].message.content

                response_clean = response.strip("```").strip("json\n")

                parsed_json = json.loads(response_clean)
                scores = parsed_json["score"]
                break
            except Exception as e:
                print(f"Attempt {attempt+1}: JSON parsing failed - {e}")

        return scores

    def __str__(self):
        return "LLMJudge"


class ConstraintDecoding(EvaluationMetric):
    """Constraint decoding metric."""

    def __init__(self):

        model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
        print(f"downloading {model_name}")
        self.model = outlines.models.transformers(model_name)

    def evaluate(self, pred, gt):
        prompt = f"""
        <|im_start|>system
        You extract information from text.
        <|im_end|>

        <|im_start|>user
        Determine if the prediction matches the solution:
        Solution: {gt}
        Prediction: {pred}
        Score the prediction with either 0 (incorrect) or 1 (correct).

        <|im_end|>
        <|im_start|>assistant
        """

        generator = outlines.generate.choice(self.model, ["0", "1"])
        correctness = generator(prompt)

        return correctness == "1"

    def __str__(self):
        return "ConstraintDecoding"
