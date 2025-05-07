import ast
import json
import os
import re
from textwrap import dedent
from typing import List

import outlines
import requests
from dotenv import load_dotenv
from guidance import gen, image, models
from outlines import generate, models
from pydantic import BaseModel


class EvaluationMetric:
    """Base class for different evaluation metrics."""

    def evaluate(self, pred, gt) -> float:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Each subclass must implement this method.")

    def evaluate_batch(self, preds, gts) -> List[float]:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Each subclass must implement this method.")


class ExactMatch(EvaluationMetric):
    def __init__(self):
        pass

    def evaluate(self, pred, gt) -> float:
        pred_as_json = None
        try:
            pred_as_json = json.loads(pred)
        except:
            pass
        try:
            if pred_as_json and "answer" in pred_as_json:
                pred = pred_as_json["answer"]
            elif pred_as_json and "final_answer" in pred_as_json:
                pred = pred_as_json["final_answer"]
            else:
                match = re.search(r"```(.*?)```", pred, re.DOTALL)
                if match:
                    pred = match.group(1).strip()
                else:
                    pred = pred.strip()
        except:
            return 0.0

        return 1.0 if str(pred).lower() == gt.strip().lower() else 0.0

    def __str__(self):
        return "ExactMatch"


# class LLMJudge(EvaluationMetric):
#     """LLM-as-a-judge evaluation metric."""

#     def __init__(self, model_name="meta/llama-3.2-90b-vision-instruct-maas", region="us-central1"):
#         PROJECT_ID = "graid-451620"
#         REGION = region
#         ENDPOINT = f"{REGION}-aiplatform.googleapis.com"
#         self.model = model_name

#         self.url = f"https://{ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"

#         with open("token.txt", "r") as token_file:
#             self.token = token_file.read().strip()


#     def evaluate(self, preds, gts):

#         prompt = f"""
#         Determine if the prediction matches the solution:
#         Solution: {gts}
#         Prediction: {preds}
#         Score the prediction with either 0 (incorrect) or 1 (correct). Make sure to only return the score and nothing else.
#         """

#         payload = {
#             "model": self.model,
#             "stream": False,
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"text": prompt, "type": "text"},
#                     ],
#                 }
#             ],
#             "max_tokens": 40,
#             "temperature": 0.4,
#             "top_k": 10,
#             "top_p": 0.95,
#             "n": 1,
#         }


#         headers = {
#             "Authorization": f"Bearer {self.token}",
#             "Content-Type": "application/json",
#         }

#         response = requests.post(self.url, headers=headers, json=payload)
#         score = response.json()["choices"][0]["message"]["content"]
#         match = re.search(r'\d', score)
#         if match:
#             pred = match.group()

#         return int(pred)


#     def __str__(self):
#         return "LLMJudge"


class LLMJudge(EvaluationMetric):
    """LLM-as-a-judge evaluation metric."""

    def __init__(self, location="us-central1"):
        from google import genai

        PROJECT_ID = "graid-451620"
        MAAS_ENDPOINT = "https://us-central1-aiplatform.googleapis.com"

        self.client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location="us-central1",
        )
        self.model = "meta/llama-3.2-90b-vision-instruct-maas"

    def evaluate(self, pred, gt) -> float:

        prompt = f"""\
        Determine if the prediction matches the solution:
        Score each prediction with either False (incorrect) or True (correct). Give the score for all predictions in a list wrapped by "```" format like this: ```[False, True]```.
        Don't include any other numbers in your response besides the score.

        Here're some examples for you to follow:

        Example 1:
        Solution: right, left
        Prediction: Off to the right, there's a car on the right
        Score: True, False
        Return: ```[True, False]```

        Example 2:
        Solution: left, centered
        Prediction: centered, No car is detected in the image
        Score: False, False
        Return: ```[False, False]```
        
        Example 3:
        Solution: centered, right
        Prediction: looks like it's centered, it's on the right of the image
        Score: True, True
        Return: ```[True, True]```

        Example 4:
        Solution: left
        Prediction: I don't know the answer
        Score: False
        Return: ```[False]```

        Here's the actual task:
        Solution: {gt}
        Prediction: {pred}
        """

        score = 0
        for attempt in range(3):
            try:
                completion = self.client.models.generate_content(
                    model=self.model,
                    contents=[dedent(prompt)],
                )
                response = completion.text or ""
                matches = re.findall(r"```(.*?)```", response, flags=re.DOTALL)
                if matches is None or len(matches) == 0:
                    score = 0
                    break

                value = matches[0].strip("\n")
                value = ast.literal_eval(value)
                if isinstance(value, list):
                    score = float(value[0])
                else:
                    score = float(value)
                break
            except Exception as e:
                print(f"Attempt {attempt+1}: JSON parsing failed - {e}")
                score = 0

        return score

    def __str__(self):
        return "LLMJudge"


class Score(BaseModel):
    score: float


class Scores(BaseModel):
    scores: list[Score]


class ConstrainedDecoding(EvaluationMetric):
    """Constrained decoding metric."""

    def __init__(self, gpu=0, use_batch=False):

        model = models.transformers(
            "microsoft/Phi-3-mini-4k-instruct", device=f"cuda:{gpu}"
        )
        # print(f"downloading {model_name}")
        self.use_batch = use_batch
        if use_batch:
            self.generator = generate.json(model, Scores)
        else:
            self.generator = generate.json(model, Score)

    def evaluate(self, pred, gt):
        return (
            self._evaluate_batch(pred, gt)
            if self.use_batch
            else self._evaluate(pred, gt)
        )

    def _evaluate(self, pred, gt) -> float:
        prompt = f"""
        Determine if the prediction matches the solution:
        Solution: {gt}
        Prediction: {pred}
        Score the prediction with either 0 (incorrect) or 1 (correct).
        """
        result: Score = self.generator(prompt)  # type: ignore

        return result.score

    def _evaluate_batch(self, pred, gt) -> List[float]:
        prompt = f"""
        Determine if the prediction matches the solution:
        Solution: {gt}
        Prediction: {pred}
        Score each prediction with either 0 (incorrect) or 1 (correct). Give the score for all predictions as a list whose length is the same as the number of predictions.
        """
        results: Scores = self.generator(prompt)  # type: ignore

        return [result.score for result in results.scores]

    def __str__(self):
        return "ConstrainedDecoding"
