from dotenv import load_dotenv
import os
import re
from guidance import image, models, gen

class EvaluationMetric:
    """Base class for different evaluation metrics."""
    def evaluate(self, pred, gt):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Each subclass must implement this method.")

class ExactMatch(EvaluationMetric):
    """Exact match metric."""
    def evaluate(self, pred, gt):
        return 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0

class LLMJudge(EvaluationMetric):
    """LLM-as-a-judge evaluation metric."""
    def __init__(self, llm_model="gpt-4"):
        from openai import OpenAI
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=OPENAI_API_KEY)


    def evaluate(self, pred, gt):
        prompt = f"""
        Evaluate the following response:

        Expected Answer: {gt}
        Model's Response: {pred}

        Score the response between 0 (completely incorrect) and 1 (perfectly correct).
        Provide a short justification.
        """

        prompt = f"""
        Determine if the prediction matches the ground truth solution:
        Ground Truth: {gt}
        Prediction: {pred}

        Score the response with either 0 (incorrect) or 1 (correct).
        """


        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            )


        response = completion.choices[0].message
        correctness = re.search(r"[01]", response)

        return correctness == "1"

class ConstraintDecoding(EvaluationMetric):
    """Constraint decoding metric."""
    def __init__(self):

        self.gemini = models.VertexAI("gemini-pro-vision")

    def generate_prompt(self, image, question):
        with user():
            lm = self.gemini + question + image(image)

        with assistant():
            lm += gen("answer")
