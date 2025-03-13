from dotenv import load_dotenv
import os
import re
from guidance import image, models, gen
import outlines

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


    def evaluate(self, pred, gt):

        prompt = f"""
        Determine if the prediction matches the solution:
        Solution: {gt}
        Prediction: {pred}
        Score the prediction with either 0 (incorrect) or 1 (correct).

        Here're some examples for you to follow:

        Example 1:
        Solution: right
        Prediction: Offtotheright
        Score: 1

        Example 2:
        Solution: left
        Prediction: centered
        Score: 0

        Example 3:
        Solution: centered
        Prediction: looks like it's centered
        Score: 1

        Example 4:
        Solution: left
        Prediction: Empty
        Score: 0
        """

        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            )

        response = completion.choices[0].message.content
        correctness = re.search(r"[01]", response).group()

        return correctness == '1'

class ConstraintDecoding(EvaluationMetric):
    """Constraint decoding metric."""
    def __init__(self):

        model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
        print(f'downloading {model_name}')
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

        import pdb
        pdb.set_trace()

        return correctness == '1'


