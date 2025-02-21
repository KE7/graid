import difflib
import openai
import re

class PromptingStrategy:
    """Base class for different prompting strategies."""

    def generate_prompt(self, query):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Each subclass must implement this method.")

class ZeroShotPrompt(PromptingStrategy):
    """Zero-shot prompting method."""
    def generate_prompt(self, query):
        return f"Answer the question based on the image: {query}"

class FewShotPrompt(PromptingStrategy):
    """Few-shot prompting method."""
    def __init__(self, examples):
        """
        Args:
            examples (list): List of (input, output) examples for few-shot prompting.
        """
        self.examples = examples

    def generate_prompt(self, query):
        if not self.examples:
            raise ValueError("Few-shot examples are required but not provided.")
        
        prompt = "Here are some examples:\n"
        for i, (inp, out) in enumerate(self.examples):
            prompt += f"Example {i+1}:\nInput: {inp}\nOutput: {out}\n\n"

        prompt += f"Now, answer the following question:\n{query}"
        return prompt

class SetOfMarkPrompt(PromptingStrategy):
    """Set-of-mark prompting method."""
    def __init__(self, set_of_mark):
        """
        Args:
            set_of_mark (list): List of constraints or reference points.
        """
        self.set_of_mark = set_of_mark

    def generate_prompt(self, query, image):
        if not self.set_of_mark:
            raise ValueError("Set of mark constraints are required but not provided.")

        constraints = "\n".join([f"- {mark}" for mark in self.set_of_mark])
        return f"Ensure the response follows these constraints:\n{constraints}\n\nQuestion: {query}"



class EvaluationMetric:
    """Base class for different evaluation metrics."""
    def evaluate(self, prediction, ground_truth=None):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Each subclass must implement this method.")

class ExactMatchMetric(EvaluationMetric):
    """Exact match metric."""
    def evaluate(self, prediction, ground_truth):
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

class LLMJudgeMetric(EvaluationMetric):
    """LLM-as-a-judge evaluation metric."""
    def __init__(self, llm_model="gpt-4"):
        self.llm_model = llm_model

    def evaluate(self, prediction, ground_truth):
        prompt = f"""
        Evaluate the following response:

        Expected Answer: {ground_truth}
        Model's Response: {prediction}

        Score the response between 0 (completely incorrect) and 1 (perfectly correct).
        Provide a short justification.
        """

        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=[{"role": "system", "content": "You are a fair evaluator of AI-generated text."},
                      {"role": "user", "content": prompt}]
        )

        score_text = response["choices"][0]["message"]["content"]
        score = float(re.search(r"[-+]?\d*\.\d+|\d+", score_text).group())

        return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1

class ConstraintDecodingMetric(EvaluationMetric):
    """Constraint decoding metric."""
    def __init__(self, constraint_rules):
        self.constraint_rules = constraint_rules

    def evaluate(self, prediction, ground_truth=None):
        for rule in self.constraint_rules:
            if not re.search(rule, prediction):
                return 0.0
        return 1.0
