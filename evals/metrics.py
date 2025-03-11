
class EvaluationMetric:
    """Base class for different evaluation metrics."""
    def evaluate(self, prediction, ground_truth=None):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Each subclass must implement this method.")

class ExactMatch(EvaluationMetric):
    """Exact match metric."""
    def evaluate(self, prediction, ground_truth):
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

class LLMJudge(EvaluationMetric):
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

class ConstraintDecoding(EvaluationMetric):
    """Constraint decoding metric."""
    def __init__(self):

        from guidance import image

        gemini = models.VertexAI("gemini-pro-vision")

    def generate_prompt(self, image, question):
        with user():
            lm = gemini + question + image(image)

        with assistant():
            lm += gen("answer")
