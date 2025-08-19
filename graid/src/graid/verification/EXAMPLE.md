
from graid.evaluator.vlms import GPT
from graid.evaluator.prompts import SetOfMarkPrompt
from graid.verification import PrecisionVerifier, RecallVerifier

# Create VLM instance 
vlm = GPT(model_name="gpt-4o")

# PrecisionVerifier - verify predicted labels
precision_verifier = PrecisionVerifier(vlm)
is_correct = precision_verifier.verify(cropped_image, "car")

# RecallVerifier - find missed objects  
prompting_strategy = SetOfMarkPrompt()
recall_verifier = RecallVerifier(prompting_strategy, vlm)
no_objects_missed = recall_verifier.verify(region_image, ["car", "truck"])