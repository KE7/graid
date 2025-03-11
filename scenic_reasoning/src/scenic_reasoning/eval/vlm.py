import torch
from torchvision import transforms
import base64
import requests
from PIL import Image



class VLM:
    """
    A simple class interface for a Visual Language Model (VLM).
    """

    def __init__(self):

    def encode_image(self, image):
        pass

    def generate_answer(self, image, question: str) -> str:
        """
        Given an image and a question, 
        """
        pass
    
    def parse_answer(self, answer):

        pass


class GPT:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()

    def encode_image(self, image):
        if isinstance(image_tensor, torch.Tensor):
            transform = transforms.ToPILImage()
            pil_image = transform(tensor)
            image_bytes = pil_image.tobytes()
            return base64_string = base64.b64encode(image_bytes).decode()
        elif isinstance(image, str):
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
            
    def generate_answer(self, image, question: str, prompting_style):
        # reference: https://platform.openai.com/docs/guides/vision

        base64_image = encode_image(image)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "auto",  #TODO: low, high, or auto?
                                },
                        },
                    ],
                }
            ],
        )

        return response.choices[0]
    
    def parse_answer(self, answer):

        pass


class Gemini:
    def __init__():
        import PIL.Image
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key="GEMINI_API_KEY")
    
    def encode_image(self, image):
        if isinstance(image, str):
            return PIL.Image.open(image)
        elif isinstance(image, torch.Tensor):
            transform = transforms.ToPILImage()
            pil_image = transform(tensor)
            image_bytes = pil_image.tobytes()
            base64_string = base64.b64encode(image_bytes).decode()
            return types.Part.from_bytes(data=base64_string, mime_type="image/jpeg")
    
    def generate_answer(self, image, question: str):

        image = self.encode_image(image)

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",`
            contents=[question, image])`
        
        return response
    

class Llama:
    def __init__(self):
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        model_id = "meta-llama/Llama-3.2-11B-Vision"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate_answer(self, image, question):

        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        prompt = f"<|image|><|begin_of_text|>{question}"
        inputs = processor(image, prompt, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=30)
        return processor.decode(output[0])
