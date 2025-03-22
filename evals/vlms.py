import torch
from torchvision import transforms
import base64
import requests
import torch
from PIL import Image
from dotenv import load_dotenv
import os
import re
from openai import OpenAI
import json


class VLM:
    def __init__(self):
        pass

    def encode_image(self, image):
        pass

    def generate_answer(self, image, question: str) -> str:
        pass
    
    def parse_answer(self, answer):

        pass


class GPT:
    def __init__(self, model_name="gpt-4o", port=None):
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def encode_image(self, image):
        if isinstance(image, torch.Tensor):
            transform = transforms.ToPILImage()
            pil_image = transform(tensor)
            image_bytes = pil_image.tobytes()
            base64_string = base64.b64encode(image_bytes).decode()
            return
        elif isinstance(image, str):
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
            
    def generate_answer(self, image, questions: str, prompting_style):
        # reference: https://platform.openai.com/docs/guides/vision

        image, prompt = prompting_style.generate_prompt(image, questions)

        base64_image = self.encode_image(image)

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                                },
                        },
                    ],
                }
            ],
        )

        responses = completion.choices[0].message.content.split(",")

        return [re.sub(r'[^a-zA-Z]', '', response).strip() for response in responses]
    
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
    
    def generate_answer(self, image, question: str, prompting_style):

        image = self.encode_image(image)

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[question, image])
        
        return response
    

class Qwen(GPT):
    def __init__(self, model_name="unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit", port=7000):
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = f"http://localhost:{port}/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model_name = model_name
    

class Llama:
    def __init__(self, model_name="meta/llama-3.2-90b-vision-instruct-maas"):
        PROJECT_ID = "graid-451620"
        REGION = "us-central1"
        ENDPOINT = "us-central1-aiplatform.googleapis.com"
        self.model = model_name

        self.url = f"https://{ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"
    
    def encode_image(self, image):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def generate_answer(self, image, questions: str, prompting_style):
        base64_image = self.encode_image(image)
        image_gcs_url = f"data:image/jpeg;base64,{base64_image}"
        image, prompt = prompting_style.generate_prompt(image, questions)

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image_url": {"url": image_gcs_url}, "type": "image_url"},
                        {"text": prompt, "type": "text"}
                    ]
                }
            ],
            "max_tokens": 40,
            "temperature": 0.4,
            "top_k": 10,
            "top_p": 0.95,
            "n": 1
        }

        token = "ya29.a0AeXRPp6Ikm25eRVKvhol8-l1SYDMi098LbNqqE47wTphzqfx9QO4ScKNsaOQhxv67Flcpva4a1endwwREH3HfXben6rbP6NVwLqN9Odwbe504uvFfjEcqVN0oGbJz5Ex-v51U0j6XnpliG1eJg7lBd8LCx25oDADrKvQAQYviSrBOJG7aCgYKAfQSARMSFQHGX2Mi4SdPTfGkdztdeycn8rGi1w0183"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
        