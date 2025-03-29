import base64
import json
import os
import re
from google import genai
from google.genai import types
import requests
import torch
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from torchvision import transforms
import time
import cv2
import numpy as np


class GPT:
    def __init__(self, model_name="gpt-4o", port=None):
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def encode_image(self, image):
        if isinstance(image, torch.Tensor):
            transform = transforms.ToPILImage()
            pil_image = transform(image)
            image_bytes = pil_image.tobytes()
            base64_string = base64.b64encode(image_bytes).decode("utf-8")
            return base64_string
        elif isinstance(image, str):
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(image, np.ndarray):
            success, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')

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

        responses = completion.choices[0].message.content

        return responses, prompt

    def __str__(self):
        return "GPT"


class Gemini:
    def __init__(self, location="us-central1"):
        self.client = genai.Client(
            vertexai=True, project="graid-451620", location="us-central1",
            )
        self.model = "gemini-1.5-pro"
    
    def encode_image(self, image):
        if isinstance(image, str):
            image = Image.open(image)
            return image
        elif isinstance(img_np, np.ndarray):
            return Image.fromarray(img_np)
        else:
            transform = transforms.ToPILImage()
            pil_image = transform(image)
            return pil_image

    def generate_answer(self, image, questions: str, prompting_style):
        image, prompt = prompting_style.generate_prompt(image, questions)
        image = self.encode_image(image)

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        prompt, image
                    ],
                )
                break
            except Exception as e:
                print(e)
                time.sleep(5)

        return response.text, prompt
    
    def __str__(self):
        return "Gemini"


class Llama:
    def __init__(self, model_name="meta/llama-3.2-90b-vision-instruct-maas", region="us-central1"):
        PROJECT_ID = "graid-451620"
        REGION = region
        ENDPOINT = f"{REGION}-aiplatform.googleapis.com"
        self.model = model_name

        self.url = f"https://{ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"

    def encode_image(self, image):
        if isinstance(image, torch.Tensor):
            transform = transforms.ToPILImage()
            pil_image = transform(tensor)
            image_bytes = pil_image.tobytes()
            base64_string = base64.b64encode(image_bytes).decode()
            return base64_string
        elif isinstance(image, np.ndarray):
            success, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        else:
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
                        {"text": prompt, "type": "text"},
                    ],
                }
            ],
            "max_tokens": 40,
            "temperature": 0.4,
            "top_k": 10,
            "top_p": 0.95,
            "n": 1,
        }

        token = "ya29.a0AeXRPp6ikg2fDVbZbDnXJgpo3JQBtOW_GDEGqrVLzYRt1YMYOPm97jiA-_wNWZ53UkD17ejwXsLbnbbmgUjyPnk_nk77IoF1UlJjxZwJ1p22TitIp92rHPutJXx0ByX1uy8kTdJJYSRIE_naSx2LXvfmZ6tRv1YgxrSYLmgk_NXwZ230aCgYKAX4SARMSFQHGX2MiWyhWnflfKV_SL98_ju4-6Q0183"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], prompt
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None, prompt
    
    def __str__(self):
        return "Llama"





# from prompts import ZeroShotPrompt
# model = Gemini()
# model.generate_answer("../demo/demo.jpg", "Tell me about this image", prompting_style=ZeroShotPrompt())
# print()