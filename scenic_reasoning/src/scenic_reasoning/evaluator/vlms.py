import base64
import json
import os
import re
import time

import requests
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image
from torchvision import transforms
import time
import cv2
import numpy as np
import io




class GPT:
    def __init__(self, model_name="gpt-4o", port=None):
        load_dotenv()
        OPENAI_API_KEY = "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"
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def encode_image(self, image):
        if isinstance(image, torch.Tensor):
            np_img = image.mul(255).byte().numpy().transpose(1, 2, 0)
            img_pil = Image.fromarray(np_img)
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG")
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return base64_str
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
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
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
                    contents=[prompt, image],
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
        with open("token.txt", "r") as token_file:
            self.token = token_file.read().strip()

    def encode_image(self, image):
        if isinstance(image, torch.Tensor):
            np_img = image.mul(255).byte().numpy().transpose(1, 2, 0)
            img_pil = Image.fromarray(np_img)
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG")
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return base64_str
        elif isinstance(image, np.ndarray):
            success, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        else:
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_answer(self, image, questions: str, prompting_style):
        image, prompt = prompting_style.generate_prompt(image, questions)
        base64_image = self.encode_image(image)

        image_gcs_url = f"data:image/jpeg;base64,{base64_image}"
        

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
            "temperature": 0.4,
            "top_k": 10,
            "top_p": 0.95,
            "n": 1,
        }


        headers = {
            "Authorization": f"Bearer {self.token}",
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
