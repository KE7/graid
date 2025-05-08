import base64
import io
import re
import time
from enum import Enum
from typing import Any, Dict, List, Literal, Type, cast

import cv2
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field
from scenic_reasoning.utilities.coco import coco_labels
from scenic_reasoning.utilities.common import project_root_dir
from torchvision import transforms


class GPT:
    def __init__(self, model_name="gpt-4o", port=None):
        load_dotenv()
        OPENAI_API_KEY = ""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def encode_image(self, image):
        if isinstance(image, torch.Tensor):
            np_img = image.mul(255).byte().numpy().transpose(1, 2, 0)
            img_pil = Image.fromarray(np_img)
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG")
            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return base64_str
        elif isinstance(image, str):
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(image, np.ndarray):
            success, buffer = cv2.imencode(".jpg", image)
            return base64.b64encode(buffer).decode("utf-8")

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
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            temperature=0.0,
        )

        responses = completion.choices[0].message.content

        return responses, prompt

    def __str__(self):
        return "GPT"


class Gemini:
    def __init__(self, location="us-central1"):
        self.client = genai.Client(
            vertexai=True,
            project="graid-451620",
            location="us-central1",
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

        response = None
        for _ in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[image, prompt],
                )
                break
            except Exception as e:
                print(e)
                time.sleep(5)

        if response is None:
            raise Exception("Failed to generate content after multiple attempts")
        return response.text, prompt

    def __str__(self) -> str:
        return "Gemini"


class Llama:
    def __init__(self, model_name="unsloth/Llama-3.2-90B-Vision-Instruct"):
        PROJECT_ID = "graid-451620"
        REGION = "us-central1"
        ENDPOINT = f"http://127.0.0.1:9099/v1/"
        self.model = model_name

        # self.url = f"https://{ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"
        self.url = ENDPOINT

        with open("token.txt", "r") as token_file:
            self.token = token_file.read().strip()

        import openai

        self.client = openai.OpenAI(
            base_url=self.url,
            api_key="vLLM",
            # base_url=f"https://{ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi",
            # api_key=self.token,
        )

    def encode_image(self, image):
        if isinstance(image, torch.Tensor):
            np_img = image.mul(255).byte().numpy().transpose(1, 2, 0)
            img_pil = Image.fromarray(np_img)
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG")
            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return base64_str
        elif isinstance(image, np.ndarray):
            success, buffer = cv2.imencode(".jpg", image)
            return base64.b64encode(buffer).decode("utf-8")
        else:
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_answer(self, image, questions: str, prompting_style):
        image, prompt = prompting_style.generate_prompt(image, questions)
        base64_image = self.encode_image(image)

        image_gcs_url = f"data:image/jpeg;base64,{base64_image}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_gcs_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content, prompt

    def __str__(self) -> str:
        return "Llama"


# class Llama_CoT(Llama):
#     def __init__(
#         self, model_name="unsloth/Llama-3.2-90B-Vision-Instruct"
#     ):
#         super().__init__(model_name)

#     def generate_answer(self, image, questions: str, prompting_style):
#         image, prompt = prompting_style.generate_prompt(image, questions)
#         base64_image = self.encode_image(image)
#         prompt_img_path = (
#             project_root_dir()
#             / "data/nuimages/all/samples/CAM_FRONT/n010-2018-07-10-10-24-36+0800__CAM_FRONT__1531189590512488.jpg"
#         )
#         base64_image_prompt = self.encode_image(prompt_img_path)

#         image_gcs_url = f"data:image/jpeg;base64,{base64_image}"
#         prompt_image_url = f"data:image/jpeg;base64,{base64_image_prompt}"

#         payload = {
#             "model": self.model,
#             "stream": False,
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "image_url": {"url": prompt_image_url},
#                             "type": "image_url",
#                         },
#                         {"text": prompt, "type": "text"},
#                     ],
#                 },
#                 {
#                     "role": "user",
#                     "content": [
#                         {"image_url": {"url": image_gcs_url}, "type": "image_url"},
#                         {"text": questions, "type": "text"},
#                     ],
#                 },
#             ],
#             "temperature": 0.0,
#         }

#         headers = {
#             "Authorization": f"Bearer {self.token}",
#             "Content-Type": "application/json",
#         }

#         response = requests.post(self.url, headers=headers, json=payload)
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"], prompt
#         else:
#             print(f"Error {response.status_code}: {response.text}")
#             return None, prompt

#     def __str__(self):
#         return "Llama_CoT"


CocoLabelEnum = Enum(
    "CocoLabelEnum",
    list(coco_labels.values()) + ["I don't know"],
    type=str,
)


class Answer(BaseModel):
    answer: Any


class IsObjectCenteredAnswer(Answer):
    # question: str
    answer: Literal["Left", "Centered", "Right", "I don't know"]


class WidthVsHeightAnswer(Answer):
    # question: str
    answer: Literal["Yes", "No", "I don't know"]


class QuadrantsAnswer(Answer):
    # question: str
    answer: int


class LargestAppearanceAnswer(Answer):
    # question: str
    answer: CocoLabelEnum


class MostAppearanceAnswer(Answer):
    # question: str
    answer: CocoLabelEnum


class LeastAppearanceAnswer(Answer):
    # question: str
    answer: CocoLabelEnum


class LeftOfAnswer(Answer):
    # question: str
    answer: Literal["Yes", "No", "I don't know"]


class RightOfAnswer(Answer):
    # question: str
    answer: Literal["Yes", "No", "I don't know"]


class LeftMostAnswer(Answer):
    # question: str
    answer: CocoLabelEnum


class RightMostAnswer(Answer):
    # question: str
    answer: CocoLabelEnum


class HowManyAnswer(Answer):
    # question: str
    answer: CocoLabelEnum


class AreMoreAnswer(Answer):
    # question: str
    answer: Literal["Yes", "No", "I don't know"]


class WhichMoreAnswer(Answer):
    # question: str
    answer: CocoLabelEnum


class LeftMostWidthVsHeightAnswer(Answer):
    # question: str
    answer: Literal["Yes", "No", "I don't know"]


class RightMostWidthVsHeightAnswer(Answer):
    # question: str
    answer: Literal["Yes", "No", "I don't know"]


class ObjectsInRowAnswer(Answer):
    # question: str
    answer: Literal["Yes", "No", "I don't know"]


class ObjectsInLineAnswer(Answer):
    # question: str
    answer: CocoLabelEnum


class MostClusteredObjectsAnswer(Answer):
    # question: str
    answer: CocoLabelEnum


QUESTION_CLASS_MAP: Dict[str, Type[Answer]] = {
    r"centered in the image": IsObjectCenteredAnswer,
    r"width of the .* larger than the height": WidthVsHeightAnswer,
    r"In what quadrant does .* appear": QuadrantsAnswer,
    r"appears the largest": LargestAppearanceAnswer,
    r"appears the most frequently": MostAppearanceAnswer,
    r"appears the least frequently": LeastAppearanceAnswer,
    r"to the left of any": LeftOfAnswer,
    r"to the right of any": RightOfAnswer,
    r"leftmost object": LeftMostAnswer,
    r"rightmost object": RightMostAnswer,
    r"How many .* are there": HowManyAnswer,
    r"Are there more .* than .*": AreMoreAnswer,
    r"What appears the most in this image": WhichMoreAnswer,
    r"leftmost object .* wider than .* tall": LeftMostWidthVsHeightAnswer,
    r"rightmost object .* wider than .* tall": RightMostWidthVsHeightAnswer,
    r"Are there any objects arranged in a row": ObjectsInRowAnswer,
    r"What objects are arranged in a row": ObjectsInLineAnswer,
    r"group of objects .* clustered together": MostClusteredObjectsAnswer,
}


def get_answer_class_from_question(question: str) -> Type[Answer]:
    for pattern, cls in QUESTION_CLASS_MAP.items():
        if re.search(pattern, question, flags=re.IGNORECASE):
            return cls
    raise ValueError(f"No matching answer class found for: {question}")


class Step(BaseModel):
    """Represents a single step in the reasoning process."""

    explanation: str


class Reasoning(BaseModel):
    steps: List[Step]
    conclusion: str = Field(
        description="A concluding statement summarizing or linking the steps"
    )
    final_answer: str = Field(
        description="The final answer to the question, derived from the reasoning steps"
    )


class GPT_CD(GPT):
    def __init__(self, model_name="gpt-4o", port=None):
        super().__init__(model_name)

    def generate_answer(self, image, questions: str, prompting_style):
        image, prompt = prompting_style.generate_prompt(image, questions)
        base64_image = self.encode_image(image)

        answer_cls = get_answer_class_from_question(questions)

        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            response_format=answer_cls,
            temperature=0.0,
        )

        message = completion.choices[0].message
        if message.parsed:
            final_answer = message.parsed.answer
        else:
            final_answer = message.refusal

        return final_answer, prompt


class Llama_CD(Llama):
    def __init__(self, model_name="unsloth/Llama-3.2-90B-Vision-Instruct"):
        super().__init__(model_name)

    def generate_answer(self, image, questions: str, prompting_style):
        image, prompt = prompting_style.generate_prompt(image, questions)
        base64_image = self.encode_image(image)

        image_gcs_url = f"data:image/jpeg;base64,{base64_image}"

        answer_cls = get_answer_class_from_question(questions)
        # There doesn't seem to be a good way of dynamically setting the final answer type
        # to be the answer_cls so we will include it in the prompt

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_gcs_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            temperature=0.0,
            response_format=answer_cls,
        )

        message = response.choices[0].message
        if message.parsed:
            final_answer = message.parsed.answer
        else:
            final_answer = message.refusal

        return final_answer, prompt

    def __str__(self):
        return "Llama_CD"


class Gemini_CD(Gemini):
    def __init__(self, location="us-central1"):
        super().__init__(location)

    def generate_answer(self, image, questions: str, prompting_style):
        image, prompt = prompting_style.generate_prompt(image, questions)
        image = self.encode_image(image)

        response_format = get_answer_class_from_question(questions)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                image,
                prompt,
                # f"The final_answer should be of type: {response_format.model_json_schema()}",
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": response_format,
                "temperature": 0.0,
                "topK": 1,
            },
        )

        answers: Answer = cast(Answer, response.parsed)
        final_answer = answers.answer

        return final_answer, prompt

    def __str__(self):
        return "Gemini_CD"


class GPT_CoT_CD(GPT):
    def __init__(self, model_name="gpt-4o", port=None):
        super().__init__(model_name)

    def generate_answer(self, image, questions: str, prompting_style):
        image, prompt = prompting_style.generate_prompt(image, questions)
        base64_image = self.encode_image(image)

        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
                {
                    "role": "system",
                    "content": f"The final_answer should be of type: {get_answer_class_from_question(questions).model_json_schema()}",
                },
            ],
            response_format=Reasoning,
            temperature=0.0,
        )

        message = completion.choices[0].message
        if message.parsed:
            output = message.parsed
            output = output.model_dump_json()
        else:
            output = message.refusal

        return output, prompt


class Llama_CoT_CD(Llama):
    def __init__(self, model_name="unsloth/Llama-3.2-90B-Vision-Instruct"):
        super().__init__(model_name)

    def generate_answer(self, image, questions: str, prompting_style):
        image, prompt = prompting_style.generate_prompt(image, questions)
        base64_image = self.encode_image(image)

        image_gcs_url = f"data:image/jpeg;base64,{base64_image}"

        answer_cls = get_answer_class_from_question(questions)
        # There doesn't seem to be a good way of dynamically setting the final answer type
        # to be the answer_cls so we will include it in the prompt

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_gcs_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "system",
                    "content": f"The final_answer should be of type: {answer_cls.model_json_schema()}",
                },
            ],
            response_format=Reasoning,
            temperature=0.0,
        )

        message = response.choices[0].message
        if message.parsed:
            final_answer = message.parsed
            final_answer = final_answer.model_dump_json()
        else:
            final_answer = message.refusal

        return final_answer, prompt

    def __str__(self):
        return "Llama_CoT_CD"


class Gemini_CoT_CD(Gemini):
    def __init__(self, location="us-central1"):
        super().__init__(location)

    def generate_answer(self, image, questions: str, prompting_style):
        image, prompt = prompting_style.generate_prompt(image, questions)
        image = self.encode_image(image)

        response_format = get_answer_class_from_question(questions)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                image,
                prompt,
                f"The final_answer should be of type: {response_format.model_json_schema()}",
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": Reasoning,
                "temperature": 0.0,
                "topK": 1,
            },
        )

        reasoning_response: Reasoning = cast(Reasoning, response.parsed)
        final_answer = reasoning_response.final_answer

        return reasoning_response.model_dump_json(), prompt

    def __str__(self):
        return "Gemini_CoT_CD"


# from prompts import ZeroShotPrompt
# model = Gemini()
# model.generate_answer("../demo/demo.jpg", "Tell me about this image", prompting_style=ZeroShotPrompt())
# print()
