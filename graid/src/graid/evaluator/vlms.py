import base64
import io
import os
import re
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, cast

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from torchvision import transforms

from graid.utilities.coco import coco_labels


class VLM(ABC):
    """Abstract Base Class for Vision Language Models."""

    @abstractmethod
    def generate_answer(
        self, image, messages: list[dict[str, str]]
    ) -> tuple[Any, list[dict[str, str]]]:
        """
        Generates an answer from the VLM.

        Args:
            image: The input image (potentially annotated).
            messages: The list of messages for the conversation.

        Returns:
            A tuple containing the VLM's response and the messages passed.
        """
        raise NotImplementedError


class GPT(VLM):
    def __init__(self, model_name="gpt-4o", port=None):
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        if OPENAI_API_KEY == "":
            print("No OpenAI API key found")
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

    def _convert_messages_for_openai(self, messages, base64_image):
        """Convert message list to OpenAI API format with image attached to last user message."""
        converted_messages = []
        
        # Find the last user message to attach the image
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break
        
        for i, msg in enumerate(messages):
            if msg["role"] in ["system", "assistant"]:
                # Pass through system and assistant messages as-is
                converted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            elif msg["role"] == "user":
                if i == last_user_idx:
                    # Attach image to the last user message
                    converted_messages.append({
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
                                "text": msg["content"],
                            },
                        ],
                    })
                else:
                    # Regular user message without image
                    converted_messages.append({
                        "role": "user",
                        "content": msg["content"]
                    })
        
        return converted_messages

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
    )
    def generate_answer(self, image, messages: list[dict[str, str]]):
        # reference: https://platform.openai.com/docs/guides/vision
        base64_image = self.encode_image(image)
        
        converted_messages = self._convert_messages_for_openai(messages, base64_image)

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=converted_messages,
            temperature=0.0,
        )

        responses = completion.choices[0].message.content

        return responses, messages

    def __str__(self):
        return self.model_name


class Gemini(VLM):
    def __init__(self, model_name="gemini-1.5-pro", location="us-central1"):
        self.client = genai.Client(
            vertexai=True,
            project="graid-451620",
            location=location,
        )
        self.model = model_name

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

    def _prepare_gemini_request(self, messages, image):
        """Prepare (system_instruction, contents) tuple for Gemini client.

        Gemini accepts:
          • Optional system_instruction via `config.system_instruction`
          • `contents` – list that can mix strings & PIL.Image
        We consolidate conversational turns into a single prompt string so we
        only need **one** text block + the image.
        """
        system_instruction: str | None = None
        text_parts: list[str] = []

        # Traverse messages in order and build text parts / extract system
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                # Keep the very first system prompt (merge if multiple)
                system_instruction = (
                    content
                    if system_instruction is None
                    else f"{system_instruction}\n{content}"
                )
            elif role == "user":
                text_parts.append(f"User: {content}")
            elif role == "assistant":
                text_parts.append(f"Assistant: {content}")

        combined_prompt = "\n\n".join(text_parts) if text_parts else ""

        # According to docs we can pass [text, image] (text first) or vice-versa.
        contents = [combined_prompt, image]
        return system_instruction, contents

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
    )
    def generate_answer(self, image, messages: list[dict[str, str]]):
        from google.genai import types
        
        image = self.encode_image(image)

        system_instruction, contents = self._prepare_gemini_request(messages, image)

        response = None
        for _ in range(3):
            try:
                params = {"model": self.model, "contents": contents}
                if system_instruction:
                    params["config"] = types.GenerateContentConfig(
                        system_instruction=system_instruction
                    )

                response = self.client.models.generate_content(**params)
                break
            except Exception as e:
                print(e)
                time.sleep(5)

        if response is None:
            raise Exception("Failed to generate content after multiple attempts")
        return response.text, messages

    def __str__(self) -> str:
        return self.model


class Llama(VLM):
    def __init__(
        self, model_name="meta-llama/Llama-3.2-90B-Vision-Instruct", use_vllm=False
    ):
        PROJECT_ID = "graid-451620"
        REGION = "us-central1"
        ENDPOINT = f"http://127.0.0.1:9099/v1/"
        MAAS_ENDPOINT = "us-central1-aiplatform.googleapis.com"
        self.model = model_name

        if use_vllm:
            print("Using private vLLM hosted Llama")
            self.client = OpenAI(
                base_url=ENDPOINT,
                api_key="vLLM",
                # base_url=f"https://{ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi",
                # api_key=self.token,
            )
        else:
            from google.auth import default
            from google.auth.transport.requests import Request

            credentials, _ = default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            credentials.refresh(Request())

            google_url = f"https://{MAAS_ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi"

            print("Using Google Vertex hosted Llama")
            self.client = OpenAI(
                base_url=google_url,
                api_key=credentials.token,
            )
            self.model = "meta/llama-3.2-90b-vision-instruct-maas"

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

    def _convert_messages_for_openai(self, messages, base64_image):
        """Convert message list to OpenAI API format with image attached to last user message."""
        converted_messages = []
        
        # Find the last user message to attach the image
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break
        
        for i, msg in enumerate(messages):
            if msg["role"] in ["system", "assistant"]:
                # Pass through system and assistant messages as-is
                converted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            elif msg["role"] == "user":
                if i == last_user_idx:
                    # Attach image to the last user message
                    converted_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            {"type": "text", "text": msg["content"]},
                        ],
                    })
                else:
                    # Regular user message without image
                    converted_messages.append({
                        "role": "user",
                        "content": msg["content"]
                    })
        
        return converted_messages

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # stop=stop_after_attempt(5),
    )
    def generate_answer(self, image, messages: list[dict[str, str]]):
        base64_image = self.encode_image(image)

        converted_messages = self._convert_messages_for_openai(messages, base64_image)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=converted_messages,
            temperature=0.0,
        )
        print(response)
        return response.choices[0].message.content, messages

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


QUESTION_CLASS_MAP: dict[str, type[Answer]] = {
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


def get_answer_class_from_question(question: str) -> type[Answer]:
    for pattern, cls in QUESTION_CLASS_MAP.items():
        if re.search(pattern, question, flags=re.IGNORECASE):
            return cls
    raise ValueError(f"No matching answer class found for: {question}")


class Step(BaseModel):
    """Represents a single step in the reasoning process."""

    explanation: str


class Reasoning(BaseModel):
    steps: list[Step]
    conclusion: str = Field(
        description="A concluding statement summarizing or linking the steps"
    )
    final_answer: str = Field(
        description="The final answer to the question, derived from the reasoning steps"
    )


class GPT_CD(GPT):
    def __init__(self, model_name="gpt-4o", port=None):
        super().__init__(model_name)

    def generate_answer(self, image, messages: list[dict[str, str]]):
        base64_image = self.encode_image(image)

        question = messages[-1]["content"]
        answer_cls = get_answer_class_from_question(question)
        
        converted_messages = self._convert_messages_for_openai(messages, base64_image)

        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=converted_messages,
            response_format=answer_cls,
            temperature=0.0,
        )

        message = completion.choices[0].message
        if message.parsed:
            final_answer = message.parsed.answer
        else:
            final_answer = message.refusal

        return final_answer, messages

    def __str__(self):
        return self.model_name + "_CD"


class Llama_CD(Llama):
    def __init__(self, model_name="meta-llama/Llama-3.2-90B-Vision-Instruct"):
        super().__init__(model_name, use_vllm=False)

    def generate_answer(self, image, messages: list[dict[str, str]]):
        base64_image = self.encode_image(image)

        question = messages[-1]["content"]
        answer_cls = get_answer_class_from_question(question)
        # There doesn't seem to be a good way of dynamically setting the final answer type
        # to be the answer_cls so we will include it in the prompt

        converted_messages = self._convert_messages_for_openai(messages, base64_image)

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=converted_messages,
            temperature=0.0,
            response_format=answer_cls,
        )

        message = response.choices[0].message
        if message.parsed:
            final_answer = message.parsed.answer
        else:
            final_answer = message.refusal

        return final_answer, messages

    def __str__(self):
        return "Llama_CD"


class Gemini_CD(Gemini):
    def __init__(self, model_name="gemini-1.5-pro", location="us-central1"):
        super().__init__(model_name, location)

    def generate_answer(self, image, messages: list[dict[str, str]]):
        from google.genai import types
        
        image = self.encode_image(image)

        question = messages[-1]["content"]
        response_format = get_answer_class_from_question(question)

        system_instruction, contents = self._prepare_gemini_request(messages, image)

        config_kwargs = {
            "response_mime_type": "application/json",
            "response_schema": response_format,
            "temperature": 0.0,
            "topK": 1,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        answers: Answer = cast(Answer, response.parsed)
        final_answer = answers.answer

        return final_answer, messages

    def __str__(self):
        return self.model + "_CD"


class GPT_CoT_CD(GPT):
    def __init__(self, model_name="gpt-4o", port=None):
        super().__init__(model_name)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
    )
    def generate_answer(self, image, messages: list[dict[str, str]]):
        base64_image = self.encode_image(image)

        converted_messages = self._convert_messages_for_openai(messages, base64_image)
        
        question = messages[-1]["content"]
        # Add the additional system message for CoT_CD
        converted_messages.append({
            "role": "system",
            "content": f"The final_answer should be of type: {get_answer_class_from_question(question).model_json_schema()}",
        })

        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=converted_messages,
            response_format=Reasoning,
            temperature=0.0,
        )

        message = completion.choices[0].message
        if message.parsed:
            output = message.parsed
            output = output.model_dump_json()
        else:
            output = message.refusal

        return output, messages

    def __str__(self):
        return self.model_name + "_CoT_CD"


class Llama_CoT_CD(Llama):
    def __init__(self, model_name="meta-llama/Llama-3.2-90B-Vision-Instruct"):
        super().__init__(model_name, use_vllm=False)

    def generate_answer(self, image, messages: list[dict[str, str]]):
        base64_image = self.encode_image(image)

        question = messages[-1]["content"]
        answer_cls = get_answer_class_from_question(question)
        # There doesn't seem to be a good way of dynamically setting the final answer type
        # to be the answer_cls so we will include it in the prompt

        converted_messages = self._convert_messages_for_openai(messages, base64_image)
        
        # Add the additional system message for CoT_CD
        converted_messages.append({
            "role": "system",
            "content": f"The final_answer should be of type: {answer_cls.model_json_schema()}",
        })

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=converted_messages,
            response_format=Reasoning,
            temperature=0.0,
        )

        message = response.choices[0].message
        if message.parsed:
            final_answer = message.parsed
            final_answer = final_answer.model_dump_json()
        else:
            final_answer = message.refusal

        return final_answer, messages

    def __str__(self):
        return "Llama_CoT_CD"


class Gemini_CoT_CD(Gemini):
    def __init__(self, model_name="gemini-1.5-pro", location="us-central1"):
        super().__init__(model_name, location)

    def generate_answer(self, image, messages: list[dict[str, str]]):
        from google.genai import types
        
        image = self.encode_image(image)

        question = messages[-1]["content"]
        response_format = get_answer_class_from_question(question)

        system_instruction, contents = self._prepare_gemini_request(messages, image)
        
        # Add the schema information for CoT_CD as an extra text element
        contents.append(f"The final_answer should be of type: {response_format.model_json_schema()}")

        config_kwargs = {
            "response_mime_type": "application/json",
            "response_schema": Reasoning,
            "temperature": 0.0,
            "topK": 1,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        reasoning_response: Reasoning = cast(Reasoning, response.parsed)

        return reasoning_response.model_dump_json(), messages

    def __str__(self):
        return self.model + "_CoT_CD"


class Claude(VLM):
    def __init__(self, model_name="claude-3-7-sonnet-20250219"):
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model_name

    def encode_image(self, image):
        if isinstance(image, torch.Tensor):
            np_img = image.mul(255).byte().numpy().transpose(1, 2, 0)
            img_pil = Image.fromarray(np_img)
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        elif isinstance(image, str):
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(image, np.ndarray):
            success, buffer = cv2.imencode(".jpg", image)
            return base64.b64encode(buffer).decode("utf-8")

    def _convert_messages_for_claude(self, messages, base64_image):
        """Convert message list to Claude API format.

        Returns (claude_messages, system_prompt) where system_prompt may be None.
        """
        claude_messages = []
        system_prompt: str | None = None

        # Find the last user message to attach the image
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break

        for i, msg in enumerate(messages):
            role, content = msg["role"], msg["content"]
            if role == "system":
                system_prompt = content if system_prompt is None else f"{system_prompt}\n{content}"
                continue  # system prompt handled separately

            if role in ("user", "assistant"):
                if role == "user" and i == last_user_idx:
                    # Attach image to the last user message
                    claude_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_image,
                                    },
                                },
                                {"type": "text", "text": content},
                            ],
                        }
                    )
                else:
                    # Regular text-only message
                    claude_messages.append({"role": role, "content": content})

        return claude_messages, system_prompt

    def generate_answer(self, image, messages: list[dict[str, str]]):
        base64_image = self.encode_image(image)

        claude_messages, system_prompt = self._convert_messages_for_claude(messages, base64_image)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            system=system_prompt if system_prompt else None,
            messages=claude_messages,
        )

        return response.content[0].text, messages

    def __str__(self) -> str:
        return self.model


class Claude_CD(Claude):
    def __init__(self, model_name="claude-3-7-sonnet-20250219"):
        super().__init__(model_name)

    def generate_answer(self, image, messages: list[dict[str, str]]):
        import anthropic
        from pydantic import create_model

        # Get the answer class based on the question
        question = messages[-1]["content"]
        answer_class = get_answer_class_from_question(question)
        if answer_class is None:
            # Fallback to non-CD generation if no class matches
            return super().generate_answer(image, messages)

        base64_image = self.encode_image(image)

        claude_messages, system_prompt = self._convert_messages_for_claude(messages, base64_image)

        # Use anthropic.messages.create with response_model parameter for constrained decoding
        response = self.client.messages.create(
            model=self.model,
            temperature=0.0,
            system=system_prompt if system_prompt else None,
            messages=claude_messages,
            response_model=answer_class,
        )

        return response.answer, messages

    def __str__(self):
        return self.model + "_CD"


class Claude_CoT_CD(Claude):
    def __init__(self, model_name="claude-3-7-sonnet-20250219"):
        super().__init__(model_name)

    def generate_answer(self, image, messages: list[dict[str, str]]):
        # Get the answer class based on the question
        question = messages[-1]["content"]
        answer_class = get_answer_class_from_question(question)

        base64_image = self.encode_image(image)

        claude_messages, system_prompt = self._convert_messages_for_claude(messages, base64_image)
        
        # Add schema information to the last user message for Claude
        if claude_messages and claude_messages[-1]["role"] == "user":
            schema_text = f"\n\nThe final_answer should be of type: {answer_class.model_json_schema()}"
            if isinstance(claude_messages[-1]["content"], list):
                for content_item in claude_messages[-1]["content"]:
                    if content_item["type"] == "text":
                        content_item["text"] += schema_text
                        break
            else:
                claude_messages[-1]["content"] += schema_text

        response = self.client.messages.create(
            model=self.model,
            temperature=0.0,
            system=system_prompt if system_prompt else None,
            messages=claude_messages,
            response_model=Reasoning,
        )

        return response.model_dump_json(), messages

    def __str__(self):
        return self.model + "_CoT_CD"


# from prompts import ZeroShotPrompt
# model = Claude()
# model.generate_answer("../demo/demo.jpg", "Tell me about this image", prompting_style=ZeroShotPrompt())
# print()
