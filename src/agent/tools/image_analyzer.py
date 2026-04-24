from enum import Enum
import json
from loguru import logger
import mlflow
from mlflow.entities import SpanType
from pydantic import BaseModel
from typing import Any

from src.data.user_image_store import UserImageStore
from src.agent.chat_assistant_factory import ChatAssistantFactory
from src.agent.prompts.image_analysis import (
    IMAGE_ANALYSIS_VQA_SYSTEM_INSTRUCTION, 
    IMAGE_ANALYSIS_IC_SYSTEM_INSTRUCTION,
    IMAGE_ANALYSIS_OCR_SYSTEM_INSTRUCTION,
    IMAGE_ANALYSIS_OD_SYSTEM_INSTRUCTION,
)
from src.agent.chat_assistant import ChatAssistant

class ImageAnalysisTask(str, Enum):
    VQA = "vqa" #Visual Question Answering
    IC = "ic"   #Image Captioning
    OCR = "ocr" #Optical Character Recognition
    OD = "od"   #Object Detection

class ImageAnalyzer:
    def __init__(self):
        self._user_image_store = UserImageStore()
        self._factory = ChatAssistantFactory()
    
    def _get_image_analysis_assistant(self, task: ImageAnalysisTask) -> ChatAssistant:
        match task:
            case ImageAnalysisTask.VQA:
                system_instruction = IMAGE_ANALYSIS_VQA_SYSTEM_INSTRUCTION
            case ImageAnalysisTask.IC:
                system_instruction = IMAGE_ANALYSIS_IC_SYSTEM_INSTRUCTION
            case ImageAnalysisTask.OCR:
                system_instruction = IMAGE_ANALYSIS_OCR_SYSTEM_INSTRUCTION
            case ImageAnalysisTask.OD:
                system_instruction = IMAGE_ANALYSIS_OD_SYSTEM_INSTRUCTION
            case _:
                raise ValueError(f"Unsupported task type: {task}")

        assistant, _ = self._factory.get_or_create_assistant(
            assistant_name="Image Analyzer",
            model_name=None,  # Use the default model
            system_instruction=system_instruction,
            available_tools=None,  # No tools!
            session=None,  # Start a new session
        )
        return assistant
    
    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def answer_question_about_image_in_agentic_image(self, question: str, user_image_id: str) -> str:
        """
        Generates an answer to the given question about the user-provided image specified by `user_image_id`.
        In other words, this function performs Visual Question Answering (VQA) on a user image.

        Args:
            question: The question to be answered.
            user_image_id: The ID of the user image to analyze.

        Returns:
            The answer to the question.
        """

        base64_image = self._user_image_store.load_user_image(user_image_id, base64=True)
        record = {
            "source": "user_image",
            "user_image_id": user_image_id,
        }

        try:
            user_prompt = self.generate_vqa_prompt(record, question)
            assistant = self._get_image_analysis_assistant(ImageAnalysisTask.VQA)
            response_text = assistant.send_user_message(user_prompt,base64_image)
            return response_text
        
        except Exception as e:
            msg = f"Error performing VQA: {str(e)}"
            logger.error(msg)
            return msg

    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def generate_caption_for_fundus_record_image(self, user_image_id: str) -> str:
        pass
        #TODO: wether we need to generate the caption?

    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def extract_text_from_image_record(self, user_image_id: str) -> str:
        """
        Perform OCR on the user-provided image specified by the given `user_image_id`.

        Args:
            user_image_id: The ID of the user image to perform OCR on.

        Returns:
            The extracted text from the image.
        """
        base64_image = self._user_image_store.load_user_image(user_image_id, base64=True)
        record = {
            "source": "user_image",
            "user_image_id": user_image_id,
        }

        try:
            user_prompt = self.generate_ocr_prompt(record)
            assistant = self._get_image_analysis_assistant(ImageAnalysisTask.OCR)
            response_text = assistant.send_user_message(user_prompt, base64_image)
            return response_text

        except Exception as e:
            msg = f"Error performing OCR: {e}"
            logger.error(msg)
            return msg

    @mlflow.trace(
        span_type=SpanType.TOOL,
    )
    def detect_objects_in_image_record(self, user_image_id: str) -> str:
        """
        Perform object detection on the user-provided image specified by the given `user_image_id`.

        Args:
            user_image_id: The ID of the user image to perform object detection on.

        Returns:
            The detected objects in the image.
        """

        base64_image = self._user_image_store.load_user_image(user_image_id, base64=True)
        record = {
            "source": "user_image",
            "user_image_id": user_image_id,
        }

        try:
            user_prompt = self.generate_od_prompt(record)
            assistant = self._get_image_analysis_assistant(ImageAnalysisTask.OD)
            response_text = assistant.send_user_message(user_prompt, base64_image)
            return response_text

        except Exception as e:
            msg = f"Error performing object detection: {e}"
            logger.error(msg)
            return msg


    @staticmethod
    def _serialize_record(record: BaseModel | dict[str, Any]) -> str:
        # Only include metadata fields to keep the prompt compact and deterministic.
        if isinstance(record, BaseModel):
            payload: dict[str, Any] = record.model_dump(mode="json")
        else:
            payload = record
        return json.dumps(payload, indent=2)

    @staticmethod
    def generate_vqa_prompt(record: BaseModel | dict[str, Any], question: str) -> str:
        prompt = "# Question\n\n"
        prompt += f"'''\n{question}\n'''\n\n"
        prompt += "# Metadata as JSON\n\n"
        prompt += f"```json\n{ImageAnalyzer._serialize_record(record)}\n```\n"
        return prompt
    
    @staticmethod
    def generate_od_prompt(record: BaseModel | dict[str, Any]) -> str:
        prompt = "Detect all objects in the image considering the metadata.\n\n"
        prompt += "# Metadata as JSON\n\n"
        prompt += f"```json\n{ImageAnalyzer._serialize_record(record)}\n```\n"
        return prompt
    
    @staticmethod
    def generate_ocr_prompt(record: BaseModel | dict[str, Any]) -> str:
        prompt = "Extract all text in the image considering the metadata.\n\n"
        prompt += "# Metadata as JSON\n\n"
        prompt += f"```json\n{ImageAnalyzer._serialize_record(record)}\n```\n"
        return prompt
