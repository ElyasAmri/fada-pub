"""
Gemini VLM - Google Gemini Vision API wrapper implementing VLMInterface
Supports Gemini 2.0 Flash and other vision-capable models
"""

import os
import logging
from typing import List, Optional

from PIL import Image
from dotenv import load_dotenv, find_dotenv

from src.utils.image_processing import to_base64
from src.utils.api_client import call_with_retry

load_dotenv(find_dotenv('.env.local'))

import google.generativeai as genai
from src.inference.vlm_interface import VLMInterface
from experiments.evaluation.config import API_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class GeminiVLM(VLMInterface):
    """Google Gemini Vision API wrapper"""

    # Available Gemini vision models
    AVAILABLE_MODELS = {
        "gemini-3-pro-preview": "Gemini 3 Pro (Preview)",
        "gemini-2.5-pro-exp-03-25": "Gemini 2.5 Pro (Experimental)",
        "gemini-2.5-flash": "Gemini 2.5 Flash",
        "gemini-2.0-flash-exp": "Gemini 2.0 Flash (Experimental)",
        "gemini-1.5-flash": "Gemini 1.5 Flash",
        "gemini-1.5-pro": "Gemini 1.5 Pro",
    }

    def __init__(
        self,
        model_name: str = "gemini-3-pro-preview",
        api_key: Optional[str] = None,
        thinking_level: str = "low",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.4
    ):
        """
        Initialize Gemini VLM

        Args:
            model_name: Gemini model to use (default: gemini-3-pro-preview)
            api_key: API key (defaults to GEMINI_API_KEY env var)
            thinking_level: Thinking level for reasoning models ("none", "low", "medium", "high")
            max_retries: Maximum number of retries on API errors
            retry_delay: Initial delay between retries (exponential backoff)
            temperature: Sampling temperature for generation
        """
        self.model_name_id = model_name
        self.display_name = self.AVAILABLE_MODELS.get(model_name, model_name)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.thinking_level = thinking_level
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature

        self._loaded = False
        self._model = None

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or .env.local")

    def load(self) -> None:
        """Initialize the Gemini client and model"""
        if self._loaded:
            return

        genai.configure(api_key=self.api_key)

        # Configure generation settings
        generation_config = {
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }

        # Note: thinking_config not supported by google-generativeai SDK yet
        # Gemini 3 Pro will use default thinking behavior

        self._model = genai.GenerativeModel(
            model_name=self.model_name_id,
            generation_config=generation_config
        )

        self._loaded = True

    def unload(self) -> None:
        """Release resources"""
        self._model = None
        self._loaded = False

    def _image_to_part(self, image: Image.Image) -> dict:
        """Convert PIL Image to Gemini image part"""
        return {
            "mime_type": "image/jpeg",
            "data": to_base64(image)
        }

    def _extract_response_text(self, response) -> str:
        """Extract text from Gemini response, handling various formats."""
        # Try the simple accessor first
        try:
            if response.text:
                return response.text.strip()
        except ValueError:
            # response.text throws ValueError if blocked or has multiple parts
            pass

        # Try extracting from parts
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                parts = candidate.content.parts
                text_parts = []
                for part in parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    return "\n".join(text_parts).strip()

            # Check for block reason
            if hasattr(candidate, 'finish_reason'):
                finish_reason = str(candidate.finish_reason)
                if 'SAFETY' in finish_reason or 'BLOCKED' in finish_reason:
                    return f"Response blocked: {finish_reason}"

        # Check prompt feedback
        if hasattr(response, 'prompt_feedback'):
            feedback = response.prompt_feedback
            if hasattr(feedback, 'block_reason') and feedback.block_reason:
                return f"Prompt blocked: {feedback.block_reason}"

        return "No response generated."

    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        Answer a single question about an image

        Args:
            image: PIL Image
            question: Question text

        Returns:
            Answer text
        """
        if not self._loaded:
            self.load()

        # Prepare the image part
        image_part = self._image_to_part(image)

        # Create the prompt with medical context
        prompt = f"""You are a medical imaging expert analyzing a fetal ultrasound image.
Please answer the following question about this ultrasound image:

{question}

Provide a clear, professional medical response."""

        def make_request():
            response = self._model.generate_content([
                {"inline_data": image_part},
                prompt
            ])
            return self._extract_response_text(response)

        try:
            return call_with_retry(
                make_request,
                max_retries=self.max_retries,
                base_delay=self.retry_delay,
                on_retry=lambda attempt, e: logger.warning(
                    f"Gemini API attempt {attempt + 1} failed: {e}"
                )
            )
        except Exception as e:
            logger.error(f"Gemini API failed after {self.max_retries} attempts: {e}")
            raise RuntimeError(f"Gemini API failed: {e}")

    @property
    def model_name(self) -> str:
        return self.display_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def create_gemini_vlm(
    model: str = "gemini-3-pro-preview",
    thinking_level: str = "low"
) -> GeminiVLM:
    """
    Factory function to create a Gemini VLM instance

    Args:
        model: Model name to use
        thinking_level: Thinking level ("none", "low", "medium", "high")

    Returns:
        GeminiVLM instance
    """
    return GeminiVLM(model_name=model, thinking_level=thinking_level)
