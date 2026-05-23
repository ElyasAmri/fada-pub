"""
OpenAI VLM - GPT-5.2 Vision API wrapper implementing VLMInterface
Supports GPT-5.2, GPT-4o and other vision-capable models
"""

import os
import logging
from typing import Optional

from PIL import Image
from dotenv import load_dotenv, find_dotenv

from src.utils.image_processing import to_base64_data_url
from src.utils.api_client import call_with_retry

load_dotenv(find_dotenv('.env.local'))

from openai import OpenAI
from src.inference.vlm_interface import VLMInterface
from experiments.evaluation.config import API_SYSTEM_PROMPT, INTERACTIVE_TEMPERATURE

logger = logging.getLogger(__name__)


class OpenAIVLM(VLMInterface):
    """OpenAI Vision API wrapper for GPT-5.2 and GPT-4o models"""

    # Available vision models
    AVAILABLE_MODELS = {
        # GPT-5.2 series (latest)
        "gpt-5.2": "GPT-5.2 Thinking",
        "gpt-5.2-chat-latest": "GPT-5.2 Instant",
        # GPT-5.1 series
        "gpt-5.1": "GPT-5.1 Thinking",
        "gpt-5.1-chat-latest": "GPT-5.1 Instant",
        # GPT-4o series
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "GPT-4o Mini",
        "gpt-4o-2024-11-20": "GPT-4o (Nov 2024)",
        # GPT-4.1 series
        "gpt-4.1": "GPT-4.1",
        "gpt-4.1-mini": "GPT-4.1 Mini",
    }

    def __init__(
        self,
        model_name: str = "gpt-5.2-chat-latest",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize OpenAI VLM

        Args:
            model_name: Model to use (default: gpt-5.2-chat-latest for speed)
            api_key: API key (defaults to OPENAI_API_KEY env var)
            max_retries: Maximum number of retries on API errors
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.model_name_id = model_name
        self.display_name = self.AVAILABLE_MODELS.get(model_name, model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._loaded = False
        self._client = None

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or .env.local")

    def load(self) -> None:
        """Initialize the OpenAI client"""
        if self._loaded:
            return

        self._client = OpenAI(api_key=self.api_key)
        self._loaded = True

    def unload(self) -> None:
        """Release resources"""
        self._client = None
        self._loaded = False

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

        # Prepare the image as base64 URL
        image_url = to_base64_data_url(image)

        # Create the message with medical context
        messages = [
            {
                "role": "system",
                "content": API_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]

        def make_request():
            # GPT-5.x models use max_completion_tokens and don't support custom temperature
            if self.model_name_id.startswith("gpt-5"):
                response = self._client.chat.completions.create(
                    model=self.model_name_id,
                    messages=messages,
                    max_completion_tokens=1024
                )
            else:
                response = self._client.chat.completions.create(
                    model=self.model_name_id,
                    messages=messages,
                    max_tokens=1024,
                    temperature=INTERACTIVE_TEMPERATURE
                )

            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return "No response generated."

        try:
            return call_with_retry(
                make_request,
                max_retries=self.max_retries,
                base_delay=self.retry_delay,
                on_retry=lambda attempt, e: logger.warning(
                    f"OpenAI API attempt {attempt + 1} failed: {e}"
                )
            )
        except Exception as e:
            logger.error(f"OpenAI API failed after {self.max_retries} attempts: {e}")
            raise RuntimeError(f"OpenAI API failed: {e}")

    @property
    def model_name(self) -> str:
        return self.display_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def create_openai_vlm(model: str = "gpt-5.2-chat-latest") -> OpenAIVLM:
    """
    Factory function to create an OpenAI VLM instance

    Args:
        model: Model name to use (default: gpt-5.2-chat-latest)

    Returns:
        OpenAIVLM instance
    """
    return OpenAIVLM(model_name=model)
