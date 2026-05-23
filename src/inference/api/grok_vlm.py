"""
Grok VLM - xAI Grok Vision API wrapper implementing VLMInterface
Uses OpenAI-compatible API endpoint
"""

import os
import logging
from typing import List, Optional

from PIL import Image
from dotenv import load_dotenv, find_dotenv

from src.utils.image_processing import to_base64_data_url
from src.utils.api_client import call_with_retry

load_dotenv(find_dotenv('.env.local'))

from openai import OpenAI
from src.inference.vlm_interface import VLMInterface
from experiments.evaluation.config import INTERACTIVE_TEMPERATURE

logger = logging.getLogger(__name__)


class GrokVLM(VLMInterface):
    """xAI Grok Vision API wrapper using OpenAI-compatible endpoint"""

    # Available Grok vision models
    AVAILABLE_MODELS = {
        "grok-4": "Grok 4",
        "grok-4-fast": "Grok 4 Fast",
        "grok-2-vision-latest": "Grok 2 Vision (Latest)",
        "grok-2-vision-1212": "Grok 2 Vision (Dec 2024)",
    }

    # xAI API base URL
    BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        model_name: str = "grok-4",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Grok VLM

        Args:
            model_name: Grok model to use (default: grok-2-vision-latest)
            api_key: API key (defaults to XAI_API_KEY or GROK_API_KEY env var)
            max_retries: Maximum number of retries on API errors
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.model_name_id = model_name
        self.display_name = self.AVAILABLE_MODELS.get(model_name, model_name)
        self.api_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._loaded = False
        self._client = None

        if not self.api_key:
            raise ValueError("XAI_API_KEY or GROK_API_KEY not found in environment variables or .env.local")

    def load(self) -> None:
        """Initialize the OpenAI-compatible client for xAI"""
        if self._loaded:
            return

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL
        )

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

        # Prepare the image as base64 URL using shared utility
        image_url = to_base64_data_url(image)

        # Create the message with medical context
        messages = [
            {
                "role": "system",
                "content": "You are a medical imaging expert analyzing fetal ultrasound images. Provide clear, professional medical responses."
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
            response = self._client.chat.completions.create(
                model=self.model_name_id,
                messages=messages,
                max_tokens=1024,
                temperature=INTERACTIVE_TEMPERATURE
            )

            # Extract text from response
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return "No response generated."

        try:
            return call_with_retry(
                make_request,
                max_retries=self.max_retries,
                base_delay=self.retry_delay,
                on_retry=lambda attempt, e: logger.warning(
                    f"Grok API attempt {attempt + 1} failed: {e}"
                )
            )
        except Exception as e:
            logger.error(f"Grok API failed after {self.max_retries} attempts: {e}")
            raise RuntimeError(f"Grok API failed: {e}")

    @property
    def model_name(self) -> str:
        return self.display_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def create_grok_vlm(model: str = "grok-4") -> GrokVLM:
    """
    Factory function to create a Grok VLM instance

    Args:
        model: Model name to use

    Returns:
        GrokVLM instance
    """
    return GrokVLM(model_name=model)
