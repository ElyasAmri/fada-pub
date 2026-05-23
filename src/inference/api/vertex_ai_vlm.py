"""
Vertex AI VLM - Google Vertex AI MedGemma wrapper implementing VLMInterface
Supports MedGemma 4B and 27B multimodal models deployed on Vertex AI
"""

import os
import logging
import requests
from typing import Optional

from PIL import Image
from dotenv import load_dotenv, find_dotenv

from src.utils.image_processing import to_base64_data_url
from src.utils.api_client import call_with_retry
from src.inference.vlm_interface import VLMInterface
from experiments.evaluation.config import INTERACTIVE_TEMPERATURE

load_dotenv(find_dotenv('.env.local'))

import google.auth
import google.auth.transport.requests

# Suppress verbose Google auth logging
logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("google.auth.transport").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class VertexAIVLM(VLMInterface):
    """Google Vertex AI MedGemma wrapper for deployed endpoints"""

    # Available MedGemma models
    AVAILABLE_MODELS = {
        "medgemma-27b-mm-it": "MedGemma 27B Multimodal IT",
        "medgemma-4b-it": "MedGemma 4B IT (Multimodal)",
    }

    def __init__(
        self,
        model_name: str = "medgemma-27b-mm-it",
        project_id: Optional[str] = None,
        location: str = "us-central1",
        endpoint_id: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Vertex AI VLM

        Args:
            model_name: Model identifier (for display purposes)
            project_id: GCP Project ID (defaults to VERTEX_AI_PROJECT_ID env var)
            location: GCP region (defaults to us-central1)
            endpoint_id: Deployed endpoint ID (defaults to VERTEX_AI_ENDPOINT_ID env var)
            max_retries: Maximum number of retries on API errors
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.model_name_id = model_name
        self.display_name = self.AVAILABLE_MODELS.get(model_name, model_name)
        self.project_id = project_id or os.getenv("VERTEX_AI_PROJECT_ID")
        self.location = location or os.getenv("VERTEX_AI_LOCATION", "us-central1")
        self.endpoint_id = endpoint_id or os.getenv("VERTEX_AI_ENDPOINT_ID")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._loaded = False
        self._credentials = None
        self._endpoint_url = None

        if not self.project_id:
            raise ValueError("VERTEX_AI_PROJECT_ID not found in environment variables or .env.local")
        if not self.endpoint_id:
            raise ValueError("VERTEX_AI_ENDPOINT_ID not found in environment variables or .env.local")

    def load(self) -> None:
        """Initialize credentials and build endpoint URL"""
        if self._loaded:
            return

        # Get Application Default Credentials
        self._credentials, _ = google.auth.default()

        # Build endpoint URL
        self._endpoint_url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"endpoints/{self.endpoint_id}:predict"
        )

        logger.info(f"Vertex AI endpoint: {self._endpoint_url}")
        self._loaded = True

    def unload(self) -> None:
        """Release resources"""
        self._credentials = None
        self._endpoint_url = None
        self._loaded = False

    def _get_access_token(self) -> str:
        """Get fresh access token, refreshing if needed"""
        auth_req = google.auth.transport.requests.Request()
        self._credentials.refresh(auth_req)
        return self._credentials.token

    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        Answer a single question about an image using Vertex AI endpoint

        Args:
            image: PIL Image
            question: Question text

        Returns:
            Answer text
        """
        if not self._loaded:
            self.load()

        # Convert image to base64 data URL
        image_url = to_base64_data_url(image)

        # Create the prompt with medical context
        system_prompt = """You are a medical imaging expert analyzing a fetal ultrasound image.
Provide clear, professional medical responses based on what you observe in the image."""

        # Build request body in vLLM OpenAI-compatible format
        request_body = {
            "instances": [
                {
                    "@requestFormat": "chatCompletions",
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": question
                                }
                            ]
                        }
                    ],
                    "max_tokens": 1024,
                    "temperature": INTERACTIVE_TEMPERATURE
                }
            ]
        }

        def make_request():
            # Get fresh token
            token = self._get_access_token()

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                self._endpoint_url,
                headers=headers,
                json=request_body,
                timeout=120
            )

            if response.status_code != 200:
                error_msg = f"Vertex AI API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            result = response.json()
            return self._extract_response_text(result)

        try:
            return call_with_retry(
                make_request,
                max_retries=self.max_retries,
                base_delay=self.retry_delay,
                on_retry=lambda attempt, e: logger.warning(
                    f"Vertex AI API attempt {attempt + 1} failed: {e}"
                )
            )
        except Exception as e:
            logger.error(f"Vertex AI API failed after {self.max_retries} attempts: {e}")
            raise RuntimeError(f"Vertex AI API failed: {e}")

    def _extract_response_text(self, response: dict) -> str:
        """Extract text from Vertex AI response"""
        try:
            # vLLM chat completions format
            predictions = response.get("predictions", [])
            if predictions:
                choices = predictions[0].get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    if content:
                        return content.strip()

            # Fallback: try direct text extraction
            if "predictions" in response and response["predictions"]:
                pred = response["predictions"][0]
                if isinstance(pred, str):
                    return pred.strip()
                if isinstance(pred, dict) and "content" in pred:
                    return pred["content"].strip()

            logger.warning(f"Unexpected response format: {response}")
            return "No response generated."

        except Exception as e:
            logger.error(f"Error extracting response: {e}")
            return f"Error parsing response: {e}"

    @property
    def model_name(self) -> str:
        return self.display_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def create_vertex_ai_vlm(
    model: str = "medgemma-27b-mm-it",
    project_id: Optional[str] = None,
    location: str = "us-central1",
    endpoint_id: Optional[str] = None
) -> VertexAIVLM:
    """
    Factory function to create a Vertex AI VLM instance

    Args:
        model: Model name identifier
        project_id: GCP Project ID
        location: GCP region
        endpoint_id: Deployed endpoint ID

    Returns:
        VertexAIVLM instance
    """
    return VertexAIVLM(
        model_name=model,
        project_id=project_id,
        location=location,
        endpoint_id=endpoint_id
    )
