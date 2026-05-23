"""
OpenAI Client Wrapper - Unified API call handling with error management
"""

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('.env.local'))

from openai import OpenAI
from .model_config import (
    get_model_config,
    validate_model,
    DEFAULT_MODEL,
    MEDICAL_TEMPERATURE,
    MEDICAL_MAX_TOKENS
)

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper for OpenAI API with unified error handling"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = MEDICAL_TEMPERATURE,
        max_tokens: int = MEDICAL_MAX_TOKENS
    ):
        """
        Initialize OpenAI client

        Args:
            api_key: OpenAI API key (uses env var if not provided)
            model: GPT model to use
            temperature: Response randomness
            max_tokens: Maximum response length
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            self.client = None
            return

        self.client = OpenAI(api_key=self.api_key)

        # Get model configuration from environment or use defaults
        self.model = os.getenv("MODEL_NAME", model)
        self.temperature = float(os.getenv("TEMPERATURE", str(temperature)))
        self.max_tokens = int(os.getenv("MAX_TOKENS", str(max_tokens)))

        # Validate model
        if not validate_model(self.model):
            logger.warning(f"Unknown model {self.model}, falling back to {DEFAULT_MODEL}")
            self.model = DEFAULT_MODEL

        # Log configuration
        model_config = get_model_config(self.model)
        if model_config:
            logger.info(f"OpenAI configured with {model_config.display_name}")
            logger.info(f"  Context window: {model_config.context_window:,} tokens")

    def is_available(self) -> bool:
        """Check if API is configured and available"""
        return self.client is not None

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Make API call with unified error handling

        Args:
            messages: List of message dicts with role and content
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Response text or None if call fails
        """
        if not self.is_available():
            logger.warning("OpenAI API not available")
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
