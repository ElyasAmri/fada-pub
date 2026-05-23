"""
Moondream2 VLM Implementation
"""

from typing import Optional
from PIL import Image
import torch

from src.inference.local.base import LocalVLM


class MoondreamVLM(LocalVLM):
    """Moondream2 implementation using AutoModelForCausalLM"""

    MODEL_ID = "vikhyatk/moondream2"
    DISPLAY_NAME = "Moondream2"
    USE_CAUSAL_LM = True  # Moondream uses AutoModelForCausalLM

    def __init__(self, model_id: Optional[str] = None, display_name: Optional[str] = None, use_4bit: bool = False) -> None:
        # Moondream is small enough without quantization by default
        super().__init__(model_id, display_name, use_4bit)

    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer question using Moondream2 format with image embedding"""
        self._check_loaded()

        # Encode image first
        image_embeds = self.model.encode_image(image)

        with torch.no_grad():
            response = self.model.answer_question(
                image_embeds=image_embeds,
                question=question,
                tokenizer=self.tokenizer
            )

        return response
