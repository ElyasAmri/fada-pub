"""
MiniCPM-V VLM Implementation
"""

from PIL import Image
import torch

from src.inference.local.base import LocalVLM


class MiniCPMVLM(LocalVLM):
    """MiniCPM-V-2.6 implementation"""

    MODEL_ID = "openbmb/MiniCPM-V-2_6"
    DISPLAY_NAME = "MiniCPM-V-2.6"

    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer question using MiniCPM-V chat format"""
        self._check_loaded()

        msgs = [{'role': 'user', 'content': [image, question]}]
        with torch.no_grad():
            response = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer
            )

        return response
