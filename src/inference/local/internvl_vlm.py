"""
InternVL2 VLM Implementation
"""

from typing import Optional
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from src.inference.local.base import LocalVLM


class InternVLVLM(LocalVLM):
    """InternVL2 implementation with custom image preprocessing"""

    MODEL_ID = "OpenGVLab/InternVL2-2B"
    DISPLAY_NAME = "InternVL2-2B"

    # ImageNet normalization constants
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_id: Optional[str] = None, display_name: Optional[str] = None, use_4bit: bool = True, input_size: int = 448) -> None:
        super().__init__(model_id, display_name, use_4bit)
        self.input_size = input_size
        self._transform: Optional[T.Compose] = None

    def _build_transform(self) -> T.Compose:
        """Build image transform for InternVL2"""
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def load(self) -> None:
        """Load model and initialize transform"""
        super().load()
        self._transform = self._build_transform()

    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer question using InternVL2 format with custom preprocessing"""
        self._check_loaded()

        # Apply transform and move to device
        pixel_values = self._transform(image).unsqueeze(0).to(self.model.device).to(torch.bfloat16)

        with torch.no_grad():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=dict(max_new_tokens=512)
            )

        return response
