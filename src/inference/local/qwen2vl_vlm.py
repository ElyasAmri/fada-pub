"""
Qwen2-VL VLM Implementation
"""

from typing import Optional, Dict, Any
from PIL import Image
import torch
from transformers import BitsAndBytesConfig

from src.inference.local.base import LocalVLM


class Qwen2VLVLM(LocalVLM):
    """Qwen2-VL implementation with custom quantization for CPU offload"""

    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    DISPLAY_NAME = "Qwen2-VL-2B"

    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Qwen2-VL needs CPU offload for some modules"""
        if not self.use_4bit:
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

    def get_device_map(self) -> Dict[str, Any]:
        """Custom device map to allow CPU offload"""
        if not self.use_4bit:
            return {"": "auto"}

        return {
            "": "cuda:0",
            "visual": "cuda:0",
            "model": "auto",
        }

    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer question using Qwen2-VL chat format"""
        self._check_loaded()

        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]}]

        with torch.no_grad():
            response = self.model.chat(
                messages=msgs,
                tokenizer=self.tokenizer
            )

        return response
