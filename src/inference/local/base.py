"""
Local VLM Base Class - Base implementation for local GPU-based VLM inference
"""

from typing import Optional, Union, Dict, Any
from PIL import Image
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.inference.vlm_interface import VLMInterface


class LocalVLM(VLMInterface):
    """Base class for local GPU-based VLM inference using HuggingFace models"""

    # Override in subclasses
    MODEL_ID: str = ""
    DISPLAY_NAME: str = ""
    USE_CAUSAL_LM: bool = False  # True for models using AutoModelForCausalLM

    def __init__(self, model_id: Optional[str] = None, display_name: Optional[str] = None, use_4bit: bool = True) -> None:
        """
        Initialize local VLM

        Args:
            model_id: HuggingFace model ID (defaults to class MODEL_ID)
            display_name: Display name for UI (defaults to class DISPLAY_NAME)
            use_4bit: Whether to use 4-bit quantization
        """
        self.model_id = model_id or self.MODEL_ID
        self.display_name = display_name or self.DISPLAY_NAME
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization config. Override in subclasses for custom configs."""
        if not self.use_4bit:
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

    def get_device_map(self) -> Union[str, Dict[str, Any]]:
        """Get device map. Override in subclasses for custom configs."""
        return "auto"

    def load(self) -> None:
        """Load model with quantization"""
        if self._loaded:
            return

        quantization_config = self.get_quantization_config()
        device_map = self.get_device_map()

        if self.USE_CAUSAL_LM:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map=device_map if self.use_4bit else "cuda",
                torch_dtype=torch.float16 if not self.use_4bit else None
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map=device_map,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                _attn_implementation="eager"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self._loaded = True

    def unload(self) -> None:
        """Unload model from memory"""
        self._loaded = False
        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _check_loaded(self) -> None:
        """Check if model is loaded and raise if not"""
        if not self._loaded:
            raise RuntimeError(f"Model {self.display_name} is not loaded")
        if self.model is None:
            raise RuntimeError(f"Model {self.display_name} loaded but model object is None")
        if self.tokenizer is None:
            raise RuntimeError(f"Model {self.display_name} loaded but tokenizer is None")

    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer single question - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement answer_question()")

    @property
    def model_name(self) -> str:
        return self.display_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded
