"""
Qwen3-VL VLM Implementation

Supports both base model inference and fine-tuned LoRA adapters.
Compatible with models: Qwen3-VL-2B, Qwen3-VL-4B, Qwen3-VL-8B
Also supports Qwen2.5-VL as fallback.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
from PIL import Image
import torch
from transformers import BitsAndBytesConfig

from src.inference.local.base import LocalVLM
from experiments.evaluation.config import VLM_SYSTEM_PROMPT


class Qwen3VLVLM(LocalVLM):
    """
    Qwen3-VL implementation with LoRA adapter support.

    Supports:
    - Base model inference (Qwen3-VL-2B/4B/8B or Qwen2.5-VL-2B/7B)
    - Fine-tuned LoRA adapter loading
    - 4-bit quantization for memory efficiency
    """

    MODEL_ID = ""  # Must be resolved from MODEL_VARIANTS via model_variant arg
    DISPLAY_NAME = "Qwen2-VL-2B"
    USE_CAUSAL_LM = False

    # Available model variants - Verified HuggingFace model IDs
    MODEL_VARIANTS = {
        # Qwen3-VL series (newest, Oct 2025)
        "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
        "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
        "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
        # Qwen2.5-VL series (Jan 2025) - smallest is 3B
        "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        # Qwen2-VL series (older, but has 2B)
        "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    }

    def __init__(
        self,
        model_variant: str = "qwen2-vl-2b",
        lora_adapter_path: Optional[str] = None,
        use_4bit: bool = True,
        display_name: Optional[str] = None,
    ) -> None:
        """
        Initialize Qwen3-VL VLM.

        Args:
            model_variant: One of MODEL_VARIANTS keys or full HF model ID
            lora_adapter_path: Path to fine-tuned LoRA adapter (optional)
            use_4bit: Whether to use 4-bit quantization
            display_name: Custom display name
        """
        # Resolve model ID
        if model_variant in self.MODEL_VARIANTS:
            model_id = self.MODEL_VARIANTS[model_variant]
            default_display = model_variant.replace("-", " ").title().replace(" ", "-")
        else:
            model_id = model_variant
            default_display = model_variant.split("/")[-1]

        super().__init__(
            model_id=model_id,
            display_name=display_name or default_display,
            use_4bit=use_4bit
        )

        assert self.model_id, (
            f"model_id is empty after resolution. "
            f"Pass a valid model_variant key (one of {list(self.MODEL_VARIANTS)}) "
            f"or a full HuggingFace model ID."
        )

        self.lora_adapter_path = lora_adapter_path
        self.processor = None

    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Qwen-VL specific quantization config."""
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
        """Custom device map for Qwen-VL models."""
        if not self.use_4bit:
            return {"": "auto"}

        return {
            "": "cuda:0",
            "visual": "cuda:0",
            "model": "auto",
        }

    def load(self) -> None:
        """Load model with optional LoRA adapter."""
        if self._loaded:
            return

        from transformers import AutoProcessor

        # Try different model classes
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_class = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForVision2Seq
            model_class = AutoModelForVision2Seq

        quantization_config = self.get_quantization_config()
        device_map = self.get_device_map()

        # Load base model
        print(f"Loading {self.model_id}...")
        self.model = model_class.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

        # Load LoRA adapter if specified
        if self.lora_adapter_path:
            self._load_lora_adapter()

        self._loaded = True
        print(f"Model loaded: {self.display_name}")

    def _load_lora_adapter(self) -> None:
        """Load fine-tuned LoRA adapter."""
        from peft import PeftModel

        adapter_path = Path(self.lora_adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

        print(f"Loading LoRA adapter from {adapter_path}...")
        self.model = PeftModel.from_pretrained(
            self.model,
            str(adapter_path),
            is_trainable=False,
        )

        # Merge for faster inference (optional)
        # self.model = self.model.merge_and_unload()

        print("LoRA adapter loaded successfully")

    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer question using Qwen-VL chat format."""
        self._check_loaded()

        # Build conversation
        messages = [
            {
                "role": "system",
                "content": VLM_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode response (skip input tokens)
        input_len = inputs['input_ids'].shape[1]
        response = self.processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True
        )

        return response.strip()

    def unload(self) -> None:
        """Unload model from memory."""
        if self.processor is not None:
            del self.processor
            self.processor = None

        super().unload()


class Qwen3VLFineTuned(Qwen3VLVLM):
    """
    Convenience class for loading fine-tuned Qwen3-VL models.

    Usage:
        model = Qwen3VLFineTuned(
            adapter_path="experiments/fine_tuning/runs/qwen2.5-vl-2b_20241228/final"
        )
        model.load()
        response = model.answer_question(image, "What structures are visible?")
    """

    def __init__(
        self,
        adapter_path: str,
        base_model: str = "qwen2.5-vl-2b",
        use_4bit: bool = True,
    ) -> None:
        """
        Initialize fine-tuned Qwen3-VL.

        Args:
            adapter_path: Path to the fine-tuned LoRA adapter
            base_model: Base model variant to use
            use_4bit: Whether to use 4-bit quantization
        """
        super().__init__(
            model_variant=base_model,
            lora_adapter_path=adapter_path,
            use_4bit=use_4bit,
            display_name=f"Qwen-VL-FineTuned ({Path(adapter_path).parent.name})"
        )
