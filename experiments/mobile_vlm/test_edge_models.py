"""
Mobile VLM Benchmarking Infrastructure

Benchmarks edge VLMs for mobile/edge deployment feasibility.
Tracks inference latency, memory usage, and accuracy on fetal ultrasound Q7 task.

Supported Models:
- Moondream2 (1.8B) - Ultra-lightweight, <2GB RAM
- Qwen2.5-VL-3B (3B) - Best balance of capability/size
- Qwen2.5-VL-7B (7B) - Higher capability, ~8GB RAM
- MobileVLM-V2 (3B) - Optimized LDPv2 projector
- Pixtral-12B (12B) - Mistral's multimodal model
- DeepSeek-VL2-Tiny (3B) - Low-latency MoE model
- DeepSeek-VL2-Small (16B) - Mid-size MoE model
- NVILA-Lite (8B) - NVIDIA's efficient VLM

Usage:
    python test_edge_models.py --model moondream2 --device cuda --samples 10
    python test_edge_models.py --model qwen25-vl-3b --device cpu --quantization int8
    python test_edge_models.py --model deepseek-vl2-tiny --device cuda --quantization int4
    python test_edge_models.py --list-models  # Show all available models
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
import mlflow

# Model configurations - All SoTA mobile/edge VLMs (2025-2026)
MODEL_CONFIGS = {
    # Ultra-lightweight (<2GB)
    "moondream2": {
        "model_id": "vikhyatk/moondream2",
        "type": "moondream",
        "params": "1.8B",
        "ram": "<2GB",
        "description": "Ultra-lightweight, best for extreme edge constraints",
    },
    # Qwen2.5-VL Series (3B-7B)
    "qwen25-vl-3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "type": "qwen",
        "params": "3B",
        "ram": "~4GB",
        "description": "Best balance of capability and size",
    },
    "qwen25-vl-7b": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "type": "qwen",
        "params": "7B",
        "ram": "~8GB",
        "description": "Higher capability, good for GPU deployment",
    },
    # MobileVLM V2 Series
    "mobilevlm-v2-1.7b": {
        "model_id": "mtgv/MobileVLM_V2-1.7B",
        "type": "mobilevlm",
        "params": "1.7B",
        "ram": "~3GB",
        "description": "Optimized LDPv2 projector, very efficient",
    },
    "mobilevlm-v2-3b": {
        "model_id": "mtgv/MobileVLM_V2-3B",
        "type": "mobilevlm",
        "params": "3B",
        "ram": "~4GB",
        "description": "MobileVLM with better accuracy",
    },
    "mobilevlm-v2-7b": {
        "model_id": "mtgv/MobileVLM_V2-7B",
        "type": "mobilevlm",
        "params": "7B",
        "ram": "~8GB",
        "description": "Largest MobileVLM variant",
    },
    # Pixtral (Mistral)
    "pixtral-12b": {
        "model_id": "mistralai/Pixtral-12B-2409",
        "type": "pixtral",
        "params": "12B",
        "ram": "~12GB",
        "description": "Mistral's multimodal, strong reasoning",
    },
    # DeepSeek-VL2 Series (MoE)
    "deepseek-vl2-tiny": {
        "model_id": "deepseek-ai/deepseek-vl2-tiny",
        "type": "deepseek",
        "params": "3B (MoE)",
        "ram": "~4GB",
        "description": "Low-latency MoE, good for scientific data",
    },
    "deepseek-vl2-small": {
        "model_id": "deepseek-ai/deepseek-vl2-small",
        "type": "deepseek",
        "params": "16B (MoE)",
        "ram": "~10GB",
        "description": "Mid-size MoE with high capability",
    },
    # NVILA (NVIDIA)
    "nvila-lite": {
        "model_id": "nvidia/NVILA-Lite-2B-hf",
        "type": "nvila",
        "params": "2B",
        "ram": "~3GB",
        "description": "NVIDIA's efficient VLM for edge",
    },
    "nvila-8b": {
        "model_id": "nvidia/NVILA-8B-hf",
        "type": "nvila",
        "params": "8B",
        "ram": "~10GB",
        "description": "NVIDIA's full VLM",
    },
    # InternVL2 Series (lightweight variants)
    "internvl2-1b": {
        "model_id": "OpenGVLab/InternVL2-1B",
        "type": "internvl",
        "params": "1B",
        "ram": "~2GB",
        "description": "Ultra-compact InternVL variant",
    },
    "internvl2-2b": {
        "model_id": "OpenGVLab/InternVL2-2B",
        "type": "internvl",
        "params": "2B",
        "ram": "~3GB",
        "description": "Small but capable InternVL",
    },
    "internvl2-4b": {
        "model_id": "OpenGVLab/InternVL2-4B",
        "type": "internvl",
        "params": "4B",
        "ram": "~5GB",
        "description": "Balanced InternVL for edge GPU",
    },
    # PaliGemma (Google)
    "paligemma-3b": {
        "model_id": "google/paligemma-3b-pt-224",
        "type": "paligemma",
        "params": "3B",
        "ram": "~4GB",
        "description": "Google's efficient multimodal model",
    },
    # Phi-3 Vision
    "phi3-vision": {
        "model_id": "microsoft/Phi-3-vision-128k-instruct",
        "type": "phi3",
        "params": "4.2B",
        "ram": "~5GB",
        "description": "Microsoft's efficient vision-language model",
    },
}

# Test prompt for fetal ultrasound Q7 task
Q7_PROMPT = "Describe any abnormalities visible in this fetal ultrasound image."


class ModelBenchmark:
    """Handles model loading, inference, and benchmarking."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        quantization: str = "none",
    ):
        self.model_name = model_name
        self.device = device
        self.quantization = quantization

        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
            )

        self.config = MODEL_CONFIGS[model_name]
        self.model = None
        self.processor = None
        self.tokenizer = None

        # Metrics
        self.load_time = 0.0
        self.inference_times = []
        self.memory_usage = 0.0

    def load_model(self) -> float:
        """
        Load model with specified configuration.

        Returns:
            Load time in seconds
        """
        print(f"\nLoading {self.model_name}...")
        print(f"  Model ID: {self.config['model_id']}")
        print(f"  Parameters: {self.config.get('params', 'unknown')}")
        print(f"  Device: {self.device}")
        print(f"  Quantization: {self.quantization}")

        start_time = time.time()

        try:
            model_type = self.config["type"]

            if model_type == "moondream":
                self._load_moondream()
            elif model_type == "qwen":
                self._load_qwen()
            elif model_type == "mobilevlm":
                self._load_mobilevlm()
            elif model_type == "pixtral":
                self._load_pixtral()
            elif model_type == "deepseek":
                self._load_deepseek()
            elif model_type == "nvila":
                self._load_nvila()
            elif model_type == "internvl":
                self._load_internvl()
            elif model_type == "paligemma":
                self._load_paligemma()
            elif model_type == "phi3":
                self._load_phi3()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.load_time = time.time() - start_time
            print(f"  Load time: {self.load_time:.2f}s")

            # Measure memory usage
            if self.device == "cuda":
                torch.cuda.synchronize()
                self.memory_usage = torch.cuda.memory_allocated() / (1024 ** 2)
                print(f"  GPU memory: {self.memory_usage:.2f} MB")

            return self.load_time

        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def _load_moondream(self):
        """Load Moondream2 model via transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.config["model_id"]
        quantization_config = self._get_quantization_config()

        # Moondream2 uses AutoModelForCausalLM with trust_remote_code
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if self.device == "cpu" and quantization_config is None:
            self.model = self.model.to("cpu")

        self.model.eval()

    def _load_qwen(self):
        """Load Qwen2/2.5-VL model."""
        from transformers import AutoModelForVision2Seq, AutoProcessor

        model_id = self.config["model_id"]
        quantization_config = self._get_quantization_config()

        # Use AutoModelForVision2Seq for compatibility with both Qwen2-VL and Qwen2.5-VL
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        if self.device == "cpu" and quantization_config is None:
            self.model = self.model.to("cpu")

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()

    def _load_mobilevlm(self):
        """Load MobileVLM V2 model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

        model_id = self.config["model_id"]
        quantization_config = self._get_quantization_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()

    def _load_pixtral(self):
        """Load Pixtral model (Mistral's multimodal)."""
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        model_id = self.config["model_id"]
        quantization_config = self._get_quantization_config()

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()

    def _load_deepseek(self):
        """Load DeepSeek-VL2 model."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = self.config["model_id"]
        quantization_config = self._get_quantization_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()

    def _load_nvila(self):
        """Load NVILA model (NVIDIA's VLM)."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = self.config["model_id"]
        quantization_config = self._get_quantization_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()

    def _load_internvl(self):
        """Load InternVL2 model."""
        from transformers import AutoModel, AutoTokenizer

        model_id = self.config["model_id"]
        quantization_config = self._get_quantization_config()

        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()

    def _load_paligemma(self):
        """Load PaliGemma model (Google)."""
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

        model_id = self.config["model_id"]
        quantization_config = self._get_quantization_config()

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()

    def _load_phi3(self):
        """Load Phi-3 Vision model (Microsoft)."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = self.config["model_id"]
        quantization_config = self._get_quantization_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
            _attn_implementation="eager",  # Phi-3 specific
        )

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()

    def _get_quantization_config(self):
        """Get quantization config based on settings."""
        if self.quantization == "int8":
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "int4":
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        return None

    def run_inference(
        self,
        image_path: Path,
        prompt: str = Q7_PROMPT,
        warmup: bool = False,
    ) -> Tuple[str, float]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to image file
            prompt: Text prompt
            warmup: If True, don't track timing (warmup run)

        Returns:
            Tuple of (response text, inference time in ms)
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Run inference
        start_time = time.time()

        try:
            model_type = self.config["type"]

            if model_type == "moondream":
                response = self._inference_moondream(image, prompt)
            elif model_type == "qwen":
                response = self._inference_qwen(image, prompt)
            elif model_type == "mobilevlm":
                response = self._inference_mobilevlm(image, prompt)
            elif model_type == "pixtral":
                response = self._inference_pixtral(image, prompt)
            elif model_type == "deepseek":
                response = self._inference_deepseek(image, prompt)
            elif model_type == "nvila":
                response = self._inference_nvila(image, prompt)
            elif model_type == "internvl":
                response = self._inference_internvl(image, prompt)
            elif model_type == "paligemma":
                response = self._inference_paligemma(image, prompt)
            elif model_type == "phi3":
                response = self._inference_phi3(image, prompt)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            if not warmup:
                self.inference_times.append(inference_time)

            return response, inference_time

        except Exception as e:
            raise RuntimeError(f"Inference failed on {image_path}: {e}")

    def _inference_moondream(self, image: Image.Image, prompt: str) -> str:
        """Run Moondream inference."""
        # Moondream2 uses encode_image + answer_question pattern
        with torch.no_grad():
            enc_image = self.model.encode_image(image)
            response = self.model.answer_question(enc_image, prompt, self.tokenizer)
        return response

    def _inference_qwen(self, image: Image.Image, prompt: str) -> str:
        """Run Qwen inference."""
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        # Decode - handle both tensor and dict inputs
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        generated_ids = output_ids[:, input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return response

    def _inference_mobilevlm(self, image: Image.Image, prompt: str) -> str:
        """Run MobileVLM inference."""
        # MobileVLM uses LLaVA-style format
        inputs = self.processor(
            text=f"<image>\n{prompt}",
            images=image,
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        response = self.processor.decode(output_ids[0], skip_special_tokens=True)
        # Extract response after the prompt
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        return response

    def _inference_pixtral(self, image: Image.Image, prompt: str) -> str:
        """Run Pixtral inference."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        response = self.processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def _inference_deepseek(self, image: Image.Image, prompt: str) -> str:
        """Run DeepSeek-VL2 inference."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor(
            conversations=conversation,
            images=[image],
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        response = self.processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def _inference_nvila(self, image: Image.Image, prompt: str) -> str:
        """Run NVILA inference."""
        conversation = [
            {
                "role": "user",
                "content": f"<image>\n{prompt}",
            }
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        response = self.processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def _inference_internvl(self, image: Image.Image, prompt: str) -> str:
        """Run InternVL2 inference."""
        # InternVL uses a specific chat format
        pixel_values = self._preprocess_internvl_image(image)

        if self.device == "cuda":
            pixel_values = pixel_values.to("cuda")

        generation_config = {
            "max_new_tokens": 512,
            "do_sample": False,
        }

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
            )
        return response

    def _preprocess_internvl_image(self, image: Image.Image):
        """Preprocess image for InternVL."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        tensor = transform(image).unsqueeze(0)
        # Convert to bfloat16 to match model weights
        return tensor.to(torch.bfloat16)

    def _inference_paligemma(self, image: Image.Image, prompt: str) -> str:
        """Run PaliGemma inference."""
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        response = self.processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def _inference_phi3(self, image: Image.Image, prompt: str) -> str:
        """Run Phi-3 Vision inference."""
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{prompt}",
            }
        ]

        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        response = self.processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def get_metrics(self) -> Dict[str, float]:
        """Calculate benchmark metrics."""
        if not self.inference_times:
            return {
                "load_time_seconds": self.load_time,
                "inference_latency_ms": 0.0,
                "memory_mb": self.memory_usage,
                "tokens_per_second": 0.0,
            }

        avg_latency = sum(self.inference_times) / len(self.inference_times)

        # Rough tokens per second estimate (assuming ~50 tokens per response)
        tokens_per_second = 50 / (avg_latency / 1000) if avg_latency > 0 else 0

        return {
            "load_time_seconds": self.load_time,
            "inference_latency_ms": avg_latency,
            "memory_mb": self.memory_usage,
            "tokens_per_second": tokens_per_second,
        }


def find_test_images(image_dir: Path, num_samples: int) -> List[Path]:
    """
    Find test images from directory.

    Args:
        image_dir: Directory containing images
        num_samples: Number of samples to use

    Returns:
        List of image paths
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Supported image extensions
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

    # Collect all images
    images = []
    for ext in extensions:
        images.extend(image_dir.glob(ext))

    if not images:
        raise ValueError(f"No images found in {image_dir}")

    # Limit to num_samples
    images = sorted(images)[:num_samples]

    print(f"\nFound {len(images)} test images")
    return images


def run_benchmark(
    model_name: str,
    device: str,
    quantization: str,
    image_dir: Path,
    num_samples: int,
) -> Dict:
    """
    Run complete benchmark.

    Args:
        model_name: Model to benchmark
        device: Device (cpu/cuda)
        quantization: Quantization level (none/int8/int4)
        image_dir: Test images directory
        num_samples: Number of samples

    Returns:
        Benchmark results dictionary
    """
    print("=" * 70)
    print(f"Mobile VLM Benchmark: {model_name}")
    print("=" * 70)

    # Find test images
    images = find_test_images(image_dir, num_samples)

    # Initialize benchmark
    benchmark = ModelBenchmark(model_name, device, quantization)

    # Load model
    load_time = benchmark.load_model()

    # Warmup run
    print("\nRunning warmup...")
    warmup_response, warmup_time = benchmark.run_inference(images[0], warmup=True)
    print(f"  Warmup time: {warmup_time:.2f}ms")

    # Benchmark runs
    print(f"\nBenchmarking on {len(images)} images...")
    results = []

    for i, image_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] Processing {image_path.name}...", end=" ")

        response, latency = benchmark.run_inference(image_path)

        results.append({
            "image": str(image_path),
            "response": response,
            "latency_ms": latency,
        })

        print(f"{latency:.2f}ms")

    # Calculate metrics
    metrics = benchmark.get_metrics()

    print("\n" + "=" * 70)
    print("Benchmark Results:")
    print("-" * 70)
    print(f"Load time:          {metrics['load_time_seconds']:.2f}s")
    print(f"Avg latency:        {metrics['inference_latency_ms']:.2f}ms")
    print(f"Memory usage:       {metrics['memory_mb']:.2f} MB")
    print(f"Tokens/second:      {metrics['tokens_per_second']:.2f}")
    print("=" * 70)

    return {
        "model": model_name,
        "device": device,
        "quantization": quantization,
        "num_samples": len(images),
        "metrics": metrics,
        "per_sample_results": results,
    }


def log_to_mlflow(results: Dict, args: argparse.Namespace):
    """
    Log benchmark results to MLflow.

    Args:
        results: Benchmark results dictionary
        args: Command line arguments
    """
    mlflow.set_experiment("mobile_vlm_benchmark")

    run_name = f"{results['model']}_{results['device']}_{results['quantization']}"

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            "model_name": results["model"],
            "device": results["device"],
            "quantization": results["quantization"],
            "num_samples": results["num_samples"],
            "image_dir": str(args.image_dir),
        })

        # Log metrics
        metrics = results["metrics"]
        mlflow.log_metrics({
            "load_time_seconds": metrics["load_time_seconds"],
            "inference_latency_ms": metrics["inference_latency_ms"],
            "memory_mb": metrics["memory_mb"],
            "tokens_per_second": metrics["tokens_per_second"],
        })

        # Save detailed results as artifact
        results_file = Path("benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        mlflow.log_artifact(str(results_file))
        results_file.unlink()  # Clean up

        print(f"\nResults logged to MLflow experiment: mobile_vlm_benchmark")
        print(f"Run name: {run_name}")


def list_models():
    """Print all available models with their specifications."""
    print("\n" + "=" * 80)
    print("Available Edge VLM Models")
    print("=" * 80)

    # Group by size
    size_groups = {
        "Ultra-Lightweight (<2GB RAM)": [],
        "Lightweight (2-4GB RAM)": [],
        "Medium (4-8GB RAM)": [],
        "Large (8GB+ RAM)": [],
    }

    for name, config in MODEL_CONFIGS.items():
        ram = config.get("ram", "unknown")
        if "<2GB" in ram or "~2GB" in ram:
            size_groups["Ultra-Lightweight (<2GB RAM)"].append((name, config))
        elif "~3GB" in ram or "~4GB" in ram:
            size_groups["Lightweight (2-4GB RAM)"].append((name, config))
        elif "~5GB" in ram or "~6GB" in ram or "~8GB" in ram:
            size_groups["Medium (4-8GB RAM)"].append((name, config))
        else:
            size_groups["Large (8GB+ RAM)"].append((name, config))

    for group_name, models in size_groups.items():
        if not models:
            continue
        print(f"\n{group_name}")
        print("-" * 80)
        print(f"{'Model':<22} {'Params':<12} {'RAM':<10} {'Description'}")
        print("-" * 80)
        for name, config in models:
            params = config.get("params", "?")
            ram = config.get("ram", "?")
            desc = config.get("description", "")[:35]
            print(f"{name:<22} {params:<12} {ram:<10} {desc}")

    print("\n" + "=" * 80)
    print(f"Total: {len(MODEL_CONFIGS)} models available")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark edge VLMs for mobile deployment"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to benchmark",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "int8", "int4"],
        help="Quantization level",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of test samples",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing test images",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Benchmark ALL available models (requires --image-dir)",
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        list_models()
        return

    # Validate required args for benchmarking
    if not args.image_dir:
        parser.error("--image-dir is required for benchmarking")

    if not args.model and not args.all_models:
        parser.error("--model or --all-models is required")

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Determine models to benchmark
    if args.all_models:
        models_to_test = list(MODEL_CONFIGS.keys())
        print(f"\nBenchmarking ALL {len(models_to_test)} models...")
    else:
        models_to_test = [args.model]

    all_results = []
    failed_models = []

    for model_name in models_to_test:
        try:
            print(f"\n{'='*70}")
            print(f"Testing: {model_name}")
            print(f"{'='*70}")

            results = run_benchmark(
                model_name=model_name,
                device=args.device,
                quantization=args.quantization,
                image_dir=args.image_dir,
                num_samples=args.samples,
            )

            all_results.append(results)

            # Log to MLflow
            if not args.no_mlflow:
                log_to_mlflow(results, args)

            # Print sample responses (only for single model)
            if not args.all_models:
                print("\nSample Responses:")
                print("-" * 70)
                for i, result in enumerate(results["per_sample_results"][:3], 1):
                    print(f"\n[{i}] {Path(result['image']).name}")
                    print(f"Response: {result['response'][:200]}...")
                    print(f"Latency: {result['latency_ms']:.2f}ms")

            # Clear CUDA cache between models
            if args.device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nERROR: Benchmark failed for {model_name}: {e}")
            failed_models.append((model_name, str(e)))
            if not args.all_models:
                raise
            continue

    # Summary for --all-models
    if args.all_models and all_results:
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"\n{'Model':<25} {'Latency (ms)':<15} {'Memory (MB)':<15} {'Tokens/s':<12}")
        print("-" * 80)
        for r in sorted(all_results, key=lambda x: x["metrics"]["inference_latency_ms"]):
            m = r["metrics"]
            print(f"{r['model']:<25} {m['inference_latency_ms']:<15.2f} {m['memory_mb']:<15.2f} {m['tokens_per_second']:<12.2f}")

        if failed_models:
            print(f"\nFailed models ({len(failed_models)}):")
            for name, error in failed_models:
                print(f"  - {name}: {error[:50]}...")

        # Save comprehensive results
        summary_path = Path("benchmark_summary.json")
        summary_data = {
            "device": args.device,
            "quantization": args.quantization,
            "num_samples": args.samples,
            "results": [
                {
                    "model": r["model"],
                    "metrics": r["metrics"],
                }
                for r in all_results
            ],
            "failed": failed_models,
        }
        summary_path.write_text(json.dumps(summary_data, indent=2))
        print(f"\nFull results saved to: {summary_path}")

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
