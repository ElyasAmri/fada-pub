"""
Benchmark configuration for VLM fine-tuning experiments.

Each model will be trained on a separate vast.ai instance.

NOTE: The authoritative model list is docs/experiments/models-to-test.md (54 models).
This config contains the subset suitable for cloud fine-tuning benchmarks.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a single model benchmark."""
    name: str  # Short name for logging
    model_id: str  # HuggingFace model ID
    min_vram: int  # Minimum VRAM in GB
    batch_size: int = 1
    gradient_accumulation: int = 8
    use_4bit: bool = True
    epochs: int = 1  # For benchmarking, 1 epoch is enough
    max_train_samples: Optional[int] = None  # None = use all
    max_val_samples: Optional[int] = None


# Models to benchmark - ordered by size
# batch_size=1 to avoid collation issues with variable-sized VLM images
# All models use 4-bit quantization and gradient_accumulation=16 for VLM training
BENCHMARK_MODELS = [
    # SmolVLM2 series (smallest models)
    ModelConfig(
        name="smolvlm2-256m",
        model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        min_vram=2,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="smolvlm2-500m",
        model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        min_vram=3,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="smolvlm2-2.2b",
        model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        min_vram=6,
        batch_size=1,
        gradient_accumulation=16,
    ),

    # InternVL3 series (1B-78B)
    ModelConfig(
        name="internvl3-1b",
        model_id="OpenGVLab/InternVL3-1B",
        min_vram=4,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="internvl3-2b",
        model_id="OpenGVLab/InternVL3-2B",
        min_vram=6,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="internvl3-8b",
        model_id="OpenGVLab/InternVL3-8B",
        min_vram=18,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="internvl3-9b",
        model_id="OpenGVLab/InternVL3-9B",
        min_vram=20,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="internvl3-14b",
        model_id="OpenGVLab/InternVL3-14B",
        min_vram=32,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="internvl3-14b-awq",
        model_id="OpenGVLab/InternVL3-14B-AWQ",
        min_vram=12,  # AWQ reduces to ~10GB
        batch_size=1,
        gradient_accumulation=16,
    ),
    # InternVL3-38B (80GB) and InternVL3-78B (160GB) removed - unrealistic for budget

    # InternVL3.5 series
    ModelConfig(
        name="internvl3.5-1b",
        model_id="OpenGVLab/InternVL3_5-1B",
        min_vram=4,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="internvl3.5-2b",
        model_id="OpenGVLab/InternVL3_5-2B",
        min_vram=6,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="internvl3.5-8b",
        model_id="OpenGVLab/InternVL3_5-8B",
        min_vram=18,
        batch_size=1,
        gradient_accumulation=16,
    ),

    # InternVL2 series (reference)
    ModelConfig(
        name="internvl2-2b",
        model_id="OpenGVLab/InternVL2-2B",
        min_vram=6,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="internvl2-4b",
        model_id="OpenGVLab/InternVL2-4B",
        min_vram=10,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="internvl2-8b",
        model_id="OpenGVLab/InternVL2-8B",
        min_vram=18,
        batch_size=1,
        gradient_accumulation=16,
    ),

    # Qwen2.5-VL series (Jan 2025)
    ModelConfig(
        name="qwen2.5-vl-3b",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        min_vram=8,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="qwen2.5-vl-3b-gptq",
        model_id="hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4",
        min_vram=4,  # GPTQ reduces to ~3GB
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="qwen2.5-vl-7b",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        min_vram=16,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="qwen2.5-vl-7b-gptq",
        model_id="hfl/Qwen2.5-VL-7B-Instruct-GPTQ-Int4",
        min_vram=8,  # GPTQ reduces to ~6GB
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="qwen2.5-vl-32b",
        model_id="Qwen/Qwen2.5-VL-32B-Instruct",
        min_vram=70,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="qwen2.5-vl-32b-awq",
        model_id="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        min_vram=20,  # AWQ reduces to ~18GB
        batch_size=1,
        gradient_accumulation=16,
    ),
    # Qwen2.5-VL-72B (150GB) removed - unrealistic for budget

    # Qwen3-VL series (current gen, Nov 2025)
    ModelConfig(
        name="qwen3-vl-2b",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        min_vram=8,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="qwen3-vl-4b",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        min_vram=12,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="qwen3-vl-8b",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        min_vram=20,
        batch_size=1,
        gradient_accumulation=16,
    ),

    # Qwen2-VL series (previous gen, kept for comparison)
    ModelConfig(
        name="qwen2-vl-2b",
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        min_vram=6,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="qwen2-vl-7b",
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        min_vram=16,
        batch_size=1,
        gradient_accumulation=16,
    ),

    # MiniCPM-V series
    ModelConfig(
        name="minicpm-v-2.6",
        model_id="openbmb/MiniCPM-V-2_6",
        min_vram=10,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="minicpm-o-2.6",
        model_id="openbmb/MiniCPM-o-2_6",
        min_vram=10,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="minicpm-v-4",
        model_id="openbmb/MiniCPM-V-4",
        min_vram=12,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="minicpm-v-4.5",
        model_id="openbmb/MiniCPM-V-4.5",
        min_vram=12,
        batch_size=1,
        gradient_accumulation=16,
    ),

    # Kimi-VL series (MoE)
    ModelConfig(
        name="kimi-vl-a3b-instruct",
        model_id="moonshotai/Kimi-VL-A3B-Instruct",
        min_vram=10,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="kimi-vl-a3b-thinking",
        model_id="moonshotai/Kimi-VL-A3B-Thinking",
        min_vram=10,
        batch_size=1,
        gradient_accumulation=16,
    ),

    # Llama 3.2 Vision series
    ModelConfig(
        name="llama-3.2-11b-vision",
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        min_vram=26,
        batch_size=1,
        gradient_accumulation=16,
    ),
    # Llama-3.2-90B (180GB) removed - unrealistic for budget

    # Gemma 3 series
    ModelConfig(
        name="gemma-3-1b",
        model_id="google/gemma-3-1b-it",
        min_vram=4,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="gemma-3-4b",
        model_id="google/gemma-3-4b-it",
        min_vram=10,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="gemma-3-12b",
        model_id="google/gemma-3-12b-it",
        min_vram=28,
        batch_size=1,
        gradient_accumulation=16,
    ),
    ModelConfig(
        name="gemma-3-27b",
        model_id="google/gemma-3-27b-it",
        min_vram=60,
        batch_size=1,
        gradient_accumulation=16,
    ),

    # Phi-4 multimodal
    ModelConfig(
        name="phi-4-multimodal",
        model_id="microsoft/Phi-4-multimodal-instruct",
        min_vram=32,
        batch_size=1,
        gradient_accumulation=16,
    ),
]

# Quick benchmark subset (for testing the pipeline)
# Only the smallest models for rapid testing with limited samples
QUICK_BENCHMARK_MODELS = [
    ModelConfig(
        name="smolvlm2-256m",
        model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        min_vram=2,
        batch_size=1,
        gradient_accumulation=16,
        max_train_samples=100,
        max_val_samples=20,
    ),
    ModelConfig(
        name="internvl3-1b",
        model_id="OpenGVLab/InternVL3-1B",
        min_vram=4,
        batch_size=1,
        gradient_accumulation=16,
        max_train_samples=100,
        max_val_samples=20,
    ),
    ModelConfig(
        name="qwen2-vl-2b",
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        min_vram=6,
        batch_size=1,
        gradient_accumulation=16,
        max_train_samples=100,
        max_val_samples=20,
    ),
    ModelConfig(
        name="qwen2.5-vl-3b-gptq",
        model_id="hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4",
        min_vram=4,
        batch_size=1,
        gradient_accumulation=16,
        max_train_samples=100,
        max_val_samples=20,
    ),
]

# GPU presets for different model sizes
GPU_PRESETS = {
    "tiny": {  # <4GB models (quantized)
        "gpu_name": "RTX_3060",
        "min_vram": 8,
        "max_price": 0.20,
        "disk_gb": 30,  # Small model weights + dataset
    },
    "small": {  # 2B-3B models
        "gpu_name": "RTX_4090",
        "min_vram": 16,
        "max_price": 0.40,
        "disk_gb": 40,  # Model weights (~7GB) + dataset + training outputs
    },
    "medium": {  # 7B models
        "gpu_name": "RTX_4090",
        "min_vram": 24,
        "max_price": 0.50,
        "disk_gb": 60,  # Model weights (~14GB) + dataset + training outputs
    },
    "large": {  # 8B+ models
        "gpu_name": "A100_PCIE",
        "min_vram": 40,
        "max_price": 1.00,
        "disk_gb": 80,  # Large model weights + caches
    },
}


def get_gpu_preset(model_config: ModelConfig) -> dict:
    """Get appropriate GPU preset for a model."""
    if model_config.min_vram <= 8:
        return GPU_PRESETS["tiny"]
    elif model_config.min_vram <= 16:
        return GPU_PRESETS["small"]
    elif model_config.min_vram <= 24:
        return GPU_PRESETS["medium"]
    else:
        return GPU_PRESETS["large"]
