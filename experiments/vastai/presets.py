"""
GPU Presets and Model Mappings for Vast.ai.

Automatically selects appropriate GPU tier based on model size.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# Pinned Docker image for reproducibility
# Using official PyTorch image with CUDA 12.4
DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

# Fallback images by CUDA version (in case primary unavailable)
DOCKER_IMAGES_BY_CUDA = {
    "12.4": "vastai/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
    "12.1": "vastai/pytorch:2.4.0-cuda12.1-cudnn8-runtime",
    "11.8": "vastai/pytorch:2.3.0-cuda11.8-cudnn8-runtime",
}

# Minimum requirements
MIN_DRIVER_VERSION = 535
MIN_CUDA_VERSION = "12.1"


@dataclass
class GPUPreset:
    """GPU configuration preset."""
    name: str
    gpu_name: str  # Vast.ai GPU name (e.g., "RTX_4090")
    min_vram: int  # Minimum VRAM in GB
    max_price: float  # Maximum $/hr
    disk_gb: int  # Default disk size
    description: str
    cuda_version_min: str = "12.1"  # Minimum CUDA version
    driver_version_min: int = 535   # Minimum driver version


# GPU presets by tier
PRESETS: Dict[str, GPUPreset] = {
    "small": GPUPreset(
        name="small",
        gpu_name="RTX_4090",
        min_vram=20,
        max_price=0.40,
        disk_gb=10,  # Minimal: use network volume for data
        description="RTX 4090 (~24GB) - For 1B-4B models",
        cuda_version_min="12.1",
        driver_version_min=535,
    ),
    "medium": GPUPreset(
        name="medium",
        gpu_name="A100",
        min_vram=40,
        max_price=1.20,
        disk_gb=10,  # Minimal: use network volume for data
        description="A100 40GB - For 7B-14B models",
        cuda_version_min="12.1",
        driver_version_min=535,
    ),
    "large": GPUPreset(
        name="large",
        gpu_name="H100",
        min_vram=80,
        max_price=2.50,
        disk_gb=15,  # Slightly more for larger model caches
        description="H100 80GB - For 30B+ models",
        cuda_version_min="12.1",
        driver_version_min=535,
    ),
    "budget": GPUPreset(
        name="budget",
        gpu_name="RTX_3090",
        min_vram=20,
        max_price=0.25,
        disk_gb=10,  # Minimal: container has its own disk, data on HuggingFace
        description="RTX 3090 (~24GB) - Budget option for small models",
        cuda_version_min="11.8",
        driver_version_min=520,
    ),
}


# Model configurations with size and VRAM estimates
# Format: "model_id": (params_billions, vram_estimate_gb, recommended_preset)
MODEL_CONFIGS: Dict[str, Tuple[float, int, str]] = {
    # InternVL3 family
    "OpenGVLab/InternVL3-1B": (1.0, 4, "small"),
    "OpenGVLab/InternVL3-2B": (2.0, 6, "small"),
    "OpenGVLab/InternVL3-8B": (8.0, 18, "small"),
    "OpenGVLab/InternVL3-9B": (9.0, 20, "small"),
    "OpenGVLab/InternVL3-14B": (14.0, 32, "medium"),
    "OpenGVLab/InternVL3-38B": (38.0, 80, "large"),
    "OpenGVLab/InternVL3-78B": (78.0, 160, "large"),

    # InternVL3.5 family
    "OpenGVLab/InternVL3_5-1B": (1.0, 4, "small"),
    "OpenGVLab/InternVL3_5-2B": (2.0, 6, "small"),
    "OpenGVLab/InternVL3_5-8B": (8.0, 18, "small"),

    # Qwen2.5-VL family
    "Qwen/Qwen2.5-VL-3B-Instruct": (3.0, 8, "small"),
    "Qwen/Qwen2.5-VL-7B-Instruct": (7.0, 16, "small"),
    "Qwen/Qwen2.5-VL-32B-Instruct": (32.0, 70, "large"),
    "Qwen/Qwen2.5-VL-72B-Instruct": (72.0, 150, "large"),

    # Qwen2-VL family (already tested, for reference)
    "Qwen/Qwen2-VL-2B-Instruct": (2.0, 6, "small"),
    "Qwen/Qwen2-VL-7B-Instruct": (7.0, 16, "small"),

    # SmolVLM2 family
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct": (0.25, 2, "budget"),
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct": (0.5, 3, "budget"),
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct": (2.2, 6, "small"),

    # MiniCPM-V family
    "openbmb/MiniCPM-V-2_6": (8.0, 10, "small"),
    "openbmb/MiniCPM-o-2_6": (8.0, 10, "small"),
    "openbmb/MiniCPM-V-4": (8.0, 12, "small"),
    "openbmb/MiniCPM-V-4.5": (8.0, 12, "small"),

    # Kimi-VL (MoE)
    "moonshotai/Kimi-VL-A3B-Instruct": (2.8, 10, "small"),  # Active params
    "moonshotai/Kimi-VL-A3B-Thinking": (2.8, 10, "small"),

    # Llama 3.2 Vision
    "meta-llama/Llama-3.2-11B-Vision-Instruct": (11.0, 26, "medium"),
    "meta-llama/Llama-3.2-90B-Vision-Instruct": (90.0, 180, "large"),

    # Gemma 3
    "google/gemma-3-1b-it": (1.0, 4, "small"),
    "google/gemma-3-4b-it": (4.0, 10, "small"),
    "google/gemma-3-12b-it": (12.0, 28, "medium"),
    "google/gemma-3-27b-it": (27.0, 60, "large"),

    # Phi-4
    "microsoft/Phi-4-multimodal-instruct": (14.0, 32, "medium"),

    # InternVL2 family (already tested, for reference)
    "OpenGVLab/InternVL2-2B": (2.0, 6, "small"),
    "OpenGVLab/InternVL2-4B": (4.0, 10, "small"),
    "OpenGVLab/InternVL2-8B": (8.0, 18, "small"),
}

# Model name aliases (short names -> full HuggingFace IDs)
MODEL_ALIASES: Dict[str, str] = {
    # InternVL3
    "internvl3-1b": "OpenGVLab/InternVL3-1B",
    "internvl3-2b": "OpenGVLab/InternVL3-2B",
    "internvl3-8b": "OpenGVLab/InternVL3-8B",
    "internvl3-9b": "OpenGVLab/InternVL3-9B",
    "internvl3-14b": "OpenGVLab/InternVL3-14B",

    # InternVL3.5
    "internvl3.5-1b": "OpenGVLab/InternVL3_5-1B",
    "internvl3.5-2b": "OpenGVLab/InternVL3_5-2B",
    "internvl3.5-8b": "OpenGVLab/InternVL3_5-8B",

    # Qwen2.5-VL
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",

    # SmolVLM2
    "smolvlm2-256m": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    "smolvlm2-500m": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "smolvlm2-2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",

    # MiniCPM-V
    "minicpm-v-2.6": "openbmb/MiniCPM-V-2_6",
    "minicpm-v-4": "openbmb/MiniCPM-V-4",
    "minicpm-v-4.5": "openbmb/MiniCPM-V-4.5",

    # Kimi-VL
    "kimi-vl": "moonshotai/Kimi-VL-A3B-Instruct",
    "kimi-vl-thinking": "moonshotai/Kimi-VL-A3B-Thinking",

    # Llama
    "llama-3.2-vision-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",

    # Gemma
    "gemma-3-4b": "google/gemma-3-4b-it",
    "gemma-3-12b": "google/gemma-3-12b-it",

    # Phi
    "phi-4-multimodal": "microsoft/Phi-4-multimodal-instruct",
}


def resolve_model_id(model_input: str) -> str:
    """
    Resolve a model name/alias to full HuggingFace ID.

    Args:
        model_input: Model name, alias, or full HF ID

    Returns:
        Full HuggingFace model ID
    """
    # Check if it's already a full ID
    if "/" in model_input:
        return model_input

    # Check aliases (case-insensitive)
    lower_input = model_input.lower()
    if lower_input in MODEL_ALIASES:
        return MODEL_ALIASES[lower_input]

    # Try to find partial match
    for alias, full_id in MODEL_ALIASES.items():
        if lower_input in alias or alias in lower_input:
            return full_id

    # Return as-is (might be a new model)
    return model_input


def get_model_config(model_id: str) -> Tuple[float, int, str]:
    """
    Get model configuration (params, VRAM, preset).

    Args:
        model_id: HuggingFace model ID

    Returns:
        Tuple of (params_billions, vram_gb, preset_name)
    """
    resolved = resolve_model_id(model_id)

    if resolved in MODEL_CONFIGS:
        return MODEL_CONFIGS[resolved]

    # Default: assume medium-sized model
    print(f"Warning: Unknown model {resolved}, using default config")
    return (7.0, 16, "small")


def get_preset_for_model(model_id: str) -> GPUPreset:
    """
    Get the recommended GPU preset for a model.

    Args:
        model_id: Model name, alias, or HuggingFace ID

    Returns:
        GPUPreset for running this model
    """
    _, _, preset_name = get_model_config(model_id)
    return PRESETS[preset_name]


def estimate_vram(model_id: str, use_4bit: bool = True) -> int:
    """
    Estimate VRAM needed for a model.

    Args:
        model_id: Model identifier
        use_4bit: Whether 4-bit quantization will be used

    Returns:
        Estimated VRAM in GB
    """
    _, base_vram, _ = get_model_config(model_id)

    if use_4bit:
        # 4-bit uses roughly 40% of FP16 VRAM
        return int(base_vram * 0.4) + 2  # +2GB for overhead

    return base_vram


def list_models_by_preset(preset_name: str) -> list:
    """List all models that use a given preset."""
    models = []
    for model_id, (params, vram, preset) in MODEL_CONFIGS.items():
        if preset == preset_name:
            models.append((model_id, params, vram))
    return sorted(models, key=lambda x: x[1])  # Sort by params


def print_presets():
    """Print all available presets."""
    print("\nAvailable GPU Presets:")
    print("=" * 70)
    for name, preset in PRESETS.items():
        print(f"\n  {name.upper()}: {preset.description}")
        print(f"    GPU: {preset.gpu_name}, Min VRAM: {preset.min_vram}GB, Max Price: ${preset.max_price}/hr")

        models = list_models_by_preset(name)
        if models:
            print(f"    Models: {', '.join(m[0].split('/')[-1] for m in models[:5])}")
            if len(models) > 5:
                print(f"            ...and {len(models) - 5} more")


def get_docker_image(cuda_version: str = None) -> str:
    """
    Get the appropriate Docker image for the CUDA version.

    Args:
        cuda_version: Target CUDA version (e.g., "12.4"). If None, returns default.

    Returns:
        Docker image string
    """
    if cuda_version is None:
        return DOCKER_IMAGE

    # Find best matching image
    for version, image in sorted(DOCKER_IMAGES_BY_CUDA.items(), reverse=True):
        if cuda_version >= version:
            return image

    # Fallback to default
    return DOCKER_IMAGE


def print_model_list():
    """Print all supported models with their configurations."""
    print("\nSupported Models:")
    print("=" * 90)
    print(f"{'Model':<45} {'Params':>8} {'VRAM':>8} {'Preset':>10}")
    print("-" * 90)

    for model_id in sorted(MODEL_CONFIGS.keys()):
        params, vram, preset = MODEL_CONFIGS[model_id]
        short_name = model_id.split("/")[-1]
        print(f"{short_name:<45} {params:>7.1f}B {vram:>7}GB {preset:>10}")
