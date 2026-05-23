"""
VLM Factory - Factory functions for creating VLM instances
"""

from src.inference.vlm_interface import VLMManager
from src.inference.local import MiniCPMVLM, Qwen2VLVLM, InternVLVLM, MoondreamVLM


def create_top_vlms(use_api: bool = False, api_endpoint: str = None, api_key: str = None, gpu_8gb: bool = True) -> VLMManager:
    """
    Create VLM manager with top-3 models

    Args:
        use_api: Whether to use API endpoints instead of local inference
        api_endpoint: API endpoint URL (if use_api=True)
        api_key: API key (if use_api=True)
        gpu_8gb: Whether using 8GB GPU (use smaller models if True)

    Returns:
        VLMManager with registered models
    """
    manager = VLMManager()

    if use_api:
        raise NotImplementedError(
            "API-based VLM inference is no longer supported via vlm_factory. "
            "Use experiments/api_models/test_api_vlm.py for API model evaluation."
        )
    else:
        # Local GPU models
        if gpu_8gb:
            # Use models that fit comfortably in 8GB
            manager.register_model("minicpm", MiniCPMVLM(use_4bit=True))
            manager.register_model("internvl2_2b", InternVLVLM(use_4bit=True))
            manager.register_model("moondream", MoondreamVLM(use_4bit=False))
        else:
            # Use larger models if GPU has more memory
            manager.register_model("minicpm", MiniCPMVLM(use_4bit=True))
            manager.register_model("qwen2vl", Qwen2VLVLM(use_4bit=True))
            manager.register_model("internvl2", InternVLVLM(
                model_id="OpenGVLab/InternVL2-4B",
                display_name="InternVL2-4B",
                use_4bit=True
            ))

    return manager
