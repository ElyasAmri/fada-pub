"""
Model Loading Functions for FADA Web App
"""

import streamlit as st
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Check imports
try:
    from src.models.classifier import FetalUltrasoundClassifier12
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    logger.error(f"Classifier import failed: {e}")

try:
    from src.models.vqa_model import UltrasoundVQA
    VQA_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    VQA_AVAILABLE = False
    logger.warning(f"VQA import failed: {e}")

try:
    from src.config.constants import get_vqa_model_key, VQA_MODEL_MAPPING
    CONSTANTS_AVAILABLE = True
except ImportError:
    CONSTANTS_AVAILABLE = False
    VQA_MODEL_MAPPING = {}


@st.cache_resource
def load_model():
    """Load the trained classification model"""
    if not MODEL_AVAILABLE:
        return None

    try:
        model_paths = [
            Path("models/best_model_efficientnet_b0_12class.pth"),
            Path("models/best_model_efficientnet_b0.pth"),
            Path("models/best_model.pth")
        ]

        for model_path in model_paths:
            if model_path.exists():
                if "12class" in str(model_path):
                    num_classes = 12
                else:
                    num_classes = 5

                model = FetalUltrasoundClassifier12(
                    num_classes=num_classes,
                    backbone='efficientnet_b0',
                    pretrained=False
                )

                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)

                model.eval()
                return model

        return None
    except (RuntimeError, KeyError, ValueError, OSError) as e:
        st.error(f"Failed to load classification model: {str(e)}")
        return None


@st.cache_resource
def load_vqa_model():
    """Load the VQA model (lazy loading)"""
    if not VQA_AVAILABLE:
        return None

    try:
        vqa = UltrasoundVQA(
            model_path="outputs/blip2_1epoch/final_model",
            device="auto"
        )
        return vqa
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"VQA initialization failed: {e}")
        return None


def get_vqa_model_for_category(category):
    """Get VQA model path for a specific organ category"""
    if CONSTANTS_AVAILABLE:
        model_key = get_vqa_model_key(category)
    else:
        # Fallback if constants not available
        model_key = VQA_MODEL_MAPPING.get(category, "1epoch")

    model_path = f"outputs/blip2_{model_key}/final_model"

    if Path(model_path).exists():
        return model_path
    else:
        return "outputs/blip2_1epoch/final_model"


def load_category_vqa(category):
    """Load VQA model for specific category"""
    if not VQA_AVAILABLE:
        return None

    try:
        model_path = get_vqa_model_for_category(category)
        vqa = UltrasoundVQA(model_path=model_path, device="auto")
        logger.info(f"Loading VQA for {category}: {model_path}")
        return vqa
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Failed to load VQA for {category}: {e}")
        return None
