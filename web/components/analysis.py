"""
Analysis Functions for FADA Web App
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
import time
import logging

logger = logging.getLogger(__name__)

# Import constants
try:
    from src.config.constants import CLASSES, ORGAN_INFO, get_display_name
    CONSTANTS_AVAILABLE = True
except ImportError:
    CONSTANTS_AVAILABLE = False
    logger.warning("Constants import failed, using local definitions")
    CLASSES = [
        "Abdomen", "Aorta", "Cervical", "Cervix", "Femur",
        "Non_standard_NT", "Public_Symphysis_fetal_head",
        "Standard_NT", "Thorax", "Trans-cerebellum",
        "Trans-thalamic", "Trans-ventricular"
    ]
    ORGAN_INFO = {
        "Abdomen": "The abdominal view shows stomach, liver, and cord insertion.",
        "Femur": "The femur view is used to measure femur length (FL).",
        "Thorax": "The thoracic view shows lung fields and diaphragm.",
    }
    def get_display_name(name):
        return name


def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return transform(image).unsqueeze(0)


def analyze_ultrasound(image, model=None):
    """Analyze ultrasound image and return results"""
    if model is None:
        st.error("Classification model not loaded. Check console for errors.")
        return {
            "organ": "Error",
            "confidence": 0.0,
            "quality": "Unknown",
            "orientation": "Unknown",
            "processing_time": 0.0
        }

    with torch.no_grad():
        start_time = time.time()
        image_tensor = preprocess_image(image)
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)
        organ_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities.max().item()

        organs = CLASSES

        if organ_idx >= len(organs):
            organ_idx = len(organs) - 1

        detected_organ = organs[organ_idx]

        if CONSTANTS_AVAILABLE:
            display_name = get_display_name(detected_organ)
        else:
            display_name = detected_organ

        # Confidence-based quality assessment
        if confidence > 0.85:
            quality = "High confidence"
        elif confidence > 0.7:
            quality = "Good confidence"
        elif confidence > 0.5:
            quality = "Moderate confidence"
        else:
            quality = "Low confidence"

        return {
            "organ": display_name,
            "raw_organ": detected_organ,
            "confidence": confidence,
            "quality": quality,
            "orientation": "Standard View",
            "processing_time": time.time() - start_time
        }


def generate_response(analysis_results):
    """Generate chatbot response based on analysis results"""
    organ = analysis_results['organ']
    confidence = analysis_results['confidence']
    quality = analysis_results['quality']
    orientation = analysis_results['orientation']

    response = f"## Analysis Complete\n\n"
    response += f"**Detected View:** {organ}\n"
    response += f"**Classification Confidence:** {confidence:.1%}\n"
    response += f"**Classification Quality:** {quality}\n"
    response += f"**Orientation:** {orientation}\n\n"

    response += "### Note\n"
    response += "This is a research prototype for organ classification only. "
    response += "It identifies the type of ultrasound view but does NOT assess anatomical normality or detect abnormalities. "
    response += "Clinical interpretation requires a qualified healthcare professional.\n\n"

    if organ in ORGAN_INFO:
        response += f"### View Details\n{ORGAN_INFO[organ]}\n"

    return response
