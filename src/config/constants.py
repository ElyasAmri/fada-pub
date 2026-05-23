"""
FADA Constants - Centralized configuration for fetal ultrasound classification

This module provides a single source of truth for class labels, display names,
and mappings used throughout the FADA system.
"""

from typing import Dict, List

# 14-class labels for fetal ultrasound classification
CLASSES: List[str] = [
    'Abdomen',
    'Aorta',
    'CRL-View',
    'Cervical',
    'Cervix',
    'Femur',
    'NT-View',
    'Non_standard_NT',
    'Public_Symphysis_fetal_head',
    'Standard_NT',
    'Thorax',
    'Trans-cerebellum',
    'Trans-thalamic',
    'Trans-ventricular'
]

# Display names for UI (corrects typos for user-facing text)
DISPLAY_NAMES: Dict[str, str] = {
    'Abdomen': 'Abdomen',
    'Aorta': 'Aortic Arch',
    'Cervical': 'Cervical View',
    'Cervix': 'Cervix',
    'CRL-View': 'Crown-Rump Length View',
    'Femur': 'Femur',
    'Non_standard_NT': 'Non-standard NT',
    'NT-View': 'Nuchal Translucency View',
    'Public_Symphysis_fetal_head': 'Fetal Head Position',
    'Standard_NT': 'Standard NT',
    'Thorax': 'Thorax',
    'Trans-cerebellum': 'Transcerebellar Plane',
    'Trans-thalamic': 'Transthalamic Plane',
    'Trans-ventricular': 'Transventricular Plane'
}

# Clinical descriptions for each class
CLASS_DESCRIPTIONS: Dict[str, str] = {
    'Abdomen': 'Abdominal cross-section for organ assessment',
    'Aorta': 'Aortic arch view for cardiac output assessment',
    'Cervical': 'Cervical view for cervix evaluation',
    'Cervix': 'Direct cervix view for length measurement',
    'CRL-View': 'Crown-rump length measurement view for first trimester dating',
    'Femur': 'Femur length measurement for growth assessment',
    'Non_standard_NT': 'Non-standard nuchal translucency view',
    'NT-View': 'Nuchal translucency screening view',
    'Public_Symphysis_fetal_head': 'Fetal head position relative to pubic symphysis',
    'Standard_NT': 'Standard nuchal translucency measurement',
    'Thorax': 'Thoracic cross-section for lung and heart assessment',
    'Trans-cerebellum': 'Transcerebellar plane for posterior fossa evaluation',
    'Trans-thalamic': 'Transthalamic plane for midline structures',
    'Trans-ventricular': 'Transventricular plane for ventricle measurement'
}

# Category to VQA model directory mapping
VQA_MODEL_MAPPING: Dict[str, str] = {
    'Abdomen': 'abdomen',
    'Femur': 'femur',
    'Thorax': 'thorax',
    'Standard_NT': 'standard_nt',
    'Non_standard_NT': '1epoch',  # Original/fallback model
    # Brain categories - fallback to original model
    'Trans-cerebellum': '1epoch',
    'Trans-thalamic': '1epoch',
    'Trans-ventricular': '1epoch',
    # Other categories - fallback
    'Aorta': '1epoch',
    'Cervical': '1epoch',
    'Cervix': '1epoch',
    'CRL-View': '1epoch',
    'NT-View': '1epoch',
    'Public_Symphysis_fetal_head': '1epoch',
}

# Organ information for response generation (keyed by display name)
ORGAN_INFO: Dict[str, str] = {
    'Abdomen': 'The abdominal view shows stomach, liver, and cord insertion.',
    'Aortic Arch': 'The aortic arch view assesses cardiac output and vessel structure.',
    'Crown-Rump Length View': 'The CRL view is used for crown-rump length measurement in first trimester dating.',
    'Cervical View': 'The cervical view evaluates the cervical region.',
    'Cervix': 'The cervical view measures cervical length.',
    'Femur': 'The femur view is used to measure femur length (FL).',
    'Nuchal Translucency View': 'The NT view is used for nuchal translucency screening.',
    'Non-standard NT': 'Non-standard nuchal translucency measurement view.',
    'Fetal Head Position': 'Fetal head position relative to pubic symphysis.',
    'Standard NT': 'Standard nuchal translucency measurement for screening.',
    'Thorax': 'The thoracic view shows lung fields and diaphragm.',
    'Transcerebellar Plane': 'Transcerebellar plane for posterior fossa evaluation.',
    'Transthalamic Plane': 'Transthalamic plane for midline brain structures.',
    'Transventricular Plane': 'Transventricular plane for ventricle measurement.',
}

# Confidence thresholds for quality assessment
# See docs/experiments/models.yaml for detailed documentation
CONFIDENCE_THRESHOLDS = {
    'high': 0.85,
    'good': 0.70,
    'moderate': 0.50,
}

# Image validation constants
MIN_IMAGE_SIZE = 10  # Minimum width/height in pixels for valid images


def get_display_name(class_name: str) -> str:
    """Get user-friendly display name for a class."""
    return DISPLAY_NAMES.get(class_name, class_name)


def get_vqa_model_key(category: str) -> str:
    """Get VQA model directory key for a category."""
    return VQA_MODEL_MAPPING.get(category, '1epoch')
