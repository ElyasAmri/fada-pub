"""
Response Generation System for Fetal Ultrasound Chatbot
Generates natural language responses based on classification results
"""

import random
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """Container for model predictions"""
    predicted_class: str
    confidence: float
    top_3_predictions: List[Tuple[str, float]]


class ResponseGenerator:
    """Generate natural language responses for ultrasound classifications"""

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        """
        Initialize response generator

        Args:
            confidence_threshold: Minimum confidence for definitive statements
        """
        self.confidence_threshold = confidence_threshold

        # Clinical descriptions for each anatomical view
        self.class_descriptions = {
            'Abdomen': {
                'name': 'Fetal Abdomen',
                'clinical_purpose': 'abdominal circumference measurement and organ assessment',
                'structures': ['stomach bubble', 'liver', 'umbilical vein', 'kidneys'],
                'measurements': ['Abdominal Circumference (AC)'],
                'trimester': 'typically second/third'
            },
            'Aorta': {
                'name': 'Aortic Arch',
                'clinical_purpose': 'cardiac outflow tract assessment',
                'structures': ['aortic arch', 'ascending aorta', 'descending aorta'],
                'measurements': ['aortic diameter'],
                'trimester': 'second/third'
            },
            'Cervical': {
                'name': 'Cervical Region',
                'clinical_purpose': 'cervical spine and neck assessment',
                'structures': ['cervical spine', 'nuchal region', 'soft tissues'],
                'measurements': ['nuchal fold thickness'],
                'trimester': 'first/second'
            },
            'Cervix': {
                'name': 'Maternal Cervix',
                'clinical_purpose': 'cervical length assessment for preterm birth risk',
                'structures': ['internal os', 'external os', 'cervical canal', 'cervical length'],
                'measurements': ['Cervical Length (CL)'],
                'trimester': 'second/third'
            },
            'Femur': {
                'name': 'Fetal Femur',
                'clinical_purpose': 'gestational age estimation and growth assessment',
                'structures': ['femur diaphysis', 'femoral head', 'distal femur'],
                'measurements': ['Femur Length (FL)'],
                'trimester': 'second/third'
            },
            'Non_standard_NT': {
                'name': 'Non-standard Nuchal Translucency View',
                'clinical_purpose': 'alternative nuchal translucency assessment',
                'structures': ['nuchal translucency', 'fetal profile'],
                'measurements': ['NT thickness'],
                'trimester': 'first (11-14 weeks)'
            },
            'Public_Symphysis_fetal_head': {
                'name': 'Fetal Head at Pubic Symphysis',
                'clinical_purpose': 'fetal station and engagement assessment',
                'structures': ['fetal skull', 'maternal pubic symphysis', 'bladder'],
                'measurements': ['station', 'angle of progression'],
                'trimester': 'third (labor assessment)'
            },
            'Standard_NT': {
                'name': 'Standard Nuchal Translucency',
                'clinical_purpose': 'first trimester screening for chromosomal abnormalities',
                'structures': ['nuchal translucency', 'nasal bone', 'fetal profile'],
                'measurements': ['NT measurement (mm)'],
                'trimester': 'first (11-14 weeks)'
            },
            'Thorax': {
                'name': 'Fetal Thorax',
                'clinical_purpose': 'cardiac and pulmonary assessment',
                'structures': ['four-chamber heart', 'lungs', 'diaphragm', 'ribs'],
                'measurements': ['cardiothoracic ratio', 'chest circumference'],
                'trimester': 'second/third'
            },
            'Trans-cerebellum': {
                'name': 'Transcerebellar View',
                'clinical_purpose': 'posterior fossa evaluation and cerebellar measurement',
                'structures': ['cerebellum', 'cisterna magna', 'nuchal fold'],
                'measurements': ['Transcerebellar Diameter (TCD)', 'cisterna magna depth'],
                'trimester': 'second'
            },
            'Trans-thalamic': {
                'name': 'Transthalamic View',
                'clinical_purpose': 'midline brain structures and BPD measurement',
                'structures': ['thalami', 'third ventricle', 'cavum septum pellucidum'],
                'measurements': ['Biparietal Diameter (BPD)', 'Head Circumference (HC)'],
                'trimester': 'second/third'
            },
            'Trans-ventricular': {
                'name': 'Transventricular View',
                'clinical_purpose': 'lateral ventricle assessment',
                'structures': ['lateral ventricles', 'choroid plexus', 'atrium'],
                'measurements': ['ventricular atrium width (<10mm normal)'],
                'trimester': 'second/third'
            }
        }

        # Response templates based on confidence levels
        self.high_confidence_templates = [
            "This appears to be a **{name}** view with {confidence:.1f}% confidence. This view is used for {purpose}.",
            "I can identify this as a **{name}** scan ({confidence:.1f}% confidence). The primary purpose of this view is {purpose}.",
            "Based on the image characteristics, this is a **{name}** view ({confidence:.1f}% confident). This scan is typically performed for {purpose}."
        ]

        self.medium_confidence_templates = [
            "This appears to be a **{name}** view, though I'm moderately confident ({confidence:.1f}%). This type of scan is generally used for {purpose}.",
            "I believe this is a **{name}** view ({confidence:.1f}% confidence). If correct, this view would be used for {purpose}.",
            "This seems to be a **{name}** scan based on the visible features ({confidence:.1f}% confidence). Such views are typically for {purpose}."
        ]

        self.low_confidence_templates = [
            "I'm not entirely certain, but this might be a **{name}** view ({confidence:.1f}% confidence). You may want to verify with your healthcare provider.",
            "Based on limited confidence ({confidence:.1f}%), this could be a **{name}** view. Please confirm with a medical professional.",
            "My analysis suggests this might be a **{name}** scan, but confidence is low ({confidence:.1f}%). Professional verification is recommended."
        ]

    def generate_response(
        self,
        predicted_class: str,
        confidence: float,
        top_3_predictions: Optional[List[Tuple[str, float]]] = None,
        include_details: bool = True
    ) -> str:
        """
        Generate natural language response for classification result

        Args:
            predicted_class: Predicted anatomical view class
            confidence: Prediction confidence (0-100)
            top_3_predictions: Top 3 predictions with confidences
            include_details: Whether to include anatomical details

        Returns:
            Natural language response string
        """

        # Get class information
        if predicted_class not in self.class_descriptions:
            return self._generate_unknown_class_response(predicted_class, confidence)

        class_info = self.class_descriptions[predicted_class]

        # Select template based on confidence
        if confidence >= self.confidence_threshold * 100:
            template = random.choice(self.high_confidence_templates)
        elif confidence >= 50:
            template = random.choice(self.medium_confidence_templates)
        else:
            template = random.choice(self.low_confidence_templates)

        # Generate base response
        response = template.format(
            name=class_info['name'],
            confidence=confidence,
            purpose=class_info['clinical_purpose']
        )

        # Add details if requested and confidence is sufficient
        if include_details and confidence >= 50:
            response += self._generate_details(class_info, confidence)

        # Add alternative possibilities if confidence is low
        if confidence < self.confidence_threshold * 100 and top_3_predictions:
            response += self._generate_alternatives(top_3_predictions, predicted_class)

        # Add disclaimer
        response += "\n\n*Note: This is an AI-based analysis for educational purposes only. Always consult with qualified healthcare professionals for medical interpretation.*"

        return response

    def _generate_details(self, class_info: Dict, confidence: float) -> str:
        """Generate detailed information about the view"""
        details = []

        # Add visible structures
        if class_info['structures'] and confidence >= 60:
            structures = ', '.join(class_info['structures'][:-1])
            if len(class_info['structures']) > 1:
                structures += f" and {class_info['structures'][-1]}"
            else:
                structures = class_info['structures'][0]
            details.append(f"\n\n**Anatomical structures typically visible:** {structures}")

        # Add measurements
        if class_info['measurements']:
            measurements = ', '.join(class_info['measurements'])
            details.append(f"\n**Common measurements taken:** {measurements}")

        # Add trimester information
        if class_info['trimester']:
            details.append(f"\n**Typically performed:** {class_info['trimester']} trimester")

        return ''.join(details) if details else ""

    def _generate_alternatives(
        self,
        top_3_predictions: List[Tuple[str, float]],
        predicted_class: str
    ) -> str:
        """Generate alternative possibilities when confidence is low"""

        alternatives = []
        for class_name, conf in top_3_predictions:
            if class_name != predicted_class and conf >= 20:  # Only show alternatives with >20% confidence
                if class_name in self.class_descriptions:
                    alt_name = self.class_descriptions[class_name]['name']
                    alternatives.append(f"{alt_name} ({conf:.1f}%)")

        if alternatives:
            return f"\n\n**Alternative possibilities:** {', '.join(alternatives[:2])}"
        return ""

    def _generate_unknown_class_response(self, predicted_class: str, confidence: float) -> str:
        """Handle unknown class predictions"""
        return (
            f"I detected a classification as '{predicted_class}' with {confidence:.1f}% confidence, "
            "but I don't have detailed information about this view type. "
            "Please consult with a healthcare professional for proper interpretation.\n\n"
            "*Note: This is an AI-based analysis for educational purposes only.*"
        )

    def generate_error_response(self, error_type: str = "general") -> str:
        """Generate response for various error conditions"""

        error_responses = {
            "general": "I encountered an error while analyzing the ultrasound image. Please try again or consult with a healthcare professional.",
            "image_quality": "The image quality appears to be insufficient for accurate analysis. Please ensure the ultrasound image is clear and properly formatted.",
            "model_error": "I'm experiencing technical difficulties with the analysis. Please try again later or contact support.",
            "invalid_image": "The uploaded file doesn't appear to be a valid ultrasound image. Please upload a proper ultrasound scan image."
        }

        return error_responses.get(error_type, error_responses["general"])

    def generate_batch_summary(self, results: List[ClassificationResult]) -> str:
        """Generate summary for multiple images"""

        if not results:
            return "No images were analyzed."

        summary = f"**Analysis Summary for {len(results)} images:**\n\n"

        # Count by class
        class_counts = {}
        avg_confidence = 0

        for result in results:
            class_counts[result.predicted_class] = class_counts.get(result.predicted_class, 0) + 1
            avg_confidence += result.confidence

        avg_confidence /= len(results)

        # Generate summary text
        summary += f"**Average confidence:** {avg_confidence:.1f}%\n\n"
        summary += "**Detected views:**\n"

        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            if class_name in self.class_descriptions:
                display_name = self.class_descriptions[class_name]['name']
                summary += f"- {display_name}: {count} image{'s' if count > 1 else ''}\n"
            else:
                summary += f"- {class_name}: {count} image{'s' if count > 1 else ''}\n"

        summary += "\n*Note: Individual image analyses are provided above.*"

        return summary


# Example usage and testing
if __name__ == "__main__":
    # Initialize generator
    generator = ResponseGenerator(confidence_threshold=0.7)

    # Test high confidence response
    print("HIGH CONFIDENCE EXAMPLE:")
    print("-" * 50)
    response = generator.generate_response(
        predicted_class="Trans-thalamic",
        confidence=92.5,
        top_3_predictions=[
            ("Trans-thalamic", 92.5),
            ("Trans-ventricular", 5.2),
            ("Trans-cerebellum", 2.3)
        ]
    )
    print(response)

    print("\n\nLOW CONFIDENCE EXAMPLE:")
    print("-" * 50)
    response = generator.generate_response(
        predicted_class="Femur",
        confidence=45.3,
        top_3_predictions=[
            ("Femur", 45.3),
            ("Abdomen", 32.1),
            ("Aorta", 22.6)
        ]
    )
    print(response)

    print("\n\nERROR EXAMPLE:")
    print("-" * 50)
    print(generator.generate_error_response("image_quality"))