"""
Core Chatbot Module for Fetal Ultrasound Analysis
Integrates classification model with response generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import json
import time

from src.models.classifier import create_model
from src.data.augmentation import get_validation_augmentation
from src.chatbot.response_generator import ResponseGenerator, ClassificationResult
from src.chatbot.openai_integration import HybridResponseGenerator
from src.config.constants import CLASSES

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result for an ultrasound image"""
    image_path: str
    predicted_class: str
    confidence: float
    top_3_predictions: List[Tuple[str, float]]
    response_text: str
    processing_time: float
    metadata: Optional[Dict] = None


class UltrasoundChatbot:
    """Main chatbot class for ultrasound image analysis and response generation"""

    # Import from centralized constants
    CLASSES = CLASSES

    # Valid backbone architectures
    VALID_BACKBONES = ['efficientnet_b0', 'efficientnet_b1', 'resnet50', 'resnet18']

    def __init__(
        self,
        model_path: Optional[str] = None,
        backbone: str = 'efficientnet_b0',
        use_gpu: bool = True,
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the chatbot

        Args:
            model_path: Path to trained model checkpoint
            backbone: Model architecture
            use_gpu: Whether to use GPU if available
            use_openai: Whether to use OpenAI for response generation
            openai_api_key: OpenAI API key
            confidence_threshold: Minimum confidence for definitive statements

        Raises:
            ValueError: If confidence_threshold is not between 0 and 1
            ValueError: If backbone is not a supported architecture
        """
        # Validate parameters
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be between 0 and 1, got {confidence_threshold}")

        if backbone not in self.VALID_BACKBONES:
            raise ValueError(f"Invalid backbone: {backbone}. Must be one of {self.VALID_BACKBONES}")

        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load the model
        self.model = self._load_model(model_path, backbone)
        self.model.eval()

        # Initialize image preprocessing
        self.transform = get_validation_augmentation(224)

        # Initialize response generator
        self.response_generator = HybridResponseGenerator(
            use_openai=use_openai,
            openai_api_key=openai_api_key,
            fallback_to_templates=True
        )

        self.confidence_threshold = confidence_threshold

        # Conversation history for context
        self.conversation_history = []

        logger.info("Chatbot initialized successfully")

    def _load_model(self, model_path: Optional[str], backbone: str) -> nn.Module:
        """Load the trained classification model"""

        # Create model architecture
        model, _ = create_model(
            num_classes=12,
            backbone=backbone,
            pretrained=False,  # We'll load our trained weights
            dropout_rate=0.2
        )

        # Load weights if path provided
        if model_path and Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded model from checkpoint: {model_path}")
                    if 'epoch' in checkpoint:
                        logger.info(f"Model trained for {checkpoint['epoch']} epochs")
                else:
                    model.load_state_dict(checkpoint)
                    logger.info(f"Loaded model weights from: {model_path}")

            except (RuntimeError, KeyError, ValueError) as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                logger.warning("Using untrained model")
        else:
            logger.warning(f"No model found at {model_path}, using untrained model")

        return model.to(self.device)

    def analyze_image(
        self,
        image_path: Union[str, Path, Image.Image, np.ndarray],
        user_question: Optional[str] = None,
        include_details: bool = True
    ) -> AnalysisResult:
        """
        Analyze a single ultrasound image and generate response

        Args:
            image_path: Path to image or PIL Image or numpy array
            user_question: Optional specific question from user
            include_details: Whether to include anatomical details

        Returns:
            AnalysisResult with classification and response
        """
        start_time = time.time()

        try:
            # Load and preprocess image
            image_tensor = self._preprocess_image(image_path)

            # Get model predictions
            predicted_class, confidence, top_3 = self._classify_image(image_tensor)

            # Generate response
            response = self.response_generator.generate_response(
                predicted_class=predicted_class,
                confidence=confidence,
                top_3_predictions=top_3,
                user_question=user_question,
                use_ai=True
            )

            # Create result
            result = AnalysisResult(
                image_path=str(image_path) if isinstance(image_path, (str, Path)) else "uploaded_image",
                predicted_class=predicted_class,
                confidence=confidence,
                top_3_predictions=top_3,
                response_text=response,
                processing_time=time.time() - start_time,
                metadata={
                    'model_device': str(self.device),
                    'include_details': include_details
                }
            )

            # Add to conversation history
            self._update_conversation_history(result, user_question)

            return result

        except (ValueError, RuntimeError, OSError, IOError) as e:
            logger.error(f"Error analyzing image: {e}")

            # Return error result
            error_response = self.response_generator.template_generator.generate_error_response("general")

            return AnalysisResult(
                image_path=str(image_path) if isinstance(image_path, (str, Path)) else "uploaded_image",
                predicted_class="error",
                confidence=0.0,
                top_3_predictions=[],
                response_text=error_response,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )

    def _preprocess_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """Preprocess image for model input"""

        # Load image if path
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Convert to numpy array
        image_np = np.array(image)

        # Apply transformations
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image']

        # Add batch dimension and move to device
        return image_tensor.unsqueeze(0).to(self.device)

    def _classify_image(
        self,
        image_tensor: torch.Tensor
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Run classification on preprocessed image"""

        with torch.no_grad():
            # Get model predictions
            outputs = self.model(image_tensor)

            # Apply softmax to get probabilities
            probs = F.softmax(outputs, dim=1)

            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probs[0], k=min(3, len(self.CLASSES)))

            # Convert to class names and percentages
            predicted_class = self.CLASSES[top3_indices[0].item()]
            confidence = top3_probs[0].item() * 100

            top_3_predictions = [
                (self.CLASSES[idx.item()], prob.item() * 100)
                for idx, prob in zip(top3_indices, top3_probs)
            ]

        return predicted_class, confidence, top_3_predictions

    def _update_conversation_history(
        self,
        result: AnalysisResult,
        user_question: Optional[str] = None
    ):
        """Update conversation history for context"""

        # Add user question if provided
        if user_question:
            self.conversation_history.append({
                "role": "user",
                "content": user_question
            })

        # Add analysis result
        self.conversation_history.append({
            "role": "assistant",
            "content": result.response_text,
            "metadata": {
                "predicted_class": result.predicted_class,
                "confidence": result.confidence
            }
        })

        # Keep only last 10 exchanges
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def answer_followup(self, question: str) -> str:
        """
        Answer follow-up questions based on previous analysis

        Args:
            question: User's follow-up question

        Returns:
            Response to the question
        """
        if not self.conversation_history:
            return "I need to analyze an ultrasound image first before I can answer questions about it."

        # Get last analysis from history
        last_analysis = None
        for msg in reversed(self.conversation_history):
            if msg.get("role") == "assistant" and "metadata" in msg:
                last_analysis = msg["metadata"]
                break

        if not last_analysis:
            return "I need to analyze an ultrasound image first before I can answer questions about it."

        # Generate response using context
        if hasattr(self.response_generator, 'openai_generator') and self.response_generator.openai_generator:
            # Try to use OpenAI for follow-up
            from src.chatbot.openai_integration import AnalysisContext

            class_info = self.response_generator.template_generator.class_descriptions.get(
                last_analysis["predicted_class"], {}
            )

            context = AnalysisContext(
                predicted_class=last_analysis["predicted_class"],
                confidence=last_analysis["confidence"],
                class_description=class_info.get('name', last_analysis["predicted_class"]),
                visible_structures=class_info.get('structures', []),
                clinical_purpose=class_info.get('clinical_purpose', 'medical assessment'),
                measurements=class_info.get('measurements', [])
            )

            response = self.response_generator.openai_generator.answer_followup_question(
                question=question,
                context=context,
                conversation_history=self.conversation_history
            )

            if response:
                return response

        # Fallback to simple response
        return (
            f"Based on my previous analysis showing a {last_analysis['predicted_class']} view "
            f"with {last_analysis['confidence']:.1f}% confidence, I can provide limited information. "
            "For specific medical questions, please consult with a healthcare professional."
        )

    def batch_analyze(
        self,
        image_paths: List[Union[str, Path]],
        generate_summary: bool = True
    ) -> List[AnalysisResult]:
        """
        Analyze multiple ultrasound images

        Args:
            image_paths: List of image paths
            generate_summary: Whether to generate batch summary

        Returns:
            List of analysis results
        """
        results = []

        for image_path in image_paths:
            logger.info(f"Analyzing: {image_path}")
            result = self.analyze_image(image_path, include_details=False)
            results.append(result)

        if generate_summary and results:
            # Convert to format expected by response generator
            classification_results = [
                ClassificationResult(
                    predicted_class=r.predicted_class,
                    confidence=r.confidence,
                    top_3_predictions=r.top_3_predictions
                )
                for r in results
            ]

            summary = self.response_generator.template_generator.generate_batch_summary(
                classification_results
            )

            logger.info(f"\n{summary}")

        return results

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")


# Standalone function for quick testing
def test_chatbot(image_path: str, model_path: Optional[str] = None) -> None:
    """Quick test function for the chatbot"""

    print("Initializing chatbot...")
    chatbot = UltrasoundChatbot(
        model_path=model_path or "models/best_model_efficientnet_b0_12class.pth",
        use_openai=False  # Use templates for testing
    )

    print(f"\nAnalyzing image: {image_path}")
    result = chatbot.analyze_image(image_path)

    print(f"\nPredicted class: {result.predicted_class}")
    print(f"Confidence: {result.confidence:.2f}%")
    print(f"\nTop 3 predictions:")
    for class_name, conf in result.top_3_predictions:
        print(f"  - {class_name}: {conf:.2f}%")

    print(f"\nGenerated Response:")
    print("-" * 60)
    print(result.response_text)
    print("-" * 60)

    print(f"\nProcessing time: {result.processing_time:.2f} seconds")

    # Test follow-up question
    print("\n\nTesting follow-up question...")
    followup = chatbot.answer_followup("What measurements are typically taken in this view?")
    print(f"Follow-up response:\n{followup}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test ultrasound chatbot')
    parser.add_argument('--image', type=str,
                        default='data/Fetal Ultrasound/Trans-thalamic/Trans-thalamic_001.png',
                        help='Path to ultrasound image')
    parser.add_argument('--model', type=str,
                        default='models/best_model_efficientnet_b0_12class.pth',
                        help='Path to trained model')
    args = parser.parse_args()

    test_chatbot(args.image, args.model)