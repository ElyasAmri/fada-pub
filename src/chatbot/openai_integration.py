"""
OpenAI API Integration for Enhanced Response Generation
Provides more natural and contextual responses using GPT models
"""

import logging
from typing import Dict, List, Optional, Tuple

from .openai_client import OpenAIClient
from .prompt_builder import PromptBuilder, AnalysisContext, MEDICAL_SYSTEM_PROMPT
from .model_config import DEFAULT_MODEL, MEDICAL_TEMPERATURE, MEDICAL_MAX_TOKENS

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = ['AnalysisContext', 'OpenAIResponseGenerator', 'HybridResponseGenerator']


class OpenAIResponseGenerator:
    """Generate sophisticated responses using OpenAI GPT models"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = MEDICAL_TEMPERATURE,
        max_tokens: int = MEDICAL_MAX_TOKENS
    ):
        """
        Initialize OpenAI response generator

        Args:
            api_key: OpenAI API key (uses env var if not provided)
            model: GPT model to use
            temperature: Response randomness
            max_tokens: Maximum response length
        """
        self.client = OpenAIClient(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.prompt_builder = PromptBuilder()

    def is_available(self) -> bool:
        """Check if OpenAI API is configured and available"""
        return self.client.is_available()

    def generate_response(
        self,
        context: AnalysisContext,
        user_question: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate natural language response using GPT

        Args:
            context: Analysis context with classification results
            user_question: Optional specific question from user

        Returns:
            Generated response or None if API unavailable
        """
        prompt = self.prompt_builder.build_analysis_prompt(context, user_question)
        messages = self.prompt_builder.build_messages(prompt)
        return self.client.call(messages)

    def generate_conversational_response(
        self,
        context: AnalysisContext,
        conversation_history: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Generate response considering conversation history

        Args:
            context: Current analysis context
            conversation_history: Previous messages in conversation

        Returns:
            Generated response or None if API unavailable
        """
        prompt = self.prompt_builder.build_analysis_prompt(context)
        messages = self.prompt_builder.build_messages(prompt, conversation_history)
        return self.client.call(messages)

    def answer_followup_question(
        self,
        question: str,
        context: AnalysisContext,
        conversation_history: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Answer follow-up questions about the ultrasound

        Args:
            question: User's follow-up question
            context: Original analysis context
            conversation_history: Previous messages

        Returns:
            Answer to the question or None if API unavailable
        """
        prompt = self.prompt_builder.build_followup_prompt(question, context)
        messages = self.prompt_builder.build_messages(prompt)
        return self.client.call(messages)

    def generate_educational_content(
        self,
        predicted_class: str,
        class_description: str
    ) -> Optional[str]:
        """
        Generate educational content about a specific ultrasound view

        Args:
            predicted_class: The anatomical view class
            class_description: Description of the view

        Returns:
            Educational content or None if API unavailable
        """
        prompt = self.prompt_builder.build_educational_prompt(predicted_class, class_description)
        messages = self.prompt_builder.build_messages(prompt)
        return self.client.call(messages, temperature=0.6, max_tokens=600)


class HybridResponseGenerator:
    """Combines template-based and OpenAI responses"""

    def __init__(
        self,
        use_openai: bool = True,
        openai_api_key: Optional[str] = None,
        fallback_to_templates: bool = True
    ):
        """
        Initialize hybrid response generator

        Args:
            use_openai: Whether to attempt OpenAI responses
            openai_api_key: OpenAI API key
            fallback_to_templates: Use templates if OpenAI fails
        """
        from .response_generator import ResponseGenerator

        self.template_generator = ResponseGenerator()
        self.openai_generator = None

        if use_openai:
            self.openai_generator = OpenAIResponseGenerator(api_key=openai_api_key)
            if self.openai_generator.is_available():
                logger.info("OpenAI integration enabled")
            else:
                logger.warning("OpenAI API not available, using template responses only")

        self.fallback_to_templates = fallback_to_templates

    def generate_response(
        self,
        predicted_class: str,
        confidence: float,
        top_3_predictions: Optional[List[Tuple[str, float]]] = None,
        user_question: Optional[str] = None,
        use_ai: bool = True
    ) -> str:
        """
        Generate response using best available method

        Args:
            predicted_class: Predicted anatomical view
            confidence: Prediction confidence
            top_3_predictions: Top predictions
            user_question: Optional user question
            use_ai: Whether to try AI response

        Returns:
            Generated response string
        """
        if use_ai and self.openai_generator and self.openai_generator.is_available():
            class_info = self.template_generator.class_descriptions.get(predicted_class, {})

            context = AnalysisContext(
                predicted_class=predicted_class,
                confidence=confidence,
                class_description=class_info.get('name', predicted_class),
                visible_structures=class_info.get('structures', []),
                clinical_purpose=class_info.get('clinical_purpose', 'medical assessment'),
                measurements=class_info.get('measurements', []),
                top_3_predictions=top_3_predictions
            )

            ai_response = self.openai_generator.generate_response(context, user_question)

            if ai_response:
                return ai_response

        if self.fallback_to_templates:
            return self.template_generator.generate_response(
                predicted_class=predicted_class,
                confidence=confidence,
                top_3_predictions=top_3_predictions,
                include_details=True
            )

        return "Unable to generate response at this time."
