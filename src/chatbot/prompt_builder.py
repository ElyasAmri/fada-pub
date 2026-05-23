"""
Prompt Builder - Constructs prompts for OpenAI API calls
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AnalysisContext:
    """Context for generating AI responses"""
    predicted_class: str
    confidence: float
    class_description: str
    visible_structures: List[str]
    clinical_purpose: str
    measurements: List[str]
    top_3_predictions: Optional[List[Tuple[str, float]]] = None
    additional_findings: Optional[Dict] = None


# System prompt for medical accuracy and appropriate tone
MEDICAL_SYSTEM_PROMPT = """You are an AI assistant helping interpret fetal ultrasound images for educational purposes.
Your responses should be:
1. Medically accurate but accessible to non-medical users
2. Clear about the limitations of AI analysis
3. Encouraging users to consult healthcare professionals
4. Professional yet friendly in tone
5. Focused on explaining what the image shows and its clinical relevance

Important guidelines:
- Never diagnose conditions or abnormalities
- Always include appropriate disclaimers
- Use medical terms but explain them simply
- Be helpful but not prescriptive
- Acknowledge uncertainty when confidence is low"""


class PromptBuilder:
    """Builds prompts for various OpenAI API calls"""

    def __init__(self, system_prompt: str = MEDICAL_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    def build_analysis_prompt(
        self,
        context: AnalysisContext,
        user_question: Optional[str] = None
    ) -> str:
        """Build prompt for analysis response"""
        prompt_parts = [
            f"I've analyzed a fetal ultrasound image with the following results:",
            f"- Detected view: {context.predicted_class} ({context.class_description})",
            f"- Confidence level: {context.confidence:.1f}%",
            f"- Clinical purpose: {context.clinical_purpose}"
        ]

        if context.visible_structures:
            prompt_parts.append(f"- Typical structures visible: {', '.join(context.visible_structures)}")

        if context.measurements:
            prompt_parts.append(f"- Common measurements: {', '.join(context.measurements)}")

        # Add alternatives if confidence is low
        if context.confidence < 70 and context.top_3_predictions:
            alternatives = [f"{cls} ({conf:.1f}%)" for cls, conf in context.top_3_predictions[1:3]]
            prompt_parts.append(f"- Alternative possibilities: {', '.join(alternatives)}")

        prompt_parts.append("")

        if user_question:
            prompt_parts.append(f"The user asks: '{user_question}'")
            prompt_parts.append("Please provide a helpful response addressing their question while explaining what this ultrasound view shows.")
        else:
            prompt_parts.append("Please provide a clear, informative explanation of what this ultrasound view shows and its clinical significance.")

        # Add confidence-based instruction
        if context.confidence < 60:
            prompt_parts.append("\nNote: Given the low confidence, emphasize uncertainty and recommend professional verification.")
        elif context.confidence < 80:
            prompt_parts.append("\nNote: Given moderate confidence, acknowledge some uncertainty while being informative.")

        return "\n".join(prompt_parts)

    def build_followup_prompt(
        self,
        question: str,
        context: AnalysisContext
    ) -> str:
        """Build prompt for follow-up questions"""
        structures = ', '.join(context.visible_structures) if context.visible_structures else 'standard anatomical features'

        return f"""Based on the ultrasound analysis showing a {context.class_description} view
with {context.confidence:.1f}% confidence, the user asks: "{question}"

Context: This view is used for {context.clinical_purpose}.
Visible structures typically include: {structures}.

Please provide a clear, helpful answer while maintaining medical accuracy and appropriate disclaimers."""

    def build_educational_prompt(
        self,
        predicted_class: str,
        class_description: str
    ) -> str:
        """Build prompt for educational content"""
        return f"""Please create educational content about the {class_description} ultrasound view.

Include:
1. What this view shows and why it's important
2. What healthcare providers look for in this view
3. When during pregnancy this scan is typically performed
4. What measurements or assessments are made
5. Simple explanation of the anatomy visible

Keep the explanation accessible to expecting parents while being medically accurate.
Format with clear sections and bullet points where appropriate."""

    def build_messages(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Build message list for API call"""
        messages = [{"role": "system", "content": self.system_prompt}]

        if conversation_history:
            messages.extend(conversation_history[-5:])

        messages.append({"role": "user", "content": prompt})

        return messages
