"""
OpenAI Model Configuration and Selection
Updated with latest GPT models (as of 2024)
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for OpenAI models"""
    name: str
    display_name: str
    description: str
    context_window: int
    max_output: int
    recommended_for: str
    cost_tier: str  # low, medium, high
    speed: str  # slow, medium, fast


# Available OpenAI models with their configurations
AVAILABLE_MODELS: Dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        display_name="GPT-4o (Omni)",
        description="Most capable multimodal model with vision, function calling, and JSON mode",
        context_window=128000,
        max_output=4096,
        recommended_for="Complex medical analysis requiring highest accuracy",
        cost_tier="high",
        speed="medium"
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        display_name="GPT-4o Mini",
        description="Smaller, faster version of GPT-4o, great balance of capability and cost",
        context_window=128000,
        max_output=16384,
        recommended_for="Standard medical responses with good accuracy",
        cost_tier="low",
        speed="fast"
    ),
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        display_name="GPT-4 Turbo",
        description="Previous generation turbo model with vision capabilities",
        context_window=128000,
        max_output=4096,
        recommended_for="Legacy compatibility, still very capable",
        cost_tier="medium",
        speed="medium"
    ),
    "gpt-4-turbo-preview": ModelConfig(
        name="gpt-4-turbo-preview",
        display_name="GPT-4 Turbo Preview",
        description="Latest preview with improved instruction following",
        context_window=128000,
        max_output=4096,
        recommended_for="Testing newest capabilities",
        cost_tier="medium",
        speed="medium"
    ),
    "gpt-4": ModelConfig(
        name="gpt-4",
        display_name="GPT-4",
        description="Original GPT-4 model, highly capable but slower",
        context_window=8192,
        max_output=8192,
        recommended_for="When consistency with older deployments is needed",
        cost_tier="high",
        speed="slow"
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        description="Fast, cost-effective model for simpler tasks",
        context_window=16385,
        max_output=4096,
        recommended_for="Simple responses where cost is critical",
        cost_tier="very-low",
        speed="very-fast"
    )
}

# Recommended models for medical chatbot use cases
RECOMMENDED_MODELS = {
    "production": "gpt-4o-mini",  # Best balance for production
    "development": "gpt-4o-mini",  # Cost-effective for testing
    "high_accuracy": "gpt-4o",     # When accuracy is paramount
    "budget": "gpt-3.5-turbo"      # Minimum cost option
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get configuration for a specific model

    Args:
        model_name: Name of the OpenAI model

    Returns:
        ModelConfig object or None if not found
    """
    return AVAILABLE_MODELS.get(model_name)


def get_recommended_model(use_case: str = "production") -> str:
    """
    Get recommended model for a specific use case

    Args:
        use_case: One of 'production', 'development', 'high_accuracy', 'budget'

    Returns:
        Model name string
    """
    return RECOMMENDED_MODELS.get(use_case, "gpt-4o-mini")


def validate_model(model_name: str) -> bool:
    """
    Check if a model name is valid and available

    Args:
        model_name: Name of the model to validate

    Returns:
        True if valid, False otherwise
    """
    return model_name in AVAILABLE_MODELS


def get_model_comparison() -> str:
    """
    Generate a comparison table of available models

    Returns:
        Formatted string with model comparison
    """
    comparison = "Model Comparison for Medical Chatbot:\n\n"

    for model_name, config in AVAILABLE_MODELS.items():
        comparison += f"**{config.display_name}**\n"
        comparison += f"  - Context: {config.context_window:,} tokens\n"
        comparison += f"  - Speed: {config.speed}\n"
        comparison += f"  - Cost: {config.cost_tier}\n"
        comparison += f"  - Best for: {config.recommended_for}\n\n"

    return comparison


# Model selection logic based on requirements
class ModelSelector:
    """Intelligent model selection based on requirements"""

    @staticmethod
    def select_model(
        require_vision: bool = False,
        require_speed: bool = False,
        require_accuracy: bool = True,
        budget_conscious: bool = False
    ) -> str:
        """
        Select best model based on requirements

        Args:
            require_vision: Need vision/image capabilities
            require_speed: Prioritize response speed
            require_accuracy: Prioritize accuracy
            budget_conscious: Minimize costs

        Returns:
            Recommended model name
        """
        if budget_conscious:
            return "gpt-3.5-turbo" if not require_accuracy else "gpt-4o-mini"

        if require_vision:
            return "gpt-4o" if require_accuracy else "gpt-4o-mini"

        if require_speed:
            return "gpt-4o-mini"

        if require_accuracy:
            return "gpt-4o"

        # Default balanced choice
        return "gpt-4o-mini"


# Export key configurations
DEFAULT_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-3.5-turbo"
MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500

# Medical-specific prompting parameters
MEDICAL_TEMPERATURE = 0.6  # Lower for more consistent medical responses
MEDICAL_MAX_TOKENS = 800  # Longer for detailed explanations
MEDICAL_SYSTEM_PROMPT_ADDITION = """
Always maintain medical accuracy while being accessible.
Use proper medical terminology but explain it simply.
Never provide diagnoses, only educational information.
"""

if __name__ == "__main__":
    # Test model configuration
    print("Available OpenAI Models for Medical Chatbot:\n")
    print(get_model_comparison())

    print("\nRecommended Models by Use Case:")
    for use_case, model in RECOMMENDED_MODELS.items():
        config = get_model_config(model)
        print(f"  {use_case}: {config.display_name}")

    print("\nIntelligent Selection Examples:")
    print(f"  High accuracy needed: {ModelSelector.select_model(require_accuracy=True)}")
    print(f"  Speed priority: {ModelSelector.select_model(require_speed=True)}")
    print(f"  Budget conscious: {ModelSelector.select_model(budget_conscious=True)}")
    print(f"  Vision capabilities: {ModelSelector.select_model(require_vision=True)}")