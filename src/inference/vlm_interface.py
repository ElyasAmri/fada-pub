"""
VLM Interface - Abstract base class and manager for VLM inference
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from PIL import Image


class VLMInterface(ABC):
    """Abstract base class for VLM inference"""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory"""
        pass

    @abstractmethod
    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        Answer a single question about an image

        Args:
            image: PIL Image
            question: Question text

        Returns:
            Answer text
        """
        pass

    def answer_batch(self, image: Image.Image, questions: List[str]) -> List[str]:
        """
        Answer multiple questions about an image (default sequential implementation)

        Args:
            image: PIL Image
            questions: List of question texts

        Returns:
            List of answer texts
        """
        return [self.answer_question(image, q) for q in questions]

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name"""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether model is currently loaded"""
        pass


class VLMManager:
    """Manager for multiple VLM models"""

    def __init__(self) -> None:
        self.models: Dict[str, VLMInterface] = {}

    def register_model(self, key: str, model: VLMInterface) -> None:
        """Register a VLM model"""
        self.models[key] = model

    def get_model(self, key: str) -> Optional[VLMInterface]:
        """Get a registered model"""
        return self.models.get(key)

    def get_all_models(self) -> List[str]:
        """Get list of registered model keys"""
        return list(self.models.keys())

    def unload_all(self) -> None:
        """Unload all models from memory"""
        for model in self.models.values():
            if model.is_loaded:
                model.unload()


__all__ = ['VLMInterface', 'VLMManager']
