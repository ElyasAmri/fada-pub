"""
BLIP-2 VQA Model for Fetal Ultrasound Analysis
Visual Question Answering using fine-tuned BLIP-2 with LoRA
"""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel
from pathlib import Path
from typing import Union, List, Dict
from PIL import Image


class UltrasoundVQA:
    """
    Visual Question Answering model for fetal ultrasound images

    Uses BLIP-2 OPT-2.7B base model with LoRA adapters trained on
    Non_standard_NT images with 8 medical questions per image.
    """

    # Standard 8 questions -- canonical source in src.config.questions.
    # Historical wording differed slightly; now unified.
    from src.config.questions import QUESTIONS as STANDARD_QUESTIONS

    def __init__(self,
                 model_path: Union[str, Path] = "outputs/blip2_1epoch/final_model",
                 base_model: str = "Salesforce/blip2-opt-2.7b",
                 device: str = "auto"):
        """
        Initialize VQA model

        Args:
            model_path: Path to LoRA adapter weights
            base_model: Base BLIP-2 model name
            device: Device to run on ("cuda", "cpu", or "auto")
        """
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Load BLIP-2 base model and LoRA adapters"""
        if self.model is not None:
            return  # Already loaded

        print(f"Loading BLIP-2 VQA model...")
        print(f"  Base model: {self.base_model}")
        print(f"  LoRA adapters: {self.model_path}")
        print(f"  Device: {self.device}")

        # Load processor
        self.processor = Blip2Processor.from_pretrained(self.base_model)

        # Load base model with 8-bit quantization
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            base = Blip2ForConditionalGeneration.from_pretrained(
                self.base_model,
                quantization_config=quantization_config,
                device_map=self.device if self.device == "auto" else {"": self.device}
            )
        except ImportError:
            # Fallback to old method
            base = Blip2ForConditionalGeneration.from_pretrained(
                self.base_model,
                load_in_8bit=True,
                device_map=self.device if self.device == "auto" else {"": self.device}
            )

        # Load LoRA adapters if they exist
        if self.model_path.exists():
            self.model = PeftModel.from_pretrained(base, str(self.model_path))
            print("  LoRA adapters loaded successfully")
        else:
            self.model = base
            print("  Warning: LoRA adapters not found, using base model")

        self.model.eval()
        print(f"  Model loaded (Memory: {self.model.get_memory_footprint() / 1e9:.2f} GB)")

    def answer_question(self,
                       image: Union[Image.Image, str, Path],
                       question: str,
                       max_new_tokens: int = 100) -> str:
        """
        Answer a question about an ultrasound image

        Args:
            image: PIL Image or path to image
            question: Question to answer
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated answer string
        """
        if self.model is None:
            self.load_model()

        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        # Format prompt
        prompt = f"Question: {question} Answer:"

        # Process inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        # Generate answer with conservative parameters for fine-tuned model
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=5,
                num_beams=3,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                do_sample=False,  # Deterministic generation for medical model
                early_stopping=True
            )

        # Decode answer
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Remove the prompt from answer if it's included
        if answer.startswith(prompt):
            answer = answer[len(prompt):].strip()

        # Post-process to remove repetitive phrases
        answer = self._clean_repetitions(answer)

        return answer

    def _clean_repetitions(self, text: str) -> str:
        """
        Clean repetitive patterns from generated text

        Args:
            text: Generated text that may contain repetitions

        Returns:
            Cleaned text with repetitions removed
        """
        # Split into sentences
        sentences = text.split('. ')

        # Remove consecutive duplicate phrases
        cleaned = []
        for sentence in sentences:
            # Check if this sentence is already in cleaned (avoid exact duplicates)
            if sentence not in cleaned:
                # Check for repetitive patterns within the sentence
                words = sentence.split()
                if len(words) > 5:
                    # Look for repeating sequences of 3+ words
                    for i in range(3, min(10, len(words)//2)):
                        pattern = ' '.join(words[:i])
                        rest = ' '.join(words[i:])
                        if rest.startswith(pattern):
                            # Found repetition, keep only first occurrence
                            sentence = pattern
                            break
                cleaned.append(sentence)

        # Rejoin sentences
        result = '. '.join(cleaned)
        if not result.endswith('.'):
            result += '.'

        return result

    def answer_all_questions(self,
                            image: Union[Image.Image, str, Path],
                            questions: List[str] = None,
                            max_new_tokens: int = 100) -> Dict[str, str]:
        """
        Answer multiple questions about an image

        Args:
            image: PIL Image or path to image
            questions: List of questions (defaults to STANDARD_QUESTIONS)
            max_new_tokens: Maximum tokens per answer

        Returns:
            Dictionary mapping questions to answers
        """
        if questions is None:
            questions = self.STANDARD_QUESTIONS

        # Load image once if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        results = {}
        for question in questions:
            answer = self.answer_question(image, question, max_new_tokens)
            results[question] = answer

        return results

    def get_question_shortcuts(self) -> Dict[str, str]:
        """
        Get shortened versions of standard questions for UI

        Returns:
            Dictionary mapping short names to full questions
        """
        shortcuts = {
            "Anatomical Structures": self.STANDARD_QUESTIONS[0],
            "Fetal Orientation": self.STANDARD_QUESTIONS[1],
            "Imaging Plane": self.STANDARD_QUESTIONS[2],
            "Biometric Measurements": self.STANDARD_QUESTIONS[3],
            "Gestational Age": self.STANDARD_QUESTIONS[4],
            "Image Quality": self.STANDARD_QUESTIONS[5],
            "Normality Assessment": self.STANDARD_QUESTIONS[6],
            "Clinical Recommendations": self.STANDARD_QUESTIONS[7]
        }
        return shortcuts

    def unload_model(self) -> None:
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            print("VQA model unloaded")


if __name__ == "__main__":
    # Test the VQA model
    print("Testing UltrasoundVQA model...")

    vqa = UltrasoundVQA()

    # Check if model path exists
    if vqa.model_path.exists():
        print(f"Model path exists: {vqa.model_path}")

        # Test loading
        vqa.load_model()

        # Test with dummy image
        dummy_image = Image.new('RGB', (224, 224), color='gray')

        print("\nTesting single question:")
        answer = vqa.answer_question(
            dummy_image,
            "What organ is visible in this image?"
        )
        print(f"Answer: {answer}")

        print("\nStandard questions:")
        shortcuts = vqa.get_question_shortcuts()
        for short_name in shortcuts:
            print(f"  - {short_name}")
    else:
        print(f"Model path does not exist: {vqa.model_path}")
        print("Please train the model first using notebooks/train_blip2_1epoch.ipynb")
