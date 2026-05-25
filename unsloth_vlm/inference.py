"""
Interactive Inference for Fine-tuned Qwen3-VL

Run inference on individual ultrasound images to test the model.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import torch
from PIL import Image
from unsloth import FastVisionModel

from prepare_dataset import Q7_PROMPT


# Default paths
BASE_MODEL = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
ADAPTER_DIR = Path(__file__).parent / "outputs" / "qwen3vl_ultrasound" / "lora_adapters"
DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "Fetal Ultrasound"


class UltrasoundAnalyzer:
    """Wrapper class for ultrasound image analysis."""

    def __init__(self, adapter_dir: Optional[Path] = None, load_in_4bit: bool = True):
        """
        Initialize the analyzer.

        Args:
            adapter_dir: Path to LoRA adapters (None for base model)
            load_in_4bit: Whether to use 4-bit quantization
        """
        self.adapter_dir = adapter_dir
        self.model = None
        self.tokenizer = None
        self._load_model(load_in_4bit)

    def _load_model(self, load_in_4bit: bool):
        """Load model with optional adapters."""
        print(f"Loading base model: {BASE_MODEL}")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            BASE_MODEL,
            load_in_4bit=load_in_4bit,
        )

        if self.adapter_dir and self.adapter_dir.exists():
            print(f"Loading LoRA adapters from: {self.adapter_dir}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(self.adapter_dir))
            print("Adapters loaded!")

        # Set to inference mode
        FastVisionModel.for_inference(self.model)
        print("Model ready for inference!")

    def analyze(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Analyze an ultrasound image.

        Args:
            image_path: Path to the ultrasound image
            prompt: Custom prompt (uses Q7 prompt by default)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Model's analysis of the image
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Use default prompt if not provided
        if prompt is None:
            prompt = Q7_PROMPT

        return self._generate(image, prompt, max_new_tokens, temperature)

    def analyze_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Analyze a PIL Image directly.

        Args:
            image: PIL Image object
            prompt: Custom prompt (uses Q7 prompt by default)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Model's analysis of the image
        """
        if prompt is None:
            prompt = Q7_PROMPT

        return self._generate(image, prompt, max_new_tokens, temperature)

    def _generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Internal generation method."""
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize with image
        inputs = self.tokenizer(
            input_text,
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()


def interactive_mode(analyzer: UltrasoundAnalyzer):
    """Run interactive inference mode."""
    print("\n" + "=" * 60)
    print("Interactive Ultrasound Analysis")
    print("=" * 60)
    print("Enter image path to analyze (or 'quit' to exit)")
    print("Use 'prompt:' prefix to set custom prompt")
    print()

    custom_prompt = None

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if user_input.lower().startswith("prompt:"):
                custom_prompt = user_input[7:].strip()
                print(f"Custom prompt set: {custom_prompt[:50]}...")
                continue

            if user_input.lower() == "reset":
                custom_prompt = None
                print("Prompt reset to default")
                continue

            # Treat input as image path
            image_path = Path(user_input)

            # Try to find the image
            if not image_path.exists():
                # Try relative to data root
                alt_path = DATA_ROOT / user_input
                if alt_path.exists():
                    image_path = alt_path
                else:
                    print(f"Image not found: {image_path}")
                    continue

            print(f"Analyzing: {image_path}")
            print("-" * 40)

            response = analyzer.analyze(
                str(image_path),
                prompt=custom_prompt
            )

            print(f"\nAnalysis:\n{response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_inference(analyzer: UltrasoundAnalyzer, image_paths: list, output_file: Optional[str] = None):
    """Run batch inference on multiple images."""
    import json
    from tqdm import tqdm

    results = []

    for path in tqdm(image_paths, desc="Analyzing"):
        try:
            response = analyzer.analyze(str(path))
            results.append({
                "image": str(path),
                "analysis": response,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "image": str(path),
                "analysis": None,
                "status": f"error: {e}"
            })

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen3-VL")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to single image to analyze")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Directory of images for batch inference")
    parser.add_argument("--adapter-dir", type=str, default=str(ADAPTER_DIR),
                        help="Path to LoRA adapters")
    parser.add_argument("--no-adapters", action="store_true",
                        help="Use base model without fine-tuning")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (default: Q7 normality assessment)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for batch results")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")

    args = parser.parse_args()

    # Determine adapter path
    adapter_path = None if args.no_adapters else Path(args.adapter_dir)
    if adapter_path and not adapter_path.exists():
        print(f"Warning: Adapter path not found: {adapter_path}")
        print("Using base model without fine-tuning")
        adapter_path = None

    # Load analyzer
    analyzer = UltrasoundAnalyzer(adapter_dir=adapter_path)

    # Run based on mode
    if args.interactive:
        interactive_mode(analyzer)

    elif args.image:
        # Single image inference
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return

        print(f"Analyzing: {image_path}")
        response = analyzer.analyze(str(image_path), prompt=args.prompt)
        print(f"\nAnalysis:\n{response}")

    elif args.images_dir:
        # Batch inference
        images_dir = Path(args.images_dir)
        if not images_dir.exists():
            print(f"Error: Directory not found: {images_dir}")
            return

        image_paths = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        print(f"Found {len(image_paths)} images in {images_dir}")

        results = batch_inference(analyzer, image_paths, args.output)
        print(f"\nProcessed {len(results)} images")

    else:
        # Default to interactive mode
        interactive_mode(analyzer)


if __name__ == "__main__":
    main()
