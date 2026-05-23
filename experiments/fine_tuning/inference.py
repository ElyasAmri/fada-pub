"""
Inference script for fine-tuned Qwen2.5-VL model on fetal ultrasound images.

Usage:
    # Analyze a single image with all 8 questions
    python experiments/fine_tuning/inference.py --image path/to/image.png

    # Ask a specific question
    python experiments/fine_tuning/inference.py --image path/to/image.png --question "What structures are visible?"

    # Interactive mode
    python experiments/fine_tuning/inference.py --interactive

    # Compare base vs fine-tuned model
    python experiments/fine_tuning/inference.py --image path/to/image.png --compare
"""

import argparse
from pathlib import Path
from typing import Optional, List

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel


# Model paths
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = Path(__file__).parent.parent.parent / "models" / "qwen25vl7b_finetuned" / "final"

from src.config.questions import QUESTIONS, QUESTION_SHORT_NAMES
from experiments.evaluation.config import VLM_SYSTEM_PROMPT

SYSTEM_PROMPT = VLM_SYSTEM_PROMPT


def _is_gemma_processor(processor) -> bool:
    name = getattr(getattr(processor, "tokenizer", None), "name_or_path", "") or ""
    return "gemma" in name.lower()


def _decode_generated_text(processor, output_ids, input_len: int, is_gemma: bool) -> str:
    if not is_gemma:
        return processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
    if "<channel|>" in raw:
        raw = raw.split("<channel|>", 1)[1]
    return raw.strip()


def load_model(
    use_adapter: bool = True,
    use_4bit: bool = True,
    device_map: str = "auto"
) -> tuple:
    """Load the model and processor."""
    print(f"Loading base model: {BASE_MODEL_ID}")

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if use_adapter:
        if not ADAPTER_PATH.exists():
            raise FileNotFoundError(f"Adapter not found at {ADAPTER_PATH}")
        print(f"Loading fine-tuned adapter from: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))

    model.eval()
    return model, processor


def generate_response(
    model,
    processor,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Generate a response for the given image and question."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    is_gemma = _is_gemma_processor(processor)
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if is_gemma:
        chat_kwargs["enable_thinking"] = True
    text = processor.apply_chat_template(messages, **chat_kwargs)

    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    input_len = inputs['input_ids'].shape[1]
    return _decode_generated_text(processor, output_ids, input_len, is_gemma)


def analyze_image(
    model,
    processor,
    image_path: str,
    questions: Optional[List[str]] = None,
    max_new_tokens: int = 1024,
) -> dict:
    """Analyze an image with multiple questions."""

    image = Image.open(image_path).convert('RGB')

    if questions is None:
        questions = QUESTIONS

    results = {}
    for i, question in enumerate(questions):
        short_name = QUESTION_SHORT_NAMES[i] if i < len(QUESTION_SHORT_NAMES) else f"Q{i+1}"
        print(f"\n[{short_name}]")

        response = generate_response(model, processor, image, question, max_new_tokens)
        results[short_name] = response
        print(response)

    return results


def compare_models(
    image_path: str,
    question: str,
    max_new_tokens: int = 512,
):
    """Compare base model vs fine-tuned model responses."""

    image = Image.open(image_path).convert('RGB')

    print("=" * 60)
    print("Loading BASE model (no fine-tuning)...")
    print("=" * 60)
    base_model, processor = load_model(use_adapter=False)

    print("\n[BASE MODEL RESPONSE]")
    base_response = generate_response(base_model, processor, image, question, max_new_tokens)
    print(base_response)

    # Free memory
    del base_model
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Loading FINE-TUNED model...")
    print("=" * 60)
    ft_model, processor = load_model(use_adapter=True)

    print("\n[FINE-TUNED MODEL RESPONSE]")
    ft_response = generate_response(ft_model, processor, image, question, max_new_tokens)
    print(ft_response)

    return {"base": base_response, "fine_tuned": ft_response}


def interactive_mode(model, processor):
    """Interactive mode for testing the model."""

    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Commands:")
    print("  load <path>  - Load an image")
    print("  q<1-8>       - Ask standard question (e.g., q1, q3)")
    print("  ask <text>   - Ask a custom question")
    print("  all          - Ask all 8 standard questions")
    print("  quit         - Exit")
    print("=" * 60)

    current_image = None
    current_image_path = None

    while True:
        try:
            user_input = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "quit" or cmd == "exit":
            break

        elif cmd == "load":
            if len(parts) < 2:
                print("Usage: load <image_path>")
                continue
            path = parts[1].strip('"\'')
            if not Path(path).exists():
                print(f"File not found: {path}")
                continue
            current_image = Image.open(path).convert('RGB')
            current_image_path = path
            print(f"Loaded: {path}")

        elif cmd.startswith("q") and len(cmd) == 2 and cmd[1].isdigit():
            if current_image is None:
                print("Load an image first with: load <path>")
                continue
            q_idx = int(cmd[1]) - 1
            if q_idx < 0 or q_idx >= len(QUESTIONS):
                print(f"Invalid question number. Use q1-q{len(QUESTIONS)}")
                continue
            print(f"\n[{QUESTION_SHORT_NAMES[q_idx]}]")
            response = generate_response(model, processor, current_image, QUESTIONS[q_idx])
            print(response)

        elif cmd == "ask":
            if current_image is None:
                print("Load an image first with: load <path>")
                continue
            if len(parts) < 2:
                print("Usage: ask <your question>")
                continue
            question = parts[1]
            print(f"\n[Custom Question]")
            response = generate_response(model, processor, current_image, question)
            print(response)

        elif cmd == "all":
            if current_image is None:
                print("Load an image first with: load <path>")
                continue
            for i, question in enumerate(QUESTIONS):
                print(f"\n[{QUESTION_SHORT_NAMES[i]}]")
                response = generate_response(model, processor, current_image, question)
                print(response)

        else:
            print("Unknown command. Type 'quit' to exit.")


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen2.5-VL model")

    parser.add_argument(
        '--image', type=str, default=None,
        help='Path to ultrasound image'
    )
    parser.add_argument(
        '--question', type=str, default=None,
        help='Custom question to ask (default: all 8 standard questions)'
    )
    parser.add_argument(
        '--question-num', type=int, default=None, choices=range(1, 9),
        help='Standard question number (1-8)'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Compare base vs fine-tuned model'
    )
    parser.add_argument(
        '--interactive', action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--no-adapter', action='store_true',
        help='Use base model without fine-tuning'
    )
    parser.add_argument(
        '--max-tokens', type=int, default=1024,
        help='Maximum tokens to generate'
    )

    args = parser.parse_args()

    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available, inference will be slow!")

    if args.compare:
        if not args.image:
            print("Error: --image required for comparison")
            return
        question = args.question or QUESTIONS[0]
        compare_models(args.image, question, args.max_tokens)

    elif args.interactive:
        model, processor = load_model(use_adapter=not args.no_adapter)
        interactive_mode(model, processor)

    elif args.image:
        model, processor = load_model(use_adapter=not args.no_adapter)

        if args.question:
            # Custom question
            image = Image.open(args.image).convert('RGB')
            print(f"\n[Custom Question]")
            response = generate_response(model, processor, image, args.question, args.max_tokens)
            print(response)
        elif args.question_num:
            # Specific standard question
            image = Image.open(args.image).convert('RGB')
            q_idx = args.question_num - 1
            print(f"\n[{QUESTION_SHORT_NAMES[q_idx]}]")
            response = generate_response(model, processor, image, QUESTIONS[q_idx], args.max_tokens)
            print(response)
        else:
            # All questions
            analyze_image(model, processor, args.image, max_new_tokens=args.max_tokens)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python inference.py --image data/Fetal\\ Ultrasound/Brain/Brain_001.png")
        print("  python inference.py --image image.png --question 'What do you see?'")
        print("  python inference.py --image image.png --compare")
        print("  python inference.py --interactive")


if __name__ == "__main__":
    main()
