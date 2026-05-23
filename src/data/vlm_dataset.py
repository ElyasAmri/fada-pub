"""
VLM Dataset Converter - Convert API responses to fine-tuning format

Converts MedGemma/Gemini/GPT-4o responses to conversation format
compatible with Unsloth and HuggingFace VLM fine-tuning.

Usage:
    # Generate training data from MedGemma responses
    python -m src.data.vlm_dataset --source medgemma --output data/vlm_training/

    # Generate from Gemini checkpoint
    python -m src.data.vlm_dataset --source gemini --output data/vlm_training/
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd

from src.data.dataset_splits import load_splits, DATA_ROOT, SPLITS_FILE
from src.config.questions import QUESTIONS, QUESTION_COLUMNS
from experiments.evaluation.config import VLM_SYSTEM_PROMPT


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "api_models" / "results"
VLM_TRAINING_DIR = PROJECT_ROOT / "data" / "vlm_training"

# Default pseudo-label sources
MEDGEMMA_EXCEL = RESULTS_DIR / "medgemma_4b_responses.xlsx"
GEMINI_CHECKPOINT = RESULTS_DIR / "checkpoint_gemini_gemini-3-flash-preview.json"
GEMINI_COMPLETE = RESULTS_DIR / "gemini_annotations_complete.json"


@dataclass
class VLMSample:
    """Single VLM training sample"""
    image_path: str           # Relative path to image
    question: str             # Full question text
    answer: str               # Model response (pseudo-label)
    category: str             # Fetal ultrasound category
    question_id: int          # Question number (1-8)
    source_model: str         # Model that generated the response


def load_medgemma_responses(excel_path: Path = MEDGEMMA_EXCEL) -> Dict[str, Dict[str, str]]:
    """
    Load MedGemma responses from Excel file.

    Returns:
        Dict mapping image_path -> {question_col: response}
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"MedGemma results not found: {excel_path}")

    df = pd.read_excel(excel_path)

    responses = {}
    for _, row in df.iterrows():
        image_path = row['Full Path']  # e.g., "Abdomen/Abdomen_001.png"

        image_responses = {}
        for col in QUESTION_COLUMNS:
            if col in row and pd.notna(row[col]):
                image_responses[col] = str(row[col])

        if image_responses:
            responses[image_path] = image_responses

    return responses


def load_gemini_checkpoint(checkpoint_path: Path = GEMINI_CHECKPOINT) -> Dict[str, Dict[str, str]]:
    """
    Load Gemini responses from checkpoint JSON.

    Returns:
        Dict mapping image_path -> {question_col: response}
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Gemini checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    responses = {}
    for entry in checkpoint.get('responses', []):
        image_path = entry.get('image_path', '')

        if not image_path:
            continue

        image_responses = {}
        for i, q_col in enumerate(QUESTION_COLUMNS):
            key = f"q{i+1}"
            if key in entry and entry[key]:
                image_responses[q_col] = str(entry[key])

        if image_responses:
            responses[image_path] = image_responses

    return responses


def load_gemini_complete(json_path: Path = GEMINI_COMPLETE) -> Dict[str, Dict[str, str]]:
    """
    Load Gemini responses from merged complete JSON file.

    This loader handles the format from gemini_annotations_complete.json:
    {
        "completed_images": {
            "Category/image.png": {
                "category": "...",
                "image": "...",
                "questions": [
                    {"question_idx": 0, "question": "...", "response": "...", ...},
                    ...
                ]
            }
        }
    }

    Returns:
        Dict mapping image_path -> {question_col: response}
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Gemini complete file not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    responses = {}
    for image_key, entry in data.get('completed_images', {}).items():
        image_responses = {}

        for q in entry.get('questions', []):
            q_idx = q.get('question_idx', -1)
            response = q.get('response', '')
            error = q.get('error')

            # Skip if invalid index, no response, or has error
            if q_idx < 0 or q_idx >= len(QUESTION_COLUMNS):
                continue
            if not response or error:
                continue

            q_col = QUESTION_COLUMNS[q_idx]
            image_responses[q_col] = str(response)

        if image_responses:
            responses[image_key] = image_responses

    return responses


def create_conversation_format(
    image_path: str,
    question: str,
    answer: str,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a single conversation in Qwen3-VL compatible format.

    Format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
            {"role": "assistant", "content": "..."}
        ],
        "images": ["path/to/image.png"]
    }
    """
    if system_prompt is None:
        system_prompt = VLM_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        },
        {"role": "assistant", "content": answer}
    ]

    return {
        "messages": messages,
        "images": [image_path]
    }


def create_multiturn_conversation(
    image_path: str,
    qa_pairs: List[tuple],  # List of (question, answer) tuples
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a multi-turn conversation with all 8 Q&A pairs for one image.

    This format may be more efficient for training as it reuses the image.
    """
    if system_prompt is None:
        system_prompt = VLM_SYSTEM_PROMPT

    messages = [{"role": "system", "content": system_prompt}]

    for i, (question, answer) in enumerate(qa_pairs):
        if i == 0:
            # First question includes the image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            })
        else:
            # Subsequent questions are text-only (same image context)
            messages.append({
                "role": "user",
                "content": question
            })

        messages.append({"role": "assistant", "content": answer})

    return {
        "messages": messages,
        "images": [image_path]
    }


def convert_to_vlm_dataset(
    responses: Dict[str, Dict[str, str]],
    output_dir: Path,
    source_model: str,
    splits_path: Path = SPLITS_FILE,
    format: str = "single",  # "single" or "multiturn"
    max_response_length: int = 2000,  # Truncate long responses
) -> Dict[str, int]:
    """
    Convert responses to VLM training format and save to JSONL files.

    Args:
        responses: Dict mapping image_path -> {question_col: response}
        output_dir: Directory to save JSONL files
        source_model: Name of source model (for metadata)
        splits_path: Path to dataset splits JSON
        format: "single" for one Q&A per sample, "multiturn" for all Q&A per image
        max_response_length: Maximum response length (truncate longer)

    Returns:
        Dict with counts per split
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset splits
    splits_data = load_splits(splits_path)

    # Create mapping from image path to split
    image_to_split = {}
    for split_name in ['train', 'val', 'test']:
        for category, images in splits_data['splits'][split_name].items():
            for img_path in images:
                image_to_split[img_path] = split_name

    # Prepare output files
    output_files = {
        'train': open(output_dir / f"{source_model}_train.jsonl", 'w'),
        'val': open(output_dir / f"{source_model}_val.jsonl", 'w'),
        'test': open(output_dir / f"{source_model}_test.jsonl", 'w'),
    }

    counts = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0}

    try:
        for image_path, image_responses in responses.items():
            # Determine split
            split = image_to_split.get(image_path)
            if split is None:
                counts['skipped'] += 1
                continue

            # Get relative image path for training
            rel_image_path = image_path  # e.g., "Abdomen/Abdomen_001.png"

            if format == "multiturn":
                # Create multi-turn conversation with all Q&A
                qa_pairs = []
                for i, q_col in enumerate(QUESTION_COLUMNS):
                    if q_col in image_responses:
                        answer = image_responses[q_col]
                        if len(answer) > max_response_length:
                            answer = answer[:max_response_length] + "..."
                        qa_pairs.append((QUESTIONS[i], answer))

                if qa_pairs:
                    conversation = create_multiturn_conversation(rel_image_path, qa_pairs)
                    output_files[split].write(json.dumps(conversation) + '\n')
                    counts[split] += 1

            else:  # single format
                # Create separate sample for each Q&A pair
                for i, q_col in enumerate(QUESTION_COLUMNS):
                    if q_col in image_responses:
                        answer = image_responses[q_col]
                        if len(answer) > max_response_length:
                            answer = answer[:max_response_length] + "..."

                        conversation = create_conversation_format(
                            rel_image_path,
                            QUESTIONS[i],
                            answer
                        )
                        output_files[split].write(json.dumps(conversation) + '\n')
                        counts[split] += 1

    finally:
        for f in output_files.values():
            f.close()

    # Save metadata
    metadata = {
        'source_model': source_model,
        'format': format,
        'questions': QUESTIONS,
        'counts': counts,
        'max_response_length': max_response_length,
    }

    with open(output_dir / f"{source_model}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return counts


def print_summary(counts: Dict[str, int], source: str) -> None:
    """Print summary of conversion results."""
    print("\n" + "=" * 60)
    print(f"VLM DATASET CONVERSION SUMMARY - {source}")
    print("=" * 60)
    print(f"\nSamples per split:")
    print(f"  Train: {counts.get('train', 0):,}")
    print(f"  Val:   {counts.get('val', 0):,}")
    print(f"  Test:  {counts.get('test', 0):,}")
    print(f"  Skipped (not in splits): {counts.get('skipped', 0):,}")
    print(f"\nTotal: {sum(v for k, v in counts.items() if k != 'skipped'):,}")


def main():
    """Command-line interface for VLM dataset conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert API responses to VLM fine-tuning format"
    )
    parser.add_argument(
        '--source', choices=['medgemma', 'gemini', 'gemini-complete'], default='gemini-complete',
        help='Source of pseudo-labels (default: gemini-complete)'
    )
    parser.add_argument(
        '--output', type=str, default=str(VLM_TRAINING_DIR),
        help='Output directory for JSONL files'
    )
    parser.add_argument(
        '--format', choices=['single', 'multiturn'], default='single',
        help='Conversation format (default: single Q&A per sample)'
    )
    parser.add_argument(
        '--max-length', type=int, default=2000,
        help='Maximum response length (default: 2000)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    print(f"Loading responses from {args.source}...")

    if args.source == 'medgemma':
        responses = load_medgemma_responses()
        source_model = 'medgemma_4b'
    elif args.source == 'gemini-complete':
        responses = load_gemini_complete()
        source_model = 'gemini_complete'
    else:
        responses = load_gemini_checkpoint()
        source_model = 'gemini_3_flash'

    print(f"Loaded {len(responses)} images with responses")

    print(f"\nConverting to {args.format} format...")
    counts = convert_to_vlm_dataset(
        responses=responses,
        output_dir=output_dir,
        source_model=source_model,
        format=args.format,
        max_response_length=args.max_length
    )

    print_summary(counts, args.source)
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
