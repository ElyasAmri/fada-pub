"""
VLM Evaluation Pipeline: Compare fine-tuned or zero-shot VLM against Gemini ground truth.

Uses embedding similarity to measure how well the model matches
the Gemini annotations used for training.

Usage:
    # Full evaluation with default model (from config.py)
    python experiments/evaluation/evaluate_vlm.py

    # Zero-shot evaluation with any model
    python experiments/evaluation/evaluate_vlm.py --model-id Qwen/Qwen2-VL-2B-Instruct --zero-shot

    # Fine-tuned evaluation with custom model and adapter
    python experiments/evaluation/evaluate_vlm.py --model-id Qwen/Qwen2.5-VL-7B-Instruct --adapter-path path/to/adapter

    # With custom test subset
    python experiments/evaluation/evaluate_vlm.py --test-data outputs/evaluation/test_subset.jsonl

    # Inference only (for vast.ai - saves predictions for later scoring)
    python experiments/evaluation/evaluate_vlm.py --inference-only

    # Scoring only (use cached predictions)
    python experiments/evaluation/evaluate_vlm.py --score-only --predictions outputs/evaluation/predictions.jsonl
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

SCORING_MODE = "ground_truth"  # Fixed from "pseudo_label" (L1)

from .config import (
    BASE_MODEL_ID,
    ADAPTER_PATH,
    STRATIFIED_TEST_DATA,
    OUTPUTS_DIR,
    SYSTEM_PROMPT,
    MAX_NEW_TOKENS,
    GENERATION_TEMPERATURE,
    DEFAULT_EMBEDDING_MODEL,
)


def extract_category_from_path(image_path: str) -> str:
    """Extract category name from image path."""
    parts = Path(image_path).parts
    for i, part in enumerate(parts):
        if part == "Fetal Ultrasound" and i + 1 < len(parts):
            return parts[i + 1]
    return "Unknown"


def load_test_samples(test_path: Path) -> List[Dict]:
    """Load test samples from JSONL file."""
    samples = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_predictions(predictions_path: Path) -> List[Dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


class VLMEvaluator:
    """Evaluate fine-tuned or zero-shot VLM against ground truth annotations."""

    def __init__(
        self,
        model_id: str = BASE_MODEL_ID,
        adapter_path: Optional[Path] = ADAPTER_PATH,
        use_4bit: bool = True,
        device_map: str = "auto"
    ):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.use_4bit = use_4bit
        self.device_map = device_map
        self.model = None
        self.processor = None

    def load_model(self):
        """Load model (optionally with LoRA adapter for fine-tuned evaluation)."""
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

        print(f"Loading base model: {self.model_id}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None

        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Load LoRA adapter if specified
        if self.adapter_path is not None:
            from peft import PeftModel
            print(f"Loading adapter from: {self.adapter_path}")
            if not self.adapter_path.exists():
                raise FileNotFoundError(f"Adapter not found at {self.adapter_path}")
            self.model = PeftModel.from_pretrained(model, str(self.adapter_path))
            print("Fine-tuned model loaded successfully")
        else:
            self.model = model
            print("Zero-shot model loaded successfully")

        self.model.eval()

    def generate_response(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = GENERATION_TEMPERATURE
    ) -> str:
        """Generate response for a single image-question pair."""
        import torch
        from PIL import Image

        with Image.open(image_path) as img:
            image = img.convert('RGB')

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

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        response = self.processor.tokenizer.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True
        )

        return response.strip()

    def run_inference(
        self,
        test_samples: List[Dict],
        checkpoint_interval: int = 50,
        checkpoint_path: Optional[Path] = None,
        start_index: int = 0,
        existing_results: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Run inference on all test samples.

        Each sample has format:
        {
            "messages": [system, user, assistant],
            "images": ["path/to/image.png"]
        }

        Returns list of:
        {
            "sample_id": int,
            "image_path": str,
            "category": str,
            "question": str,
            "ground_truth": str,
            "prediction": str
        }

        Args:
            test_samples: List of test samples to process
            checkpoint_interval: Save checkpoint every N samples
            checkpoint_path: Path to save checkpoints
            start_index: Start processing from this sample index (for resume)
            existing_results: Previously computed results (for resume)
        """
        results = existing_results or []

        for idx, sample in enumerate(tqdm(test_samples[start_index:], desc="Running inference", initial=start_index, total=len(test_samples))):
            image_path = sample['images'][0]
            messages = sample['messages']

            # Extract question from user message
            user_msg = messages[1]  # Second message is user
            question = None
            for content in user_msg['content']:
                if isinstance(content, dict) and content.get('type') == 'text':
                    question = content['text']
                    break

            # Extract reference response from assistant message
            reference_response = messages[2]['content']  # Third message is assistant

            # Extract category from path
            category = extract_category_from_path(image_path)

            # Generate prediction
            try:
                prediction = self.generate_response(image_path, question)
            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
                prediction = ""

            actual_idx = start_index + idx
            results.append({
                "sample_id": actual_idx,
                "image_path": image_path,
                "category": category,
                "question": question,
                "reference_response": reference_response,
                "prediction": prediction
            })

            # Periodic checkpoint save
            if checkpoint_path and (actual_idx + 1) % checkpoint_interval == 0:
                self._save_checkpoint(results, checkpoint_path)
                print(f"\n  Checkpoint saved: {actual_idx + 1}/{len(test_samples)} samples")

        return results

    def _save_checkpoint(self, results: List[Dict], checkpoint_path: Path):
        """Save intermediate results to checkpoint file."""
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


def compute_scores(
    results: List[Dict],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu"
) -> Dict:
    """Compute embedding similarity scores."""
    print("WARNING: Scoring against pseudo-labels (not sonographer GT). Use score_against_gt.py for GT evaluation.")
    from .embedding_scorer import EmbeddingScorer

    predictions = [r['prediction'] for r in results]
    # Backward-compatible: accept both 'reference_response' and legacy 'ground_truth'
    references = [r.get('reference_response', r.get('ground_truth', '')) for r in results]
    categories = [r['category'] for r in results]

    # Initialize scorer (use CPU by default to avoid CUDA compatibility issues)
    scorer = EmbeddingScorer(model_name=embedding_model, device=device)

    # Compute similarities
    similarities = scorer.compute_similarity(predictions, references)

    # Aggregate metrics
    aggregate = scorer.compute_aggregate_metrics(similarities)

    # Per-category breakdown
    category_metrics = scorer.compute_category_metrics(similarities, categories)

    # Sample-level scores
    sample_scores = [
        {
            "sample_id": r['sample_id'],
            "category": r['category'],
            "similarity": float(similarities[i])
        }
        for i, r in enumerate(results)
    ]

    return {
        "aggregate": aggregate,
        "per_category": category_metrics,
        "sample_scores": sample_scores
    }


def print_results_summary(scores: Dict):
    """Print formatted results summary."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    agg = scores['aggregate']
    print(f"\nOverall Similarity Score: {agg['mean_similarity']:.4f}")
    print(f"Standard Deviation:       {agg['std_similarity']:.4f}")
    print(f"Median:                   {agg['median_similarity']:.4f}")
    print(f"Range:                    [{agg['min_similarity']:.4f}, {agg['max_similarity']:.4f}]")
    print(f"25th Percentile:          {agg['percentile_25']:.4f}")
    print(f"75th Percentile:          {agg['percentile_75']:.4f}")
    print(f"Number of Samples:        {agg['num_samples']}")

    print("\nPer-Category Breakdown:")
    print("-" * 60)
    for category, cat_scores in sorted(scores['per_category'].items()):
        mean = cat_scores['mean_similarity']
        std = cat_scores['std_similarity']
        n = cat_scores['num_samples']
        print(f"  {category:35s}: {mean:.4f} (+/- {std:.4f})  n={n}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned or zero-shot VLM against ground truth")

    # Model configuration
    parser.add_argument(
        '--model-id', type=str, default=BASE_MODEL_ID,
        help='HuggingFace model ID (default: from config.py)'
    )
    parser.add_argument(
        '--adapter-path', type=str, default=None,
        help='Path to LoRA adapter (if None and --zero-shot not set, uses config.py default)'
    )
    parser.add_argument(
        '--zero-shot', action='store_true',
        help='Run zero-shot evaluation without adapter'
    )

    # Data and evaluation configuration
    parser.add_argument(
        '--test-data', type=str, default=str(STRATIFIED_TEST_DATA),
        help='Path to test subset JSONL'
    )
    parser.add_argument(
        '--inference-only', action='store_true',
        help='Only run inference, skip scoring (for vast.ai)'
    )
    parser.add_argument(
        '--score-only', action='store_true',
        help='Only run scoring on existing predictions'
    )
    parser.add_argument(
        '--predictions', type=str, default=None,
        help='Path to predictions JSONL (for --score-only)'
    )
    parser.add_argument(
        '--embedding-model', type=str, default=DEFAULT_EMBEDDING_MODEL,
        choices=['all-mpnet-base-v2', 'all-MiniLM-L6-v2'],
        help='Sentence transformer model for embedding similarity'
    )
    parser.add_argument(
        '--no-4bit', action='store_true',
        help='Disable 4-bit quantization (requires more VRAM)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Custom output path for results JSON'
    )

    # Resume arguments
    parser.add_argument(
        '--resume-from', type=str, default=None,
        help='Path to partial predictions JSONL file to resume from'
    )
    parser.add_argument(
        '--start-index', type=int, default=None,
        help='Start evaluation from this sample index (alternative to --resume-from)'
    )

    args = parser.parse_args()

    # Determine adapter path
    if args.zero_shot:
        adapter_path = None
    elif args.adapter_path is not None:
        adapter_path = Path(args.adapter_path)
    else:
        # Use default from config.py
        adapter_path = ADAPTER_PATH

    # Setup output directory
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Print configuration
    print("=" * 70)
    print("VLM Evaluation Pipeline")
    print("=" * 70)
    print(f"Model:           {args.model_id}")
    print(f"Adapter:         {adapter_path if adapter_path else 'None (zero-shot)'}")
    print(f"Test data:       {args.test_data}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Mode:            {'inference-only' if args.inference_only else 'score-only' if args.score_only else 'full'}")

    results = None
    predictions_path = None

    if not args.score_only:
        # Check test data exists
        test_path = Path(args.test_data)
        if not test_path.exists():
            print(f"\nError: Test data not found: {test_path}")
            print(f"Run create_test_subset.py first to generate the test subset.")
            return 1

        # Load test samples
        test_samples = load_test_samples(test_path)
        print(f"\nLoaded {len(test_samples)} test samples")

        # Handle resume
        start_index = 0
        existing_results = None
        if args.resume_from and args.start_index is not None:
            print(
                f"\nWARNING: --resume-from and --start-index are both set. "
                f"--resume-from takes precedence; --start-index={args.start_index} is ignored. "
                f"Using --resume-from to avoid duplicate sample_ids."
            )
        if args.resume_from:
            resume_path = Path(args.resume_from)
            if not resume_path.exists():
                print(f"\nError: Resume file not found: {resume_path}")
                return 1
            existing_results = load_predictions(resume_path)
            start_index = len(existing_results)
            # Deduplicate by sample_id in case resume file already contains overlap
            seen_ids = set()
            deduped = []
            for r in existing_results:
                sid = r.get("sample_id")
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    deduped.append(r)
            if len(deduped) < len(existing_results):
                print(
                    f"\nWARNING: Removed {len(existing_results) - len(deduped)} duplicate "
                    f"sample_ids from resume file."
                )
                existing_results = deduped
                start_index = len(existing_results)
            print(f"\nResuming from sample {start_index} ({len(existing_results)} already completed)")
        elif args.start_index is not None:
            start_index = args.start_index
            print(f"\nStarting from sample {start_index}")

        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"CUDA: {torch.cuda.get_device_name(0)}")
                print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                print("WARNING: CUDA not available, inference will be slow!")
        except ImportError:
            print("WARNING: PyTorch not available")

        # Initialize evaluator
        evaluator = VLMEvaluator(
            model_id=args.model_id,
            adapter_path=adapter_path,
            use_4bit=not args.no_4bit
        )
        evaluator.load_model()

        # Run inference
        checkpoint_path = OUTPUTS_DIR / f"predictions_checkpoint_{timestamp}.jsonl"
        results = evaluator.run_inference(
            test_samples,
            checkpoint_path=checkpoint_path,
            start_index=start_index,
            existing_results=existing_results
        )

        # Save predictions
        predictions_path = OUTPUTS_DIR / f"predictions_{timestamp}.jsonl"
        with open(predictions_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\nPredictions saved to: {predictions_path}")

        # Clean up checkpoint if we completed successfully
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    if args.inference_only:
        print("\nInference complete. Skipping scoring.")
        print("To score later, run with --score-only --predictions <path>")
        return 0

    # Load predictions if score-only mode
    if args.score_only:
        if not args.predictions:
            print("\nError: --predictions required when using --score-only")
            return 1

        predictions_path = Path(args.predictions)
        if not predictions_path.exists():
            print(f"\nError: Predictions file not found: {predictions_path}")
            return 1

        results = []
        with open(predictions_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        print(f"\nLoaded {len(results)} predictions from {predictions_path}")

    # Compute scores
    try:
        print("\nComputing embedding similarity scores...")
        scores = compute_scores(results, embedding_model=args.embedding_model)

        # Build final output
        final_results = {
            "metadata": {
                "timestamp": timestamp,
                "test_data": args.test_data,
                "embedding_model": args.embedding_model,
                "model_id": args.model_id,
                "adapter_path": str(adapter_path) if adapter_path else None,
                "zero_shot": args.zero_shot,
                "num_samples": len(results),
            },
            "scores": scores
        }

        # Save results
        output_path = Path(args.output) if args.output else OUTPUTS_DIR / f"evaluation_results_{timestamp}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        # Print summary
        print_results_summary(scores)

        print(f"\nResults saved to: {output_path}")
        return 0

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nScoring failed: {e}")
        if predictions_path:
            print(f"Predictions are safe at: {predictions_path}")
            print(f"Re-run scoring with:")
            print(f"  python -m experiments.evaluation.evaluate_vlm --score-only --predictions {predictions_path}")
        return 1


if __name__ == "__main__":
    exit(main())
