"""
Evaluate Fine-tuned Qwen3-VL on Ultrasound Normality Assessment

This script evaluates the fine-tuned model on the validation set,
comparing generated responses to ground truth annotations.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import mlflow
import torch
from unsloth import FastVisionModel
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from prepare_dataset import prepare_dataset, Q7_PROMPT


# Model paths
BASE_MODEL = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
ADAPTER_DIR = Path(__file__).parent / "outputs" / "qwen3vl_ultrasound" / "lora_adapters"
RESULTS_DIR = Path(__file__).parent / "outputs" / "evaluation"


def normalize_assessment(text: str) -> str:
    """
    Normalize Q7 assessment to a category for comparison.

    Categories:
    - normal: All variations of normal findings
    - abnormal: Any abnormality mentioned
    - unclear: Cannot determine
    """
    text = text.lower().strip()

    # Keywords indicating abnormality
    abnormal_keywords = [
        "abnormal", "anomaly", "anomalies", "concern", "suspicious",
        "thickening", "thickened", "increased", "decreased", "absent",
        "cystic", "irregular", "malformation", "defect", "lesion",
        "mass", "fluid", "dilated", "stenosis", "atresia"
    ]

    # Keywords indicating normal
    normal_keywords = [
        "normal", "within normal", "no abnormal", "unremarkable",
        "appropriate", "adequate", "good", "well", "healthy",
        "no evidence", "no abnormality", "no anomaly"
    ]

    # Check for abnormal indicators first (they override normal)
    for keyword in abnormal_keywords:
        if keyword in text:
            # Check for negation patterns before the keyword (no trailing \b to handle plural/suffix forms)
            negation_pattern = rf'\bno\s+{re.escape(keyword)}|\bnot\s+{re.escape(keyword)}|\bwithout\s+{re.escape(keyword)}|\bno\s+evidence\s+of\s+{re.escape(keyword)}'
            if not re.search(negation_pattern, text):
                return "abnormal"

    # Check for normal indicators
    for keyword in normal_keywords:
        if keyword in text:
            return "normal"

    return "unclear"


def load_model_with_adapters(adapter_dir: Optional[Path] = None):
    """Load base model with optional LoRA adapters."""
    print(f"Loading base model: {BASE_MODEL}")

    model, tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
    )

    if adapter_dir and adapter_dir.exists():
        print(f"Loading LoRA adapters from: {adapter_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        print("Adapters loaded successfully!")
    else:
        print("Using base model without fine-tuning")

    # Set to inference mode
    FastVisionModel.for_inference(model)

    return model, tokenizer


def generate_response(model, tokenizer, image, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate model response for an image and prompt."""
    # Create messages in the expected format
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
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize with image
    inputs = tokenizer(
        input_text,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
        )

    # Decode response (skip input tokens)
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


def evaluate_model(
    model,
    tokenizer,
    val_dataset,
    max_samples: Optional[int] = None
) -> dict:
    """
    Evaluate model on validation dataset.

    Returns dict with:
    - predictions: List of (reference_response, prediction, normalized_gt, normalized_pred)
    - metrics: Classification metrics
    """
    predictions = []
    samples_to_eval = val_dataset
    if max_samples:
        samples_to_eval = val_dataset.select(range(min(max_samples, len(val_dataset))))

    print(f"Evaluating on {len(samples_to_eval)} samples...")

    for sample in tqdm(samples_to_eval):
        # Extract image and ground truth from sample
        user_content = sample["messages"][0]["content"]
        image = user_content[0]["image"]  # PIL Image
        reference_response = sample["messages"][1]["content"][0]["text"]

        # Generate prediction
        prediction = generate_response(model, tokenizer, image, Q7_PROMPT)

        # Normalize for classification
        gt_normalized = normalize_assessment(reference_response)
        pred_normalized = normalize_assessment(prediction)

        predictions.append({
            "reference_response": reference_response,
            "prediction": prediction,
            "gt_category": gt_normalized,
            "pred_category": pred_normalized,
        })

    # Calculate metrics
    gt_categories = [p["gt_category"] for p in predictions]
    pred_categories = [p["pred_category"] for p in predictions]

    # Classification report
    labels = ["normal", "abnormal", "unclear"]
    report = classification_report(
        gt_categories, pred_categories,
        labels=labels, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(gt_categories, pred_categories, labels=labels)

    # Calculate agreement
    agreement = sum(1 for p in predictions if p["gt_category"] == p["pred_category"])
    agreement_rate = agreement / len(predictions)

    results = {
        "predictions": predictions,
        "metrics": {
            "agreement_rate": agreement_rate,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        },
        "summary": {
            "total_samples": len(predictions),
            "agreement": agreement,
            "agreement_rate": f"{agreement_rate:.2%}",
        }
    }

    return results


def print_results(results: dict):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    summary = results["summary"]
    print(f"\nTotal samples: {summary['total_samples']}")
    print(f"Agreement rate: {summary['agreement_rate']}")

    print("\nClassification Report:")
    report = results["metrics"]["classification_report"]
    print(f"  {'Category':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 44)
    for category in ["normal", "abnormal", "unclear"]:
        if category in report:
            r = report[category]
            print(f"  {category:<12} {r['precision']:>10.2f} {r['recall']:>10.2f} {r['f1-score']:>10.2f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("              normal  abnormal  unclear")
    cm = results["metrics"]["confusion_matrix"]
    for i, row in enumerate(cm):
        label = ["normal", "abnormal", "unclear"][i]
        print(f"  {label:<10} {row[0]:>6} {row[1]:>9} {row[2]:>8}")

    # Show some examples
    print("\nSample Predictions:")
    print("-" * 60)
    for i, pred in enumerate(results["predictions"][:5]):
        print(f"\n[{i+1}] Reference: {pred['reference_response'][:80]}...")
        print(f"    Prediction:   {pred['prediction'][:80]}...")
        print(f"    Categories:   GT={pred['gt_category']}, Pred={pred['pred_category']}")


def save_results(results: dict, output_path: Path):
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert predictions for JSON (remove PIL images)
    results_copy = results.copy()
    for pred in results_copy["predictions"]:
        # Remove any non-serializable objects
        pass

    with open(output_path, "w") as f:
        json.dump(results_copy, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen3-VL")
    parser.add_argument("--adapter-dir", type=str, default=str(ADAPTER_DIR),
                        help="Path to LoRA adapters (empty for base model)")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum samples to evaluate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--compare-base", action="store_true",
                        help="Also evaluate base model for comparison")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Parent MLflow run ID to log as nested run")
    parser.add_argument("--experiment", type=str, default="unsloth_vlm_ultrasound",
                        help="MLflow experiment name")

    args = parser.parse_args()

    # Prepare dataset
    print("Loading validation dataset...")
    _, val_dataset = prepare_dataset(train_ratio=0.9, seed=42)
    print(f"Validation set: {len(val_dataset)} samples")

    # Evaluate fine-tuned model
    adapter_path = Path(args.adapter_dir) if args.adapter_dir else None
    model, tokenizer = load_model_with_adapters(adapter_path)

    print("\nEvaluating fine-tuned model...")
    results = evaluate_model(model, tokenizer, val_dataset, args.max_samples)
    print_results(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        save_results(results, output_path)
    else:
        output_path = RESULTS_DIR / "evaluation_results.json"
        save_results(results, output_path)

    # Log to MLflow
    mlflow.set_experiment(args.experiment)

    if args.run_id:
        # Log as nested run under parent training run
        with mlflow.start_run(run_id=args.run_id):
            with mlflow.start_run(run_name="evaluation", nested=True):
                # Log parameters
                mlflow.log_param("adapter_dir", str(adapter_path))
                mlflow.log_param("max_samples", args.max_samples)
                mlflow.log_param("compare_base", args.compare_base)

                # Log metrics
                metrics = results["metrics"]
                mlflow.log_metric("agreement_rate", metrics["agreement_rate"])

                # Log per-class metrics
                report = metrics["classification_report"]
                for category in ["normal", "abnormal", "unclear"]:
                    if category in report:
                        mlflow.log_metric(f"{category}_precision", report[category]["precision"])
                        mlflow.log_metric(f"{category}_recall", report[category]["recall"])
                        mlflow.log_metric(f"{category}_f1", report[category]["f1-score"])

                # Log artifact
                mlflow.log_artifact(str(output_path))
    else:
        # Create standalone evaluation run
        with mlflow.start_run(run_name="evaluation"):
            # Log parameters
            mlflow.log_param("adapter_dir", str(adapter_path))
            mlflow.log_param("max_samples", args.max_samples)
            mlflow.log_param("compare_base", args.compare_base)

            # Log metrics
            metrics = results["metrics"]
            mlflow.log_metric("agreement_rate", metrics["agreement_rate"])

            # Log per-class metrics
            report = metrics["classification_report"]
            for category in ["normal", "abnormal", "unclear"]:
                if category in report:
                    mlflow.log_metric(f"{category}_precision", report[category]["precision"])
                    mlflow.log_metric(f"{category}_recall", report[category]["recall"])
                    mlflow.log_metric(f"{category}_f1", report[category]["f1-score"])

            # Log artifact
            mlflow.log_artifact(str(output_path))

    # Optionally compare with base model
    if args.compare_base:
        print("\n" + "=" * 60)
        print("COMPARING WITH BASE MODEL")
        print("=" * 60)

        # Reload base model without adapters
        del model
        torch.cuda.empty_cache()

        model_base, tokenizer_base = load_model_with_adapters(None)

        print("\nEvaluating base model...")
        results_base = evaluate_model(model_base, tokenizer_base, val_dataset, args.max_samples)
        print_results(results_base)

        base_output_path = RESULTS_DIR / "base_model_results.json"
        save_results(results_base, base_output_path)

        # Log base model results to MLflow
        if args.run_id:
            # Log as nested run under parent training run
            with mlflow.start_run(run_id=args.run_id):
                with mlflow.start_run(run_name="base_model_evaluation", nested=True):
                    # Log parameters
                    mlflow.log_param("adapter_dir", "None")
                    mlflow.log_param("max_samples", args.max_samples)
                    mlflow.log_param("compare_base", True)

                    # Log metrics
                    metrics_base = results_base["metrics"]
                    mlflow.log_metric("agreement_rate", metrics_base["agreement_rate"])

                    # Log per-class metrics
                    report_base = metrics_base["classification_report"]
                    for category in ["normal", "abnormal", "unclear"]:
                        if category in report_base:
                            mlflow.log_metric(f"{category}_precision", report_base[category]["precision"])
                            mlflow.log_metric(f"{category}_recall", report_base[category]["recall"])
                            mlflow.log_metric(f"{category}_f1", report_base[category]["f1-score"])

                    # Log artifact
                    mlflow.log_artifact(str(base_output_path))
        else:
            # Create standalone evaluation run for base model
            with mlflow.start_run(run_name="base_model_evaluation"):
                # Log parameters
                mlflow.log_param("adapter_dir", "None")
                mlflow.log_param("max_samples", args.max_samples)
                mlflow.log_param("compare_base", True)

                # Log metrics
                metrics_base = results_base["metrics"]
                mlflow.log_metric("agreement_rate", metrics_base["agreement_rate"])

                # Log per-class metrics
                report_base = metrics_base["classification_report"]
                for category in ["normal", "abnormal", "unclear"]:
                    if category in report_base:
                        mlflow.log_metric(f"{category}_precision", report_base[category]["precision"])
                        mlflow.log_metric(f"{category}_recall", report_base[category]["recall"])
                        mlflow.log_metric(f"{category}_f1", report_base[category]["f1-score"])

                # Log artifact
                mlflow.log_artifact(str(base_output_path))


if __name__ == "__main__":
    main()
