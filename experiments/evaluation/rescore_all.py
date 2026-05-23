"""Batch rescore checkpoint files with the updated scoring pipeline."""
import subprocess
import glob
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "full_test"
PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
CONVERT_SCRIPT = PROJECT_ROOT / "experiments" / "evaluation" / "checkpoint_to_predictions.py"
SCORE_SCRIPT = PROJECT_ROOT / "experiments" / "evaluation" / "score_against_gt.py"

def main():
    checkpoints = sorted(glob.glob(str(RESULTS_DIR / "checkpoint_*.json")))
    print(f"Found {len(checkpoints)} checkpoints")

    for cp in checkpoints:
        basename = os.path.basename(cp)
        # Extract model name from checkpoint filename
        # checkpoint_vllm_Qwen_Qwen3.5-2B.json -> Qwen_Qwen3.5-2B
        model_name = basename.replace("checkpoint_vllm_", "").replace("checkpoint_", "").replace(".json", "")

        pred_file = RESULTS_DIR / f"predictions_{model_name}.jsonl"
        score_file = RESULTS_DIR / f"scores_{model_name}.json"

        # Convert checkpoint to predictions if needed
        if not os.path.exists(pred_file):
            print(f"Converting {basename} -> predictions_{model_name}.jsonl")
            result = subprocess.run(
                [str(PYTHON), str(CONVERT_SCRIPT), "--checkpoint", cp, "--output", str(pred_file)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"  ERROR converting: {result.stderr[:200]}")
                continue

        # Always rescore with updated pipeline
        print(f"Scoring {model_name}...")
        result = subprocess.run(
            [str(PYTHON), str(SCORE_SCRIPT), "--predictions", str(pred_file), "--output", str(score_file)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ERROR scoring: {result.stderr[:200]}")
            continue

        # Print result
        import json
        try:
            scores = json.load(open(score_file))
            overall = scores.get("overall", {})
            primary = overall.get("primary_score_mean", 0)
            embed = overall.get("embedding_similarity_mean", 0)
            print(f"  -> primary={primary:.4f}, embed_sim={embed:.4f}")
        except Exception as e:
            print(f"  ERROR reading scores: {e}")

    print("\nDone! All models rescored.")

if __name__ == "__main__":
    main()
