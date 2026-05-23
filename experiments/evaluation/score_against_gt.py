"""
CLI entry point for scoring VLM predictions against sonographer ground truth.

Usage:
    python -m experiments.evaluation.score_against_gt \
        --predictions outputs/evaluation/predictions_fixed.jsonl \
        [--annotations data/Fetal\ Ultrasound\ Annotations\ Normalized.xlsx] \
        [--output outputs/evaluation/gt_scores.json] \
        [--device cuda]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.evaluation.config import ANNOTATIONS_PATH, OUTPUTS_DIR
from experiments.evaluation.question_scorer import MultiMetricScorer, print_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Score VLM predictions against sonographer ground truth annotations."
    )
    parser.add_argument(
        "--predictions", required=True, type=Path,
        help="Path to predictions JSONL file",
    )
    parser.add_argument(
        "--annotations", type=Path, default=ANNOTATIONS_PATH,
        help="Path to normalized annotations Excel (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for JSON results (default: outputs/evaluation/gt_scores.json)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for embedding model (default: cuda)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = OUTPUTS_DIR / "gt_scores.json"

    # Load predictions
    logger.info("Loading predictions from %s", args.predictions)
    predictions = []
    with open(args.predictions) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    logger.info("Loaded %d predictions", len(predictions))

    # Score
    scorer = MultiMetricScorer(
        annotations_path=args.annotations,
        device=args.device,
    )
    results = scorer.score_predictions(
        predictions, predictions_file=str(args.predictions)
    )

    # Print report
    print_report(results)

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
