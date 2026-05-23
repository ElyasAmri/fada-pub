#!/usr/bin/env bash
# Evaluate a fine-tuned LoRA adapter using HF+PEFT direct inference.
#
# Preserves ALL LoRA weights including visual encoder (unlike vLLM).
# Produces checkpoint + predictions files. Scoring done separately.
#
# Usage:
#   bash eval_adapter.sh <base_model> <adapter_path> <output_dir>

set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

BASE_MODEL="${1:?Usage: eval_adapter.sh <base_model> <adapter_path> <output_dir>}"
ADAPTER_PATH="${2:?Usage: eval_adapter.sh <base_model> <adapter_path> <output_dir>}"
OUTPUT_DIR="${3:?Usage: eval_adapter.sh <base_model> <adapter_path> <output_dir>}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PREDICTIONS_FILE="${OUTPUT_DIR}/predictions.jsonl"

echo "=== Evaluating adapter (HF+PEFT) ==="
echo "Base model:   ${BASE_MODEL}"
echo "Adapter:      ${ADAPTER_PATH}"
echo "Output:       ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

# Step 1: HF+PEFT inference
echo "Running HF+PEFT inference..."
python3 "${PROJECT_DIR}/experiments/framework_comparison/eval_hf_peft.py" \
    --model "${BASE_MODEL}" \
    --adapter "${ADAPTER_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --data-root "${PROJECT_DIR}/data/Fetal Ultrasound" \
    --splits "${PROJECT_DIR}/data/dataset_splits.json" \
    --checkpoint-interval 50

# Step 2: Find checkpoint and convert to predictions
FOUND_CHECKPOINT=""
for f in "${OUTPUT_DIR}"/checkpoint_hf-peft_*.json "${OUTPUT_DIR}"/checkpoint_*.json; do
    if [ -f "$f" ]; then
        FOUND_CHECKPOINT="$f"
        break
    fi
done

if [ -z "${FOUND_CHECKPOINT}" ]; then
    echo "FATAL: No checkpoint file found in ${OUTPUT_DIR}"
    exit 1
fi

echo "Converting checkpoint: ${FOUND_CHECKPOINT}"
cd "${PROJECT_DIR}"
python3 -m experiments.evaluation.checkpoint_to_predictions \
    --checkpoint "${FOUND_CHECKPOINT}" \
    --output "${PREDICTIONS_FILE}"

echo "=== Inference complete ==="
echo "Predictions: ${PREDICTIONS_FILE}"
echo "Pull and score locally with: score_against_gt.py"
