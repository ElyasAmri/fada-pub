#!/bin/bash
# Launch HF+PEFT evaluation for all completed framework comparison runs.
# Uses ft-unsloth env (has torch, transformers, peft, bitsandbytes).
# Run on RCCG with: nohup bash launch_eval.sh > eval.log 2>&1 &

export HF_HOME=/mnt/models/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/fada-v3

CONDA=/home/ubuntu/miniconda3/bin/conda
EVAL_SCRIPT=experiments/framework_comparison/eval_adapter.sh
RUNS_DIR=/mnt/models/fc-runs

echo "=== Starting HF+PEFT eval for all completed models ==="
echo

for run_dir in qwen2.5-vl-7b_unsloth qwen3-vl-8b_unsloth qwen3.5-9b_unsloth; do
    OUT="$RUNS_DIR/$run_dir"
    MANIFEST="$OUT/run_manifest.json"

    if [ -f "$OUT/scores.json" ]; then
        echo "[$run_dir] SKIP: scores already exist"
        continue
    fi

    MODEL=$(grep -o '"model": "[^"]*"' "$MANIFEST" | head -1 | cut -d'"' -f4)
    ADAPTER="$OUT/adapter"

    echo "[$run_dir] Evaluating model=$MODEL adapter=$ADAPTER"
    $CONDA run -n ft-unsloth --no-capture-output bash $EVAL_SCRIPT "$MODEL" "$ADAPTER" "$OUT"

    if [ -f "$OUT/predictions.jsonl" ]; then
        echo "[$run_dir] DONE"
    else
        echo "[$run_dir] FAILED"
    fi

    echo
done

echo "=== All evals complete ==="
