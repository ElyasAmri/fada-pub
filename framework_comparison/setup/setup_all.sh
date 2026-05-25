#!/usr/bin/env bash
# Setup all conda environments for framework comparison on RCCG H100.
# Each framework gets an isolated env to avoid dependency conflicts.
# Idempotent: skips envs that already exist.
#
# Usage: bash experiments/framework_comparison/setup/setup_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Ensure conda is available
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh"
    echo "  bash /tmp/miniconda.sh -b -p \$HOME/miniconda3"
    echo "  eval \"\$(\$HOME/miniconda3/bin/conda shell.bash hook)\""
    exit 1
fi

# Source conda for script use
eval "$(conda shell.bash hook)"

setup_env() {
    local ENV_NAME="$1"
    local REQ_FILE="$2"
    local EXTRA_CMD="${3:-}"

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "=== ${ENV_NAME}: already exists, skipping ==="
        return 0
    fi

    echo "=== Creating ${ENV_NAME} ==="
    conda create -n "${ENV_NAME}" python=3.10 -y -q
    conda activate "${ENV_NAME}"

    # Install PyTorch with CUDA 12.4 first
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q

    # Install framework-specific requirements
    pip install -r "${SCRIPT_DIR}/${REQ_FILE}" -q

    # Run extra install commands if provided
    if [ -n "${EXTRA_CMD}" ]; then
        eval "${EXTRA_CMD}"
    fi

    conda deactivate
    echo "=== ${ENV_NAME}: done ==="
}

echo "Setting up framework comparison environments..."
echo "Project dir: ${PROJECT_DIR}"
echo ""

# 1. Unsloth
setup_env "ft-unsloth" "requirements_unsloth.txt"

# 2. LLaMA Factory
setup_env "ft-llamafactory" "requirements_llamafactory.txt"

# 3. ms-swift
setup_env "ft-swift" "requirements_swift.txt"

# 4. Axolotl
setup_env "ft-axolotl" "requirements_axolotl.txt"

# 5. Eval (vLLM for adapter inference + scoring)
setup_env "ft-eval" "requirements_eval.txt"

echo ""
echo "All environments created. Verify with: conda env list"
echo ""
echo "Next steps:"
echo "  1. Convert data: conda run -n ft-eval python ${PROJECT_DIR}/experiments/framework_comparison/convert_to_sharegpt.py"
echo "  2. Run queue:    conda run -n ft-eval python ${PROJECT_DIR}/experiments/framework_comparison/run_queue.py"
