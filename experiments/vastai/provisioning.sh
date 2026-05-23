#!/bin/bash
# FADA Training Environment Provisioning Script
#
# This script validates and sets up the environment for VLM training/inference.
# Can be used as vast.ai PROVISIONING_SCRIPT or run manually after instance creation.
#
# Usage:
#   - As PROVISIONING_SCRIPT: Set in vast.ai template settings
#   - Manual: ssh to instance and run: bash /workspace/provisioning.sh
#
# Exit codes:
#   0 - Success
#   1 - GPU validation failed
#   2 - CUDA validation failed
#   3 - Package installation failed
#   4 - Final validation failed

set -e  # Exit on any error

echo "========================================"
echo "FADA Environment Provisioning"
echo "========================================"
echo "Started at: $(date)"
echo ""

# Configuration
CUDA_VERSION_MIN="12.1"
DRIVER_VERSION_MIN=535
PYTORCH_VERSION="2.5.1"
TRANSFORMERS_VERSION="4.47.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_info() { echo "[INFO] $1"; }

# ========================================
# Step 1: GPU Validation
# ========================================
echo ""
echo "Step 1: GPU Validation"
echo "----------------------------------------"

if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found - no GPU driver installed"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null)
if [ -z "$GPU_INFO" ]; then
    log_error "Could not query GPU information"
    exit 1
fi

GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
DRIVER_VERSION=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
GPU_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)

log_ok "GPU: $GPU_NAME"
log_ok "Driver: $DRIVER_VERSION"
log_ok "Memory: $GPU_MEMORY"

# Check driver version
DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d'.' -f1)
if [ "$DRIVER_MAJOR" -lt "$DRIVER_VERSION_MIN" ]; then
    log_error "Driver version $DRIVER_VERSION is below minimum $DRIVER_VERSION_MIN"
    exit 1
fi
log_ok "Driver version meets minimum requirement ($DRIVER_VERSION_MIN)"

# ========================================
# Step 2: CUDA Validation
# ========================================
echo ""
echo "Step 2: CUDA Validation"
echo "----------------------------------------"

# Check nvcc if available
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    log_ok "NVCC version: $NVCC_VERSION"
else
    log_warn "nvcc not found (may be OK if using PyTorch with bundled CUDA)"
fi

# ========================================
# Step 3: Python Environment Setup
# ========================================
echo ""
echo "Step 3: Python Environment Setup"
echo "----------------------------------------"

# Upgrade pip
log_info "Upgrading pip..."
pip install --quiet --upgrade pip || { log_error "Failed to upgrade pip"; exit 3; }

# Install PyTorch with CUDA 12.4
log_info "Installing PyTorch ${PYTORCH_VERSION} with CUDA 12.4..."
pip install --quiet \
    torch==${PYTORCH_VERSION} \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    || { log_error "Failed to install PyTorch"; exit 3; }

# Install ML dependencies with pinned versions
log_info "Installing ML dependencies..."
pip install --quiet \
    transformers==${TRANSFORMERS_VERSION} \
    accelerate==1.2.1 \
    bitsandbytes==0.45.0 \
    peft==0.14.0 \
    || { log_error "Failed to install ML dependencies"; exit 3; }

# Install additional utilities
log_info "Installing utilities..."
pip install --quiet \
    pillow>=10.0.0 \
    tqdm \
    qwen-vl-utils \
    || { log_error "Failed to install utilities"; exit 3; }

log_ok "All packages installed"

# ========================================
# Step 4: Final Validation
# ========================================
echo ""
echo "Step 4: Final Validation"
echo "----------------------------------------"

# Validate PyTorch CUDA
log_info "Validating PyTorch CUDA..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Device: {torch.cuda.get_device_name(0)}')
" || { log_error "PyTorch CUDA validation failed"; exit 4; }

# Validate key imports
log_info "Validating package imports..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig
import bitsandbytes
import accelerate
print('All imports successful')
" || { log_error "Package import validation failed"; exit 4; }

# Quick GPU memory test
log_info "Testing GPU memory allocation..."
python3 -c "
import torch
# Allocate 1GB tensor to verify GPU works
x = torch.randn(256, 1024, 1024, device='cuda')
del x
torch.cuda.empty_cache()
print('GPU memory allocation OK')
" || { log_error "GPU memory test failed"; exit 4; }

# ========================================
# Complete
# ========================================
echo ""
echo "========================================"
echo -e "${GREEN}Environment Ready!${NC}"
echo "========================================"
echo "Completed at: $(date)"
echo ""
echo "Summary:"
echo "  GPU: $GPU_NAME ($GPU_MEMORY)"
echo "  Driver: $DRIVER_VERSION"
echo "  PyTorch: ${PYTORCH_VERSION}"
echo "  Transformers: ${TRANSFORMERS_VERSION}"
echo ""
