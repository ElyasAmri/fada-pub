# FADA - Fetal Anomaly Detection Algorithm

A research prototype for fetal ultrasound image analysis using Vision Language Models (VLMs). The project benchmarks, fine-tunes, and evaluates VLMs for automated ultrasound interpretation via 8 clinical questions per image.

**IMPORTANT: This is a research prototype for educational purposes only. NOT for clinical use.**

## Current State

- **Dataset**: 19,019 images, 14 anatomical categories, 18,936 annotated
- **Models evaluated**: 46 VLMs scored zero-shot on 1,894 test images (Phase 4 complete)
- **Best zero-shot**: Qwen3.5-35B-A3B at 36.5% primary score
- **Best fine-tuned**: Qwen2.5-VL-7B at 81.1% embedding similarity (600 samples)
- **Approach**: VLM benchmarking and fine-tuning (not classification)

See `docs/` for full details, experiment results, and next steps.

## Project Structure

```
fada-v3/
+-- src/                # Source code (inference, training, data loaders)
+-- experiments/        # VLM evaluation, fine-tuning, RCCG cluster scripts
+-- data/               # Ultrasound images and annotations (not tracked)
+-- docs/               # Project documentation
+-- web/                # Streamlit prototype interface
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (RTX 4070 or better recommended)
- Windows/Linux/macOS

### Installation

1. Create virtual environment:

```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# or
venv\Scripts\activate         # Windows CMD
```

2. Install dependencies (includes PyTorch with CUDA 12.8):

```bash
pip install -r requirements.txt
```

### Key Commands

```bash
# Run API model evaluation (Gemini, GPT-4o, vLLM backends)
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py

# Fine-tune with Unsloth (local, RTX 5090)
./venv/Scripts/python.exe experiments/unsloth_vlm/train_qwen3vl.py

# Track experiments
mlflow ui --host 0.0.0.0 --port 5000

# Run Streamlit prototype (NOT for clinical use)
streamlit run web/app.py --server.port 8501
```

## Documentation

- `docs/dashboard.md` - Project overview and quick stats
- `docs/project/Tasks.md` - Current task status
- `docs/experiments/models-tracker.md` - Model evaluation status and scores
- `docs/experiments/results-summary.md` - Per-category performance
- `docs/project/next-steps.md` - Prioritized action plan

## Disclaimer

This is a research and educational project. All outputs should include:
"For educational purposes only. Not for clinical use. Consult a healthcare provider for medical advice."

## License

Research prototype - not for production use.
