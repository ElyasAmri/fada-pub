"""
Framework comparison orchestrator.

Reads config.json to build the run matrix (model x framework), executes training
and evaluation sequentially, with resume support.

Usage:
    python experiments/framework_comparison/run_queue.py
    python experiments/framework_comparison/run_queue.py --dry-run
    python experiments/framework_comparison/run_queue.py --filter-model qwen2.5
    python experiments/framework_comparison/run_queue.py --filter-framework unsloth
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Framework -> conda env mapping
CONDA_ENVS = {
    "unsloth": "ft-unsloth",
    "llamafactory": "ft-llamafactory",
    "swift": "ft-swift",
    "axolotl": "ft-axolotl",
}

# Framework -> wrapper script
WRAPPER_SCRIPTS = {
    "unsloth": SCRIPT_DIR / "wrappers" / "train_unsloth.py",
    "llamafactory": SCRIPT_DIR / "wrappers" / "train_llamafactory.py",
    "swift": SCRIPT_DIR / "wrappers" / "train_swift.py",
    "axolotl": SCRIPT_DIR / "wrappers" / "train_axolotl.py",
}

# Frameworks that use ShareGPT format
SHAREGPT_FRAMEWORKS = {"llamafactory", "axolotl"}


def resolve_conda_executable() -> str:
    """Resolve conda executable across RCCG/Linux and local Windows setups."""
    conda_from_env = os.environ.get("CONDA_EXE")
    if conda_from_env:
        return conda_from_env

    linux_default = "/home/ubuntu/miniconda3/bin/conda"
    if Path(linux_default).exists():
        return linux_default

    conda_on_path = shutil.which("conda")
    if conda_on_path:
        return conda_on_path

    # Keep dry-run and matrix operations usable even if conda isn't installed.
    # Actual training/eval will fail with a clear subprocess error if unresolved.
    return "conda"


CONDA = resolve_conda_executable()


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def build_run_matrix(config: dict, filter_model: str = None, filter_framework: str = None) -> list:
    """Build list of (model_id, short_name, framework) tuples to run."""
    runs = []
    for model in config["models"]:
        if filter_model and filter_model.lower() not in model["short_name"].lower():
            continue
        for fw in model["frameworks"]:
            if filter_framework and filter_framework.lower() != fw.lower():
                continue
            runs.append((model["id"], model["short_name"], fw))
    return runs


def run_dir_name(short_name: str, framework: str) -> str:
    return f"{short_name}_{framework}"


def is_complete(output_dir: Path) -> bool:
    """Check if a run has already completed successfully."""
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        return False
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest.get("status") == "complete"


def has_scores(output_dir: Path) -> bool:
    """Check if evaluation scores exist."""
    return (output_dir / "scores.json").exists()


def run_training(model_id: str, framework: str, output_dir: Path,
                 config: dict, config_path: Path, test_run: bool = False) -> bool:
    """Run training for a single model+framework pair. Returns True on success."""
    env_name = CONDA_ENVS[framework]
    wrapper = WRAPPER_SCRIPTS[framework]
    data_cfg = config["data"]

    # Choose data format
    if framework in SHAREGPT_FRAMEWORKS:
        train_data = str(PROJECT_ROOT / data_cfg["train_file_sharegpt"])
        val_data = str(PROJECT_ROOT / data_cfg["val_file_sharegpt"])
    else:
        train_data = str(PROJECT_ROOT / data_cfg["train_file"])
        val_data = str(PROJECT_ROOT / data_cfg["val_file"])

    data_root = str(PROJECT_ROOT / data_cfg["data_root"])

    cmd = [
        CONDA, "run", "-n", env_name, "--no-capture-output",
        "python", str(wrapper),
        "--model", model_id,
        "--train-data", train_data,
        "--val-data", val_data,
        "--data-root", data_root,
        "--output-dir", str(output_dir),
        "--config", str(config_path),
    ]
    if test_run:
        cmd.append("--test-run")

    # Ensure HF_HOME is set for model downloads
    env = os.environ.copy()
    if "HF_HOME" not in env:
        env["HF_HOME"] = "/mnt/models/huggingface"

    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    return result.returncode == 0


def run_eval(model_id: str, output_dir: Path) -> bool:
    """Run evaluation for a completed training run. Returns True on success."""
    adapter_path = output_dir / "adapter"
    if not adapter_path.exists():
        # Check manifest for actual adapter path
        manifest_path = output_dir / "run_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            ap = manifest.get("adapter_path")
            if ap and Path(ap).exists():
                adapter_path = Path(ap)

    if not adapter_path.exists():
        print(f"  ERROR: No adapter found at {adapter_path}")
        return False

    eval_script = SCRIPT_DIR / "eval_adapter.sh"
    # ft-eval may be a prefix env on the volume (disk space constraints)
    eval_env_prefix = os.environ.get("FT_EVAL_PREFIX", "/mnt/models/conda-envs/ft-eval")
    if os.path.isdir(eval_env_prefix):
        conda_env_args = ["-p", eval_env_prefix]
    else:
        conda_env_args = ["-n", "ft-eval"]
    cmd = [
        CONDA, "run", *conda_env_args, "--no-capture-output",
        "bash", str(eval_script),
        model_id,
        str(adapter_path),
        str(output_dir),
    ]

    # Ensure HF_HOME is set for model downloads
    env = os.environ.copy()
    if "HF_HOME" not in env:
        env["HF_HOME"] = "/mnt/models/huggingface"

    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    return result.returncode == 0


def log_run(log_path: Path, entry: dict):
    """Append a run entry to the JSONL log."""
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")



def main():
    parser = argparse.ArgumentParser(description="Framework comparison run queue")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--output-base", type=Path, default=SCRIPT_DIR / "runs",
                        help="Base directory for all run outputs")
    parser.add_argument("--filter-model", default=None, help="Filter by model short name substring")
    parser.add_argument("--filter-framework", default=None, help="Filter by framework name")
    parser.add_argument("--dry-run", action="store_true", help="Print run matrix without executing")
    parser.add_argument("--test-run", action="store_true", help="Quick test (100 samples per run)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation for completed runs")
    args = parser.parse_args()

    config = load_config(args.config)
    runs = build_run_matrix(config, args.filter_model, args.filter_framework)

    print(f"Run matrix: {len(runs)} combinations")
    print("-" * 60)
    for model_id, short_name, framework in runs:
        out_dir = args.output_base / run_dir_name(short_name, framework)
        status = "DONE" if is_complete(out_dir) else "PENDING"
        scored = " + SCORED" if has_scores(out_dir) else ""
        print(f"  [{status}{scored}] {short_name} x {framework}")
    print("-" * 60)

    if args.dry_run:
        print("Dry run -- exiting.")
        return

    args.output_base.mkdir(parents=True, exist_ok=True)
    log_path = args.output_base / "runs.jsonl"

    completed = 0
    failed = 0
    skipped = 0

    for i, (model_id, short_name, framework) in enumerate(runs, 1):
        out_dir = args.output_base / run_dir_name(short_name, framework)
        print(f"\n[{i}/{len(runs)}] {short_name} x {framework}")

        # Training phase
        if not args.eval_only:
            if is_complete(out_dir):
                print(f"  SKIP: already complete")
                skipped += 1
            else:
                print(f"  Training...")
                out_dir.mkdir(parents=True, exist_ok=True)
                success = run_training(model_id, framework, out_dir, config, args.config, args.test_run)
                if success:
                    print(f"  Training: OK")
                    completed += 1
                else:
                    print(f"  Training: FAILED")
                    failed += 1
                    log_run(log_path, {
                        "timestamp": datetime.now().isoformat(),
                        "model": model_id, "framework": framework,
                        "phase": "train", "status": "failed",
                    })
                    continue

        # Eval phase
        if not args.skip_eval:
            if has_scores(out_dir):
                print(f"  SKIP eval: scores already exist")
            elif is_complete(out_dir) or args.eval_only:
                print(f"  Evaluating...")
                eval_ok = run_eval(model_id, out_dir)
                if eval_ok:
                    print(f"  Eval: OK")
                else:
                    print(f"  Eval: FAILED")
                    log_run(log_path, {
                        "timestamp": datetime.now().isoformat(),
                        "model": model_id, "framework": framework,
                        "phase": "eval", "status": "failed",
                    })

        log_run(log_path, {
            "timestamp": datetime.now().isoformat(),
            "model": model_id, "framework": framework,
            "phase": "complete", "status": "ok",
        })

    print(f"\n{'=' * 60}")
    print(f"Completed: {completed} | Skipped: {skipped} | Failed: {failed}")
    print(f"Results in: {args.output_base}")
    print(f"Run log: {log_path}")

    # Run aggregation if any runs completed
    if completed > 0 or skipped > 0:
        print("\nAggregating results...")
        agg_cmd = [sys.executable, str(SCRIPT_DIR / "aggregate_results.py"),
                   "--runs-dir", str(args.output_base)]
        subprocess.run(agg_cmd, capture_output=False, text=True)


if __name__ == "__main__":
    main()
