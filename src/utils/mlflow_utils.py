"""
Shared MLflow Utilities for FADA Training Pipeline

Provides common MLflow functionality for experiment tracking across
all training scripts (classification, VLM fine-tuning, evaluation, export).
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
from mlflow.entities import Run

# Try to import TrainerCallback for VLM training
try:
    from transformers import TrainerCallback
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    TrainerCallback = object  # Placeholder


# Default experiment configurations
EXPERIMENT_CONFIGS = {
    "fada_classification": {
        "description": "Classification model experiments for fetal ultrasound organ detection",
        "tags": {"project": "fada", "task": "classification"},
    },
    "vlm_finetuning": {
        "description": "Vision-Language Model fine-tuning experiments",
        "tags": {"project": "fada", "task": "vlm_finetuning"},
    },
    "unsloth_vlm_ultrasound": {
        "description": "Unsloth VLM training for ultrasound analysis",
        "tags": {"project": "fada", "task": "vlm_training"},
    },
    "mobile_export": {
        "description": "Mobile model export and quantization tracking",
        "tags": {"project": "fada", "task": "export"},
    },
    "mobile_vlm_benchmark": {
        "description": "Edge VLM benchmarking for mobile deployment",
        "tags": {"project": "fada", "task": "benchmark"},
    },
}


def setup_mlflow_experiment(
    experiment_name: str,
    tags: Optional[Dict[str, str]] = None,
    tracking_uri: Optional[str] = None,
) -> str:
    """
    Set up an MLflow experiment with standard configuration.

    Args:
        experiment_name: Name of the experiment
        tags: Additional tags to add to the experiment
        tracking_uri: Optional MLflow tracking URI (defaults to local)

    Returns:
        Experiment ID
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Get default config if available
    default_config = EXPERIMENT_CONFIGS.get(experiment_name, {})
    default_tags = default_config.get("tags", {})

    # Merge tags
    all_tags = {**default_tags, **(tags or {})}

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags=all_tags,
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    return experiment_id


def log_training_config(config: Dict[str, Any], prefix: str = "") -> None:
    """
    Log training configuration as MLflow parameters.

    Handles nested dicts by flattening with dots.

    Args:
        config: Configuration dictionary
        prefix: Optional prefix for parameter names
    """
    def flatten_dict(d: Dict, parent_key: str = "") -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            elif isinstance(v, (list, tuple)):
                # Convert lists to strings for logging
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    flat_config = flatten_dict(config, prefix)

    # MLflow has a limit on parameter value length
    for key, value in flat_config.items():
        str_value = str(value)
        if len(str_value) > 500:
            str_value = str_value[:497] + "..."
        mlflow.log_param(key, str_value)


def log_model_architecture(
    model: Any,
    model_name: Optional[str] = None,
    log_summary: bool = True,
) -> Dict[str, Any]:
    """
    Log model architecture information to MLflow.

    Args:
        model: PyTorch model (nn.Module or similar)
        model_name: Optional name for the model
        log_summary: Whether to log model summary as artifact

    Returns:
        Dict with model statistics
    """
    stats = {}

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stats["total_parameters"] = total_params
    stats["trainable_parameters"] = trainable_params
    stats["frozen_parameters"] = total_params - trainable_params
    stats["trainable_ratio"] = trainable_params / total_params if total_params > 0 else 0

    # Log to MLflow
    mlflow.log_params({
        "model_name": model_name or model.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": f"{stats['trainable_ratio']:.2%}",
    })

    # Log model summary as artifact if requested
    if log_summary:
        summary_lines = [
            f"Model: {model_name or model.__class__.__name__}",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Frozen parameters: {total_params - trainable_params:,}",
            f"Trainable ratio: {stats['trainable_ratio']:.2%}",
            "",
            "Architecture:",
            str(model),
        ]
        summary_path = Path("model_summary.txt")
        summary_path.write_text("\n".join(summary_lines))
        mlflow.log_artifact(str(summary_path))
        summary_path.unlink()  # Clean up

    return stats


def log_evaluation_results(
    metrics: Dict[str, Any],
    artifacts: Optional[Dict[str, Union[str, Path]]] = None,
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Log evaluation results to MLflow.

    Args:
        metrics: Dictionary of metric names to values
        artifacts: Dictionary of artifact names to file paths
        step: Optional step number for metrics
        prefix: Optional prefix for metric names
    """
    # Log metrics
    logged_metrics = {}
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}" if prefix else key

        if isinstance(value, (int, float)):
            logged_metrics[metric_name] = value
        elif isinstance(value, dict):
            # Flatten nested metrics
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    logged_metrics[f"{metric_name}.{sub_key}"] = sub_value

    if logged_metrics:
        if step is not None:
            mlflow.log_metrics(logged_metrics, step=step)
        else:
            mlflow.log_metrics(logged_metrics)

    # Log artifacts
    if artifacts:
        for name, path in artifacts.items():
            path = Path(path)
            if path.exists():
                mlflow.log_artifact(str(path))
            else:
                # Create artifact from data
                if isinstance(name, str) and name.endswith(".json"):
                    temp_path = Path(name)
                    temp_path.write_text(json.dumps(path, indent=2))
                    mlflow.log_artifact(str(temp_path))
                    temp_path.unlink()


def log_confusion_matrix(
    cm: Any,  # numpy array or list
    class_names: List[str],
    filename: str = "confusion_matrix.json",
) -> None:
    """
    Log confusion matrix to MLflow as artifact.

    Args:
        cm: Confusion matrix (numpy array or list of lists)
        class_names: List of class names
        filename: Output filename
    """
    # Convert to list if numpy array
    if hasattr(cm, "tolist"):
        cm = cm.tolist()

    data = {
        "confusion_matrix": cm,
        "class_names": class_names,
    }

    path = Path(filename)
    path.write_text(json.dumps(data, indent=2))
    mlflow.log_artifact(str(path))
    path.unlink()


def log_training_curves(
    history: Dict[str, List[float]],
    filename: str = "training_curves.json",
) -> None:
    """
    Log training history curves to MLflow.

    Args:
        history: Dictionary mapping metric names to lists of values
        filename: Output filename
    """
    path = Path(filename)
    path.write_text(json.dumps(history, indent=2))
    mlflow.log_artifact(str(path))
    path.unlink()


def get_best_run(
    experiment_name: str,
    metric: str = "val_accuracy",
    ascending: bool = False,
) -> Optional[Run]:
    """
    Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to sort by
        ascending: If True, lower is better

    Returns:
        Best Run object or None if no runs found
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    order = "ASC" if ascending else "DESC"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric} IS NOT NULL",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if runs.empty:
        return None

    run_id = runs.iloc[0]["run_id"]
    return mlflow.get_run(run_id)


def create_run_name(
    model_name: str,
    task: str = "",
    suffix: str = "",
) -> str:
    """
    Create a standardized run name.

    Args:
        model_name: Name of the model
        task: Task being performed
        suffix: Optional suffix

    Returns:
        Formatted run name
    """
    parts = [model_name]
    if task:
        parts.append(task)
    if suffix:
        parts.append(suffix)

    return "_".join(parts)


class MLflowTrainerCallback(TrainerCallback):
    """
    Callback to log training metrics to MLflow for HuggingFace Trainers.

    This callback logs:
    - Per-step metrics (loss, learning rate, etc.)
    - Final training metrics
    - Evaluation metrics (if eval is performed)
    """

    def __init__(self, log_every_n_steps: int = 1):
        """
        Initialize callback.

        Args:
            log_every_n_steps: Log metrics every N steps (default: every step)
        """
        self.log_every_n_steps = log_every_n_steps
        self.training_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.training_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics at each logging step."""
        if logs is None:
            return

        # Only log at specified intervals
        if state.global_step % self.log_every_n_steps != 0:
            return

        # Filter out non-numeric values
        metrics = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                # Rename some metrics for clarity
                if k == "loss":
                    metrics["train_loss"] = v
                elif k == "eval_loss":
                    metrics["val_loss"] = v
                else:
                    metrics[k] = v

        if metrics:
            mlflow.log_metrics(metrics, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics."""
        if metrics is None:
            return

        eval_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                # Ensure eval_ prefix
                if not k.startswith("eval_"):
                    k = f"eval_{k}"
                eval_metrics[k] = v

        if eval_metrics:
            mlflow.log_metrics(eval_metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Log final metrics when training completes."""
        # Log training time
        if self.training_start_time:
            training_time = time.time() - self.training_start_time
            mlflow.log_metric("training_time_seconds", training_time)

        # Log final loss from history
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    mlflow.log_metric("final_loss", entry["loss"])
                    break

            # Log best validation loss if available
            val_losses = [
                entry.get("eval_loss")
                for entry in state.log_history
                if "eval_loss" in entry
            ]
            if val_losses:
                mlflow.log_metric("best_val_loss", min(val_losses))


class MLflowModelCheckpoint:
    """
    Save model checkpoints and log to MLflow.

    Tracks best model based on a monitored metric.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        save_dir: Optional[Union[str, Path]] = None,
        log_to_mlflow: bool = True,
    ):
        """
        Initialize checkpoint handler.

        Args:
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_dir: Directory to save checkpoints
            log_to_mlflow: Whether to log checkpoints as MLflow artifacts
        """
        self.monitor = monitor
        self.mode = mode
        self.save_dir = Path(save_dir) if save_dir else Path("checkpoints")
        self.log_to_mlflow = log_to_mlflow

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0

    def is_better(self, value: float) -> bool:
        """Check if value is better than current best."""
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def update(
        self,
        epoch: int,
        value: float,
        model: Any,
        save_fn: Optional[callable] = None,
    ) -> bool:
        """
        Update checkpoint if value is better.

        Args:
            epoch: Current epoch
            value: Current metric value
            model: Model to save
            save_fn: Optional custom save function

        Returns:
            True if checkpoint was saved
        """
        import torch

        if not self.is_better(value):
            return False

        self.best_value = value
        self.best_epoch = epoch

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.save_dir / "best_model.pth"

        # Save checkpoint
        if save_fn:
            save_fn(checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)

        # Log to MLflow
        if self.log_to_mlflow:
            mlflow.log_artifact(str(checkpoint_path))
            mlflow.log_metrics({
                f"best_{self.monitor}": value,
                "best_epoch": epoch,
            })

        return True


def log_gpu_metrics() -> Dict[str, Any]:
    """
    Log GPU utilization metrics.

    Returns:
        Dict with GPU metrics
    """
    import torch

    metrics = {}

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        metrics["gpu_name"] = torch.cuda.get_device_name(device)
        metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
        metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
        metrics["gpu_memory_total_gb"] = torch.cuda.get_device_properties(device).total_memory / 1e9

        mlflow.log_metrics({
            "gpu_memory_allocated_gb": metrics["gpu_memory_allocated_gb"],
            "gpu_memory_reserved_gb": metrics["gpu_memory_reserved_gb"],
        })

    return metrics


def log_environment():
    """Log environment info to MLflow for reproducibility."""
    import platform
    import subprocess as _subprocess

    try:
        git_hash = _subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=_subprocess.DEVNULL
        ).strip().decode()
    except Exception:
        git_hash = "unknown"

    mlflow.set_tag("mlflow.source.git.commit", git_hash)
    mlflow.log_param("python_version", platform.python_version())
    mlflow.log_param("platform", platform.platform())
    try:
        import torch
        mlflow.log_param("torch_version", torch.__version__)
        mlflow.log_param("cuda_version", torch.version.cuda or "none")
    except ImportError:
        pass
    try:
        freeze = _subprocess.check_output(
            ["pip", "freeze"], stderr=_subprocess.DEVNULL
        ).decode()
        mlflow.log_text(freeze, "pip_freeze.txt")
    except Exception:
        pass


def log_inference_metrics(
    latency_ms: float,
    model_size_mb: float,
    accuracy: Optional[float] = None,
    throughput: Optional[float] = None,
    step: Optional[int] = None,
) -> None:
    """
    Log inference performance metrics.

    Args:
        latency_ms: Inference latency in milliseconds
        model_size_mb: Model size in megabytes
        accuracy: Optional accuracy score
        throughput: Optional throughput (samples/sec)
        step: Optional step number
    """
    metrics = {
        "inference_latency_ms": latency_ms,
        "model_size_mb": model_size_mb,
    }

    if accuracy is not None:
        metrics["accuracy"] = accuracy
    if throughput is not None:
        metrics["throughput_samples_per_sec"] = throughput

    if step is not None:
        mlflow.log_metrics(metrics, step=step)
    else:
        mlflow.log_metrics(metrics)
