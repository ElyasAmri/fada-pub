"""
MLflow Callback for Hugging Face Trainers

Custom TrainerCallback that logs training metrics to MLflow
at each logging step for real-time experiment tracking.
"""

from transformers import TrainerCallback
import mlflow


class MLflowCallback(TrainerCallback):
    """Callback to log training metrics to MLflow."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics at each logging step."""
        if logs:
            # Filter out non-numeric values and log metrics
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if metrics:
                mlflow.log_metrics(metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Log final metrics when training completes."""
        if state.log_history:
            # Get the last logged loss
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    mlflow.log_metric("final_loss", entry["loss"])
                    break
