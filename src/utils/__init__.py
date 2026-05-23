# FADA Utils Module
# Lazy: heavy submodules (training, mlflow_utils) require torch/mlflow.
# Import them directly: e.g. from src.utils.mlflow_utils import setup_mlflow_experiment

# Lightweight utilities safe to import without torch
from .image_processing import to_jpeg_bytes, to_base64, to_base64_data_url, load_and_convert
from .api_client import call_with_retry, retry_decorator, exponential_backoff, APIError
from .results_manager import ResultsManager
from .logging_config import setup_logging, get_logger
