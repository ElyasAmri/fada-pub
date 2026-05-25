"""
Export FADA PyTorch model to TensorFlow Lite format for Android deployment.

Pipeline: PyTorch -> ONNX -> TensorFlow -> TFLite (with float16 quantization)

Usage:
    python app/android/export_model_tflite.py [--checkpoint PATH] [--output PATH]
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.classifier import FetalUltrasoundClassifier12

# Direct import of mlflow_utils to avoid __init__.py dependencies
import importlib.util
mlflow_utils_path = PROJECT_ROOT / "src" / "utils" / "mlflow_utils.py"
spec = importlib.util.spec_from_file_location("mlflow_utils", mlflow_utils_path)
mlflow_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlflow_utils)
setup_mlflow_experiment = mlflow_utils.setup_mlflow_experiment


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 14
) -> tuple[str, float]:
    """Export PyTorch model to ONNX format."""
    print(f"Exporting to ONNX: {output_path}")

    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    onnx_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"ONNX model saved: {output_path}")
    print(f"ONNX model size: {onnx_size_mb:.2f} MB")
    return output_path, onnx_size_mb


def convert_onnx_to_tflite(
    onnx_path: str,
    tflite_path: str,
    quantize: str = "float16"
) -> tuple[str, float]:
    """Convert ONNX model to TFLite with optional quantization."""
    try:
        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install onnx onnx-tf tensorflow")
        sys.exit(1)

    print(f"Converting ONNX to TFLite: {onnx_path} -> {tflite_path}")

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated successfully")

    # Convert to TensorFlow SavedModel
    tf_rep = prepare(onnx_model)
    saved_model_dir = tflite_path.replace('.tflite', '_saved_model')
    tf_rep.export_graph(saved_model_dir)
    print(f"TensorFlow SavedModel saved: {saved_model_dir}")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Apply quantization
    if quantize == "float16":
        print("Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        print("Applying int8 quantization (requires representative dataset)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Note: Full int8 requires representative_dataset_gen
    elif quantize == "none":
        print("No quantization applied (float32)")
    else:
        print(f"Unknown quantization: {quantize}, using float16")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Save TFLite model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    tflite_size_mb = os.path.getsize(tflite_path) / 1024 / 1024
    print(f"TFLite model saved: {tflite_path}")
    print(f"TFLite model size: {tflite_size_mb:.2f} MB")

    return tflite_path, tflite_size_mb


def validate_tflite_model(
    tflite_path: str,
    pytorch_model: torch.nn.Module,
    test_input: np.ndarray = None
) -> tuple[bool, float, float]:
    """Validate TFLite model output matches PyTorch model."""
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed, skipping validation")
        return True, 0.0, 0.0

    print("Validating TFLite model against PyTorch...")

    # Create test input if not provided
    if test_input is None:
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_input = torch.from_numpy(test_input)
        pytorch_output = pytorch_model(pytorch_input).numpy()

    # TFLite inference
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # TFLite expects NHWC, but our model uses NCHW
    # The ONNX->TF conversion should handle this, but let's check
    input_shape = input_details[0]['shape']
    print(f"TFLite input shape: {input_shape}")
    print(f"TFLite input dtype: {input_details[0]['dtype']}")

    # If TFLite expects NHWC, transpose
    if input_shape[-1] == 3:  # NHWC format
        tflite_input = np.transpose(test_input, (0, 2, 3, 1))
    else:  # NCHW format
        tflite_input = test_input

    interpreter.set_tensor(input_details[0]['index'], tflite_input.astype(np.float32))
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - tflite_output))
    mean_diff = np.mean(np.abs(pytorch_output - tflite_output))

    print(f"Max output difference: {max_diff:.6f}")
    print(f"Mean output difference: {mean_diff:.6f}")

    # Check if outputs match (with tolerance for quantization)
    tolerance = 0.1  # Allow some difference due to float16 quantization
    if max_diff < tolerance:
        print("Validation PASSED: TFLite output matches PyTorch")
        return True, max_diff, mean_diff
    else:
        print("Validation WARNING: Outputs differ significantly")
        print(f"PyTorch top-3: {np.argsort(pytorch_output[0])[-3:][::-1]}")
        print(f"TFLite top-3: {np.argsort(tflite_output[0])[-3:][::-1]}")
        return False, max_diff, mean_diff


def main():
    parser = argparse.ArgumentParser(description="Export FADA model to TFLite")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/best_model_efficientnet_b0_12class.pth",
        help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="app/android/app/src/main/assets/fada_classifier.tflite",
        help="Output TFLite model path"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "float16", "int8"],
        default="float16",
        help="Quantization type"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip model validation"
    )
    args = parser.parse_args()

    # Setup MLflow experiment
    setup_mlflow_experiment("mobile_export")

    # Resolve paths
    project_root = Path(__file__).parent.parent
    checkpoint_path = project_root / args.checkpoint
    output_path = project_root / args.output

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    print(f"Loading PyTorch model from: {checkpoint_path}")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Create model architecture
    model = FetalUltrasoundClassifier12(
        num_classes=12,
        backbone='efficientnet_b0',
        pretrained=False,  # We'll load weights from checkpoint
        dropout_rate=0.2
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("PyTorch model loaded successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create MLflow run
    run_name = f"efficientnet_b0_{args.quantize}"
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("source_checkpoint", str(checkpoint_path))
        mlflow.log_param("quantization_type", args.quantize)
        mlflow.log_param("input_shape", "1x3x224x224")
        mlflow.log_param("opset_version", 14)
        mlflow.log_metric("model_parameters", total_params)

        # Export to ONNX
        onnx_path = str(output_path).replace('.tflite', '.onnx')
        _, onnx_size_mb = export_to_onnx(model, onnx_path)
        mlflow.log_metric("onnx_model_size_mb", onnx_size_mb)

        # Convert to TFLite
        _, tflite_size_mb = convert_onnx_to_tflite(onnx_path, str(output_path), quantize=args.quantize)
        mlflow.log_metric("tflite_model_size_mb", tflite_size_mb)

        # Validate
        validation_passed = True
        max_diff = 0.0
        mean_diff = 0.0
        if not args.skip_validation:
            validation_passed, max_diff, mean_diff = validate_tflite_model(str(output_path), model)
            mlflow.log_metric("max_output_diff", max_diff)
            mlflow.log_metric("mean_output_diff", mean_diff)
            mlflow.log_metric("validation_passed", 1 if validation_passed else 0)

        # Log artifact path (not the file itself - too large)
        mlflow.log_param("tflite_output_path", str(output_path))

        print("\n" + "="*60)
        print("Export complete!")
        print(f"TFLite model: {output_path}")
        print(f"Model size: {tflite_size_mb:.2f} MB")
        print("="*60)

    # Cleanup ONNX file (optional)
    # os.remove(onnx_path)


if __name__ == "__main__":
    main()
