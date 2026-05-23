"""
Data Transfer Utilities for Vast.ai.

Handles file uploads/downloads with caching and progress tracking.
"""

import hashlib
import json
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .instance import VastInstance


# Regex pattern for extracting relative paths from "Fetal Ultrasound/..." paths
FETAL_ULTRASOUND_PATTERN = re.compile(r'Fetal Ultrasound[/\\](.+)$')

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "vastai"
CACHE_DIR = OUTPUTS_DIR / "cache"

# Remote workspace paths
REMOTE_WORKSPACE = "/workspace/fada"
REMOTE_DATA_DIR = f"{REMOTE_WORKSPACE}/data"
REMOTE_OUTPUT_DIR = f"{REMOTE_WORKSPACE}/outputs"
REMOTE_MODELS_DIR = f"{REMOTE_WORKSPACE}/models"


class DataTransfer:
    """
    Manage data transfers between local and remote instances.

    Features:
    - Caches uploaded files to avoid re-uploading
    - Tracks file hashes for change detection
    - Organizes remote workspace structure
    """

    def __init__(self, instance: VastInstance, cache_dir: Optional[Path] = None):
        """
        Initialize transfer manager.

        Args:
            instance: Connected VastInstance
            cache_dir: Directory for caching upload manifests
        """
        self.instance = instance
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _file_hash(self, path: Path) -> str:
        """Compute MD5 hash of a file."""
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_manifest_path(self, name: str) -> Path:
        """Get path to upload manifest file."""
        return self.cache_dir / f"{name}_manifest.json"

    def _load_manifest(self, name: str) -> dict:
        """Load upload manifest from cache."""
        manifest_path = self._get_manifest_path(name)
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                return json.load(f)
        return {}

    def _save_manifest(self, name: str, manifest: dict):
        """Save upload manifest to cache."""
        manifest_path = self._get_manifest_path(name)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def setup_workspace(self) -> bool:
        """Create remote workspace directory structure."""
        setup_cmds = f"""
        mkdir -p {REMOTE_DATA_DIR}
        mkdir -p {REMOTE_OUTPUT_DIR}
        mkdir -p {REMOTE_MODELS_DIR}
        mkdir -p {REMOTE_WORKSPACE}/scripts
        mkdir -p /workspace/data
        """
        ret, _, stderr = self.instance.run_ssh(setup_cmds, timeout=60)
        if ret != 0:
            print(f"Failed to create workspace: {stderr}")
            return False
        return True

    def upload_test_data(self, test_file: Path, force: bool = False) -> bool:
        """
        Upload test data file.

        Args:
            test_file: Path to test JSONL file
            force: Force upload even if cached

        Returns:
            True if upload successful
        """
        if not test_file.exists():
            print(f"Test file not found: {test_file}")
            return False

        # Check cache
        if not force:
            manifest = self._load_manifest("test_data")
            file_hash = self._file_hash(test_file)
            if manifest.get("hash") == file_hash and manifest.get("instance_id") == self.instance.instance_id:
                print(f"  Test data already uploaded (cached)")
                return True

        # Upload
        remote_path = f"{REMOTE_DATA_DIR}/test_subset.jsonl"
        print(f"  Uploading test data: {test_file.name}")
        if not self.instance.upload_file(test_file, remote_path):
            return False

        # Update cache
        manifest = {
            "hash": self._file_hash(test_file),
            "instance_id": self.instance.instance_id,
            "remote_path": remote_path,
            "local_path": str(test_file),
        }
        self._save_manifest("test_data", manifest)
        return True

    def upload_images(self, images_dir: Path, force: bool = False) -> bool:
        """
        Upload images directory using tar+ssh for speed.

        This is typically the largest upload (Fetal Ultrasound dataset).

        Args:
            images_dir: Path to images directory
            force: Force upload even if cached

        Returns:
            True if upload successful
        """
        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            return False

        # For large directories, we skip hash checking and rely on instance ID
        if not force:
            manifest = self._load_manifest("images")
            if manifest.get("instance_id") == self.instance.instance_id:
                print(f"  Images already uploaded to this instance (cached)")
                return True

        # Use tar + ssh pipe for faster upload of many files
        print(f"  Uploading images: {images_dir.name}/ (compressing and streaming...)")

        ssh_cmd = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {self.instance.ssh_port} root@{self.instance.ssh_host}"
        remote_cmd = f"cd /workspace/data && tar xzf -"

        # tar on Windows needs special handling
        if platform.system() == "Windows":
            # Use scp for Windows (tar piping is problematic)
            print(f"  (Using SCP on Windows - this may take longer...)")
            if not self.instance.upload_directory(images_dir, "/workspace/data/"):
                return False
        else:
            # Unix: use tar + ssh pipe
            tar_cmd = f"tar -czf - -C {images_dir.parent} {images_dir.name}"
            full_cmd = f"{tar_cmd} | {ssh_cmd} '{remote_cmd}'"

            result = subprocess.run(full_cmd, shell=True)
            if result.returncode != 0:
                return False

        # Update cache
        manifest = {
            "instance_id": self.instance.instance_id,
            "local_path": str(images_dir),
        }
        self._save_manifest("images", manifest)
        return True

    def upload_test_images(self, test_file: Path, force: bool = False) -> bool:
        """
        Upload only the images referenced in the test file.

        Much faster than uploading the full dataset (114MB vs 4.3GB).

        Args:
            test_file: Path to test JSONL file
            force: Force upload even if cached

        Returns:
            True if upload successful
        """
        if not test_file.exists():
            print(f"Test file not found: {test_file}")
            return False

        # Check cache
        if not force:
            manifest = self._load_manifest("test_images")
            if manifest.get("instance_id") == self.instance.instance_id:
                print(f"  Test images already uploaded to this instance (cached)")
                return True

        # Get list of unique images from test file
        images = set()
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    images.add(data['images'][0])

        print(f"  Uploading {len(images)} test images...")

        # Create temp directory with same structure
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for img_path in images:
                src = Path(img_path)
                if not src.exists():
                    print(f"    Warning: Image not found: {src}")
                    continue

                # Extract relative path from "Fetal Ultrasound/..."
                match = FETAL_ULTRASOUND_PATTERN.search(str(src))
                if match:
                    rel_path = match.group(1).replace('\\', '/')
                    dst = tmpdir / "Fetal Ultrasound" / rel_path
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

            # Upload the temp directory
            src_dir = tmpdir / "Fetal Ultrasound"
            if src_dir.exists():
                print(f"  Uploading test images ({sum(f.stat().st_size for f in src_dir.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB)...")
                if not self.instance.upload_directory(src_dir, "/workspace/data/"):
                    return False

        # Update cache
        manifest = {
            "instance_id": self.instance.instance_id,
            "test_file": str(test_file),
            "image_count": len(images),
        }
        self._save_manifest("test_images", manifest)
        return True

    def upload_adapter(self, adapter_path: Path, force: bool = False) -> bool:
        """
        Upload LoRA adapter for fine-tuned model.

        Args:
            adapter_path: Path to adapter directory
            force: Force upload even if cached

        Returns:
            True if upload successful
        """
        if not adapter_path.exists():
            print(f"Adapter not found: {adapter_path}")
            return False

        if not force:
            manifest = self._load_manifest("adapter")
            if manifest.get("instance_id") == self.instance.instance_id:
                print(f"  Adapter already uploaded to this instance (cached)")
                return True

        # Upload
        remote_path = f"{REMOTE_MODELS_DIR}/"
        print(f"  Uploading adapter: {adapter_path.name}/")
        if not self.instance.upload_directory(adapter_path, remote_path):
            return False

        # Update cache
        manifest = {
            "instance_id": self.instance.instance_id,
            "local_path": str(adapter_path),
            "remote_path": f"{REMOTE_MODELS_DIR}/{adapter_path.name}",
        }
        self._save_manifest("adapter", manifest)
        return True

    def upload_script(self, script_path: Path, remote_name: Optional[str] = None) -> bool:
        """
        Upload a Python script.

        Args:
            script_path: Path to script file
            remote_name: Remote filename (defaults to original name)

        Returns:
            True if upload successful
        """
        if not script_path.exists():
            print(f"Script not found: {script_path}")
            return False

        remote_name = remote_name or script_path.name
        remote_path = f"{REMOTE_WORKSPACE}/scripts/{remote_name}"

        print(f"  Uploading script: {script_path.name}")
        return self.instance.upload_file(script_path, remote_path)

    def upload_training_data(self, train_file: Path, val_file: Path = None, force: bool = False) -> bool:
        """
        Upload training and validation data files.

        Args:
            train_file: Path to training JSONL file
            val_file: Path to validation JSONL file (optional)
            force: Force upload even if cached

        Returns:
            True if upload successful
        """
        if not train_file.exists():
            print(f"Training file not found: {train_file}")
            return False

        # Upload training file
        remote_train = f"{REMOTE_DATA_DIR}/train.jsonl"
        print(f"  Uploading training data: {train_file.name}")
        if not self.instance.upload_file(train_file, remote_train):
            return False

        # Upload validation file if provided
        if val_file and val_file.exists():
            remote_val = f"{REMOTE_DATA_DIR}/val.jsonl"
            print(f"  Uploading validation data: {val_file.name}")
            if not self.instance.upload_file(val_file, remote_val):
                return False

        return True

    def upload_training_images(self, train_file: Path, val_file: Path = None, force: bool = False) -> bool:
        """
        Download training images from Google Drive to remote instance.

        Instead of uploading from local (slow), downloads directly from Google Drive
        to the vast.ai instance (fast server-to-server transfer).

        Args:
            train_file: Path to training JSONL file (used to count images needed)
            val_file: Path to validation JSONL file (optional)
            force: Force download even if cached

        Returns:
            True if download successful
        """
        # Google Drive file ID for the Fetal Ultrasound dataset
        GDRIVE_FILE_ID = "11wa8pIeA8YrV7aBBJrhcKSAJPxPuA6wJ"

        # Check if images already exist on remote
        ret, stdout, _ = self.instance.run_ssh(
            "ls /workspace/data/Fetal\\ Ultrasound/ 2>/dev/null | wc -l",
            timeout=30
        )
        if ret == 0 and stdout.strip() and int(stdout.strip()) > 10:
            print(f"  Images already present on instance ({stdout.strip()} dirs)")
            return True

        # Count images needed (for logging)
        images = set()
        for data_file in [train_file, val_file]:
            if data_file and data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            for img_path in data.get('images', []):
                                images.add(img_path)

        print(f"  Downloading {len(images)} training images from Google Drive...")

        # Install gdown and download dataset
        download_cmd = f"""
        set -e
        pip install -q gdown
        cd /workspace/data
        echo "Downloading dataset from Google Drive..."
        gdown --id {GDRIVE_FILE_ID} -O fetal_ultrasound.zip
        echo "Extracting..."
        unzip -q -o fetal_ultrasound.zip
        rm fetal_ultrasound.zip
        echo "Done. Contents:"
        ls -la
        """

        print("  Installing gdown and downloading from Google Drive...")
        ret, stdout, stderr = self.instance.run_ssh(download_cmd, timeout=600)

        if ret != 0:
            print(f"  Download failed: {stderr or stdout}")
            return False

        print("  Download complete!")
        return True

    def download_adapter(self, local_dir: Path) -> Optional[Path]:
        """
        Download trained LoRA adapter from remote.

        Args:
            local_dir: Local directory to save adapter

        Returns:
            Path to downloaded adapter directory, or None if not found
        """
        local_dir.mkdir(parents=True, exist_ok=True)

        # Find adapter directories (format: model_timestamp/adapter)
        pattern = f"{REMOTE_OUTPUT_DIR}/*/adapter"
        ret, stdout, _ = self.instance.run_ssh(f"ls -d {pattern} 2>/dev/null | tail -1", timeout=30)

        if ret != 0 or not stdout.strip():
            # Try alternative pattern
            pattern = f"{REMOTE_OUTPUT_DIR}/*/*.safetensors"
            ret, stdout, _ = self.instance.run_ssh(f"ls {pattern} 2>/dev/null | head -1", timeout=30)

            if ret != 0 or not stdout.strip():
                print("No adapter found on remote")
                return None

            # Get parent directory
            adapter_dir = str(Path(stdout.strip()).parent)
        else:
            adapter_dir = stdout.strip()

        print(f"  Found adapter: {adapter_dir}")

        # Download adapter directory
        adapter_name = Path(adapter_dir).name
        if adapter_name == "adapter":
            adapter_name = Path(adapter_dir).parent.name

        local_adapter = local_dir / adapter_name
        local_adapter.mkdir(parents=True, exist_ok=True)

        # Download all files in adapter directory
        ret, files, _ = self.instance.run_ssh(f"ls {adapter_dir}", timeout=30)
        if ret == 0:
            for filename in files.strip().split('\n'):
                filename = filename.strip()
                if filename:
                    remote_file = f"{adapter_dir}/{filename}"
                    local_file = local_adapter / filename
                    self.instance.download_file(remote_file, local_file)

        # Also download config.json and metrics.json from parent if present
        parent_dir = str(Path(adapter_dir).parent)
        for extra_file in ['config.json', 'metrics.json']:
            ret, _, _ = self.instance.run_ssh(f"test -f {parent_dir}/{extra_file}", timeout=10)
            if ret == 0:
                self.instance.download_file(f"{parent_dir}/{extra_file}", local_adapter / extra_file)

        return local_adapter if (local_adapter / "adapter_model.safetensors").exists() or any(local_adapter.glob("*.safetensors")) else None

    def download_predictions(self, local_dir: Path) -> Optional[Path]:
        """
        Download prediction files from remote.

        Args:
            local_dir: Local directory to save predictions

        Returns:
            Path to downloaded file, or None if not found
        """
        local_dir.mkdir(parents=True, exist_ok=True)

        # Find prediction files
        pattern = f"{REMOTE_OUTPUT_DIR}/predictions_*.jsonl"
        downloaded = self.instance.download_glob(pattern, local_dir)

        if downloaded:
            return downloaded[0]  # Return most recent
        return None

    def download_results(self, local_dir: Path) -> list:
        """
        Download all result files from remote.

        Args:
            local_dir: Local directory to save results

        Returns:
            List of downloaded file paths
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []

        # Download predictions
        pattern = f"{REMOTE_OUTPUT_DIR}/predictions_*.jsonl"
        downloaded.extend(self.instance.download_glob(pattern, local_dir))

        # Download checkpoints if any
        pattern = f"{REMOTE_OUTPUT_DIR}/*checkpoint*.jsonl"
        downloaded.extend(self.instance.download_glob(pattern, local_dir))

        # Download logs
        pattern = f"{REMOTE_OUTPUT_DIR}/*.log"
        downloaded.extend(self.instance.download_glob(pattern, local_dir / "logs"))

        return downloaded

    def clear_cache(self):
        """Clear upload cache manifests."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        print("Upload cache cleared.")


def setup_environment(instance: VastInstance, use_flash_attn: bool = False) -> bool:
    """
    Setup Python environment on remote instance.

    Args:
        instance: Connected VastInstance
        use_flash_attn: Whether to install flash-attn (requires compilation)

    Returns:
        True if setup successful
    """
    print("\nSetting up Python environment...")

    # Base dependencies with pinned versions for reproducibility
    # Note: Qwen2.5-VL requires transformers >= 4.49, accelerate >= 1.3 for compatibility
    setup_cmds = """
    set -e  # Exit on any error
    pip install -q --upgrade pip
    pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
    pip install -q transformers>=4.49.0 accelerate>=1.3.0 bitsandbytes==0.45.0 peft>=0.14.0
    pip install -q pillow>=10.0.0 tqdm
    pip install -q qwen-vl-utils  # For Qwen2-VL and Qwen2.5-VL models
    """

    if use_flash_attn:
        setup_cmds += """
    pip install -q flash-attn --no-build-isolation
    """

    ret, stdout, stderr = instance.run_ssh(setup_cmds, timeout=600)

    if ret != 0:
        print(f"Environment setup failed: {stderr}")
        return False

    print("Environment setup complete.")
    return True


def check_gpu(instance: VastInstance) -> dict:
    """
    Check GPU status on remote instance.

    Returns:
        Dict with GPU info or empty dict on error
    """
    cmd = "nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader"
    ret, stdout, _ = instance.run_ssh(cmd, timeout=30)

    if ret != 0:
        return {}

    parts = stdout.strip().split(",")
    if len(parts) >= 4:
        return {
            "name": parts[0].strip(),
            "memory_total": parts[1].strip(),
            "memory_free": parts[2].strip(),
            "temperature": parts[3].strip(),
        }
    return {}


def validate_environment(instance: VastInstance, required_packages: list = None) -> dict:
    """
    Validate GPU and Python environment on remote instance.

    Performs comprehensive checks before running jobs to catch failures early.

    Args:
        instance: Connected VastInstance
        required_packages: Additional packages to verify (default: transformers, bitsandbytes, peft)

    Returns:
        Dict with validation results:
        - success: bool
        - gpu: dict with GPU info
        - cuda_version: str
        - pytorch_version: str
        - errors: list of error messages
    """
    if required_packages is None:
        required_packages = ["transformers", "bitsandbytes", "peft", "accelerate"]

    result = {
        "success": True,
        "gpu": {},
        "cuda_version": None,
        "pytorch_version": None,
        "driver_version": None,
        "errors": [],
    }

    # Check 1: GPU accessible
    print("  Validating GPU...")
    ret, stdout, stderr = instance.run_ssh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader", timeout=30)
    if ret != 0:
        result["success"] = False
        result["errors"].append(f"GPU not accessible: {stderr}")
        return result

    parts = stdout.strip().split(",")
    if len(parts) >= 3:
        result["gpu"]["name"] = parts[0].strip()
        result["driver_version"] = parts[1].strip()
        result["gpu"]["memory"] = parts[2].strip()
        print(f"    GPU: {result['gpu']['name']}, Driver: {result['driver_version']}")

    # Check 2: PyTorch + CUDA
    print("  Validating PyTorch CUDA...")
    cuda_check = """python3 -c "
import torch
print(f'pytorch:{torch.__version__}')
print(f'cuda:{torch.version.cuda}')
print(f'available:{torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'device:{torch.cuda.get_device_name(0)}')
"
"""
    ret, stdout, stderr = instance.run_ssh(cuda_check, timeout=60)
    if ret != 0:
        result["success"] = False
        result["errors"].append(f"PyTorch CUDA check failed: {stderr}")
    else:
        for line in stdout.strip().split('\n'):
            if line.startswith('pytorch:'):
                result["pytorch_version"] = line.split(':')[1]
            elif line.startswith('cuda:'):
                result["cuda_version"] = line.split(':')[1]
            elif line.startswith('available:'):
                if line.split(':')[1].strip().lower() != 'true':
                    result["success"] = False
                    result["errors"].append("CUDA not available in PyTorch")

        if result["pytorch_version"]:
            print(f"    PyTorch: {result['pytorch_version']}, CUDA: {result['cuda_version']}")

    # Check 3: Required packages
    print("  Validating packages...")
    for pkg in required_packages:
        ret, _, stderr = instance.run_ssh(f'python3 -c "import {pkg}"', timeout=30)
        if ret != 0:
            result["success"] = False
            result["errors"].append(f"Package '{pkg}' import failed")
            print(f"    FAILED: {pkg}")
        else:
            print(f"    OK: {pkg}")

    return result


def setup_environment_with_retry(
    instance: VastInstance,
    max_retries: int = 3,
    use_flash_attn: bool = False,
    retry_delay: int = 10,
) -> bool:
    """
    Setup Python environment with retry logic and validation.

    Args:
        instance: Connected VastInstance
        max_retries: Maximum setup attempts
        use_flash_attn: Whether to install flash-attn
        retry_delay: Seconds between retries

    Returns:
        True if setup and validation successful
    """
    import time

    for attempt in range(1, max_retries + 1):
        print(f"\nSetup attempt {attempt}/{max_retries}...")

        # Run setup
        if not setup_environment(instance, use_flash_attn=use_flash_attn):
            print(f"  Setup failed on attempt {attempt}")
            if attempt < max_retries:
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            continue

        # Validate
        validation = validate_environment(instance)
        if validation["success"]:
            print("\nEnvironment validated successfully!")
            return True

        print(f"  Validation failed: {validation['errors']}")
        if attempt < max_retries:
            print(f"  Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    print(f"\nSetup failed after {max_retries} attempts")
    return False
