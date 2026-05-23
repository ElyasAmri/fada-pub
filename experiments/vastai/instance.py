"""
Vast.ai Instance Management: Create, monitor, and destroy GPU instances.

Wraps the vastai CLI tool with a clean Python API.
"""

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .presets import DOCKER_IMAGE, MIN_DRIVER_VERSION
except ImportError:
    from presets import DOCKER_IMAGE, MIN_DRIVER_VERSION


@dataclass
class NetworkVolume:
    """Represents a Vast.ai network volume."""
    volume_id: int
    size_gb: float
    storage_cost: float  # $/GB/month
    geolocation: str
    reliability: float

    @classmethod
    def from_api(cls, data: Dict) -> "NetworkVolume":
        return cls(
            volume_id=data["id"],
            size_gb=float(data.get("disk_space", 0)),
            storage_cost=float(data.get("storage_cost", 0)),
            geolocation=data.get("geolocation", "Unknown"),
            reliability=float(data.get("reliability", 0)),
        )


@dataclass
class GPUOffer:
    """Represents a Vast.ai GPU offer."""
    offer_id: int
    gpu_name: str
    gpu_ram: int  # MB
    price_per_hour: float
    download_speed: float
    reliability: float
    disk_space: float
    cuda_version: str = ""

    @classmethod
    def from_api(cls, data: Dict) -> "GPUOffer":
        return cls(
            offer_id=data["id"],
            gpu_name=data.get("gpu_name", "Unknown"),
            gpu_ram=int(data.get("gpu_ram", 0)),
            price_per_hour=float(data.get("dph_total", 999)),
            download_speed=float(data.get("inet_down", 0)),
            reliability=float(data.get("reliability", 0)),
            disk_space=float(data.get("disk_space", 0)),
            cuda_version=str(data.get("cuda_max_good", "")),
        )


class VastInstance:
    """Manage a single Vast.ai instance."""

    # Default Docker images for different use cases
    IMAGES = {
        "pytorch": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        "pytorch_nightly": "pytorch/pytorch:nightly-devel",
        "cuda": "nvidia/cuda:12.4.0-devel-ubuntu22.04",
    }

    def __init__(self, vastai_path: Optional[str] = None):
        """Initialize with vastai CLI path."""
        self.vastai = vastai_path or self._find_vastai()
        self.instance_id: Optional[int] = None
        self.ssh_host: Optional[str] = None
        self.ssh_port: Optional[int] = None
        self._offer: Optional[GPUOffer] = None

    def _find_vastai(self) -> str:
        """Find vastai CLI in common locations."""
        import shutil

        # Check if vastai is in PATH
        vastai = shutil.which("vastai")
        if vastai:
            return vastai

        # Check common locations
        common_paths = [
            Path.home() / ".local" / "bin" / "vastai",
            Path(__file__).parent.parent.parent / "venv" / "Scripts" / "vastai.exe",
            Path(__file__).parent.parent.parent / "venv" / "bin" / "vastai",
        ]

        for path in common_paths:
            if path.exists():
                return str(path)

        # Fall back to assuming it's in PATH
        return "vastai"

    def _run_vastai(self, *args, timeout: int = 60) -> Tuple[int, str, str]:
        """Run vastai CLI command."""
        cmd = [self.vastai] + list(args)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except FileNotFoundError:
            return 1, "", f"vastai CLI not found at {self.vastai}"

    # =========================================================================
    # GPU Search
    # =========================================================================

    def search_offers(
        self,
        gpu_name: Optional[str] = None,
        min_vram: int = 20,
        max_price: float = 1.0,
        min_price: float = 0.20,  # Minimum price to avoid unreliable cheap hosts
        min_reliability: float = 0.95,
        min_download: int = 100,
        num_gpus: int = 1,
        min_driver_version: Optional[int] = None,
        geolocation: Optional[str] = None,
    ) -> List[GPUOffer]:
        """
        Search for available GPU offers.

        Args:
            gpu_name: GPU type (e.g., "RTX_4090", "A100", "H100")
            min_vram: Minimum VRAM in GB
            max_price: Maximum price per hour in USD
            min_price: Minimum price to filter out unreliable cheap hosts
            min_reliability: Minimum reliability score (0-1)
            min_download: Minimum download speed in Mbps
            num_gpus: Number of GPUs required
            min_driver_version: Minimum NVIDIA driver version (e.g., 535 for CUDA 12.4+)
            geolocation: Country code filter (e.g., "IS" for Iceland)

        Returns:
            List of GPUOffer sorted by price (cheapest first)
        """
        query_parts = [
            f"gpu_ram>={min_vram}",  # API uses GB
            f"dph<{max_price}",
            f"dph>{min_price}",  # Filter out unreliable cheap hosts
            f"reliability>{min_reliability}",
            f"inet_down>{min_download}",
            f"num_gpus={num_gpus}",
        ]

        if gpu_name:
            query_parts.append(f"gpu_name={gpu_name.replace(' ', '_')}")

        if geolocation:
            query_parts.append(f"geolocation={geolocation}")

        # Filter by driver version for CUDA compatibility
        if min_driver_version:
            query_parts.append(f"driver_version>={min_driver_version}")

        query = " ".join(query_parts)

        ret, stdout, stderr = self._run_vastai("search", "offers", query, "--raw")

        if ret != 0:
            print(f"Error searching offers: {stderr}")
            return []

        try:
            data = json.loads(stdout)
            offers = [GPUOffer.from_api(o) for o in data]
            offers.sort(key=lambda x: x.price_per_hour)
            return offers
        except json.JSONDecodeError:
            print(f"Failed to parse offers: {stdout[:200]}")
            return []

    def print_offers(self, offers: List[GPUOffer], limit: int = 10):
        """Print offers in a formatted table."""
        print(f"\n{'='*80}")
        print(f"{'ID':>10} | {'GPU':^20} | {'VRAM':^8} | {'$/hr':^8} | {'DL':^8} | {'Rel':^6}")
        print(f"{'='*80}")

        for offer in offers[:limit]:
            vram_gb = offer.gpu_ram / 1000
            print(
                f"{offer.offer_id:>10} | "
                f"{offer.gpu_name[:20]:^20} | "
                f"{vram_gb:^8.0f} | "
                f"${offer.price_per_hour:^7.3f} | "
                f"{offer.download_speed:^8.0f} | "
                f"{offer.reliability:^6.2f}"
            )

        print(f"{'='*80}")
        print(f"Showing {min(limit, len(offers))} of {len(offers)} offers")

    # =========================================================================
    # Instance Lifecycle
    # =========================================================================

    def create(
        self,
        offer_id: int,
        disk_gb: int = 10,
        image: Optional[str] = None,
        network_volume_id: Optional[int] = None,
        mount_path: str = "/workspace/data",
    ) -> bool:
        """
        Create a new instance from an offer.

        Args:
            offer_id: The offer ID to use
            disk_gb: Local disk size in GB (minimal if using network volume)
            image: Docker image (key from IMAGES dict, full image name, or None for default from presets)
            network_volume_id: Optional network volume ID to attach
            mount_path: Mount path for network volume (default: /workspace/data)

        Returns:
            True if instance was created successfully
        """
        # Resolve image name
        if image is None:
            docker_image = DOCKER_IMAGE
        else:
            docker_image = self.IMAGES.get(image, image)

        args = [
            "create", "instance", str(offer_id),
            "--image", docker_image,
            "--disk", str(disk_gb),
            "--ssh", "--direct",
        ]

        # Attach network volume if provided
        if network_volume_id:
            args.extend(["--link-volume", str(network_volume_id)])
            args.extend(["--mount-path", mount_path])

        ret, stdout, stderr = self._run_vastai(*args, timeout=120)

        if ret != 0:
            print(f"Error creating instance: {stderr}")
            return False

        # Parse instance ID from output
        # Output format: "{'success': True, 'new_contract': 12345678}"
        try:
            match = re.search(r"'new_contract':\s*(\d+)", stdout)
            if match:
                self.instance_id = int(match.group(1))
                return True

            # Fallback: try to find any number
            match = re.search(r"(\d{6,})", stdout)
            if match:
                self.instance_id = int(match.group(1))
                return True
        except Exception as e:
            print(f"Failed to parse instance ID: {e}")

        print(f"Create output: {stdout}")
        return False

    def get_info(self, instance_id: Optional[int] = None) -> Optional[Dict]:
        """Get instance information."""
        iid = instance_id or self.instance_id
        if not iid:
            return None

        ret, stdout, stderr = self._run_vastai("show", "instance", str(iid), "--raw")

        if ret != 0:
            return None

        try:
            data = json.loads(stdout)
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            return data
        except json.JSONDecodeError:
            return None

    def wait_until_ready(self, timeout: int = 600, poll_interval: int = 15) -> bool:
        """
        Wait for instance to be ready for SSH connections.

        Args:
            timeout: Maximum seconds to wait
            poll_interval: Seconds between status checks

        Returns:
            True if instance is ready, False if timeout or error
        """
        if not self.instance_id:
            return False

        start_time = time.time()

        while time.time() - start_time < timeout:
            info = self.get_info()

            if info:
                status = info.get("actual_status", "")

                if status == "running":
                    # Extract SSH connection details
                    self.ssh_host = info.get("public_ipaddr", "")
                    ports = info.get("ports", {})
                    ssh_ports = ports.get("22/tcp", [])

                    if ssh_ports:
                        self.ssh_port = int(ssh_ports[0].get("HostPort", 22))
                    else:
                        self.ssh_port = info.get("ssh_port", 22)

                    if self.ssh_host and self.ssh_port:
                        return True

                elif status in ("error", "terminated", "failed", "exited"):
                    print(f"Instance {self.instance_id} failed with status: {status}")
                    return False

                print(f"Waiting for instance {self.instance_id}... (status: {status})")
            else:
                print(f"Waiting for instance {self.instance_id}... (status: unknown)")

            time.sleep(poll_interval)

        print(f"Timeout waiting for instance {self.instance_id}")
        return False

    def validate_instance(self, max_retries: int = 3, retry_delay: int = 15) -> Dict:
        """
        Validate the running instance meets requirements.

        Args:
            max_retries: Number of retries for SSH commands
            retry_delay: Seconds to wait between retries

        Returns:
            Dict with:
            - valid: bool - Whether instance meets all requirements
            - driver_version: str - NVIDIA driver version
            - cuda_version: str - CUDA runtime version
            - errors: list - List of validation errors
        """
        for attempt in range(1, max_retries + 1):
            result = {"valid": True, "errors": []}

            # Check driver version
            ret, stdout, _ = self.run_ssh(
                "nvidia-smi --query-gpu=driver_version --format=csv,noheader",
                timeout=30
            )
            if ret == 0 and stdout.strip():
                driver = stdout.strip()
                result["driver_version"] = driver
                # Extract major version (e.g., "535.104.05" -> 535)
                try:
                    driver_major = int(driver.split('.')[0])
                    if driver_major < MIN_DRIVER_VERSION:
                        result["valid"] = False
                        result["errors"].append(
                            f"Driver {driver} (v{driver_major}) < required v{MIN_DRIVER_VERSION}"
                        )
                except (ValueError, IndexError):
                    result["valid"] = False
                    result["errors"].append(f"Could not parse driver version: {driver}")
            else:
                result["valid"] = False
                result["errors"].append("Could not query driver version")
                result["driver_version"] = "unknown"

            # Check CUDA version via nvidia-smi (more reliable than nvcc)
            ret, stdout, _ = self.run_ssh(
                "nvidia-smi --query-gpu=driver_version,name --format=csv,noheader",
                timeout=30
            )
            if ret == 0 and stdout.strip():
                # nvidia-smi doesn't return CUDA version directly, get from nvidia-smi main output
                ret2, stdout2, _ = self.run_ssh(
                    "nvidia-smi | grep 'CUDA Version' | awk '{print $9}'",
                    timeout=30
                )
                if ret2 == 0 and stdout2.strip():
                    result["cuda_version"] = stdout2.strip()
                else:
                    result["cuda_version"] = "unknown"
            else:
                result["cuda_version"] = "unknown"

            # If validation passed or we got driver info, return
            if result.get("valid", False) or result.get("driver_version") != "unknown":
                return result

            # Retry if SSH commands failed
            if attempt < max_retries:
                print(f"  Validation attempt {attempt} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        return result

    def destroy(self, instance_id: Optional[int] = None) -> bool:
        """Destroy/terminate an instance."""
        iid = instance_id or self.instance_id
        if not iid:
            return False

        ret, stdout, stderr = self._run_vastai("destroy", "instance", str(iid))
        return ret == 0

    def stop_instance(self, instance_id: Optional[int] = None) -> bool:
        """Stop an instance (can be restarted later)."""
        iid = instance_id or self.instance_id
        if not iid:
            return False

        ret, stdout, stderr = self._run_vastai("stop", "instance", str(iid))
        return ret == 0

    def start_instance(self, instance_id: Optional[int] = None) -> bool:
        """Start a stopped instance."""
        iid = instance_id or self.instance_id
        if not iid:
            return False

        ret, stdout, stderr = self._run_vastai("start", "instance", str(iid))
        return ret == 0

    def destroy_instance(self, instance_id: Optional[int] = None) -> bool:
        """Alias for destroy() for consistency."""
        return self.destroy(instance_id)

    # =========================================================================
    # Network Volume Operations
    # =========================================================================

    def search_network_volumes(
        self,
        min_disk_space: int = 20,
        min_reliability: float = 0.95,
        geolocation: Optional[str] = None,
    ) -> List[NetworkVolume]:
        """
        Search for available network volume offers.

        Args:
            min_disk_space: Minimum available space in GB
            min_reliability: Minimum reliability score (0-1)
            geolocation: Optional country code filter (e.g., "US", "NL")

        Returns:
            List of NetworkVolume sorted by storage cost (cheapest first)
        """
        query_parts = [
            f"disk_space>{min_disk_space}",
            f"reliability>{min_reliability}",
        ]

        if geolocation:
            query_parts.append(f"geolocation={geolocation}")

        query = " ".join(query_parts)

        ret, stdout, stderr = self._run_vastai(
            "search", "network-volumes", query, "--raw", "-o", "storage_cost"
        )

        if ret != 0:
            print(f"Error searching network volumes: {stderr}")
            return []

        try:
            data = json.loads(stdout)
            volumes = [NetworkVolume.from_api(v) for v in data]
            volumes.sort(key=lambda x: x.storage_cost)
            return volumes
        except json.JSONDecodeError:
            print(f"Failed to parse volumes: {stdout[:200]}")
            return []

    def create_network_volume(
        self,
        offer_id: int,
        size_gb: int = 30,
        name: Optional[str] = None,
    ) -> Optional[int]:
        """
        Create a network volume from an offer.

        Args:
            offer_id: The network volume offer ID
            size_gb: Size in GB
            name: Optional name for the volume

        Returns:
            Volume ID if created, None otherwise
        """
        args = ["create", "network-volume", str(offer_id), "-s", str(size_gb)]
        if name:
            args.extend(["-n", name])

        ret, stdout, stderr = self._run_vastai(*args, timeout=60)

        if ret != 0:
            print(f"Error creating network volume: {stderr}")
            return None

        # Parse volume ID from output
        # Format can be: "Created. {'success': True, 'volume_name': 'V.30576677', ...}"
        # or: "{'success': True, 'new_contract': 12345}"
        try:
            # Try to find volume_name with ID (e.g., V.30576677)
            match = re.search(r"'volume_name':\s*'V\.(\d+)'", stdout)
            if match:
                return int(match.group(1))

            # Try new_contract format
            match = re.search(r"'?new_contract'?:\s*(\d+)", stdout)
            if match:
                return int(match.group(1))

            # Try JSON parsing
            # Extract JSON part from output like "Created. {...}"
            json_match = re.search(r"\{.*\}", stdout)
            if json_match:
                data = json.loads(json_match.group().replace("'", '"'))
                if "volume_name" in data:
                    # Extract ID from V.12345 format
                    vol_match = re.search(r"V\.(\d+)", data["volume_name"])
                    if vol_match:
                        return int(vol_match.group(1))
                if "new_contract" in data:
                    return data["new_contract"]
        except Exception as e:
            print(f"Failed to parse volume ID: {e}, output: {stdout}")

        return None

    def delete_network_volume(self, volume_id: int) -> bool:
        """Delete a network volume."""
        ret, stdout, stderr = self._run_vastai("delete", "volume", str(volume_id))
        return ret == 0

    def show_volumes(self) -> List[Dict]:
        """List all rented volumes."""
        ret, stdout, stderr = self._run_vastai("show", "volumes", "--raw")

        if ret != 0:
            return []

        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return []

    def create_instance(
        self,
        offer_id: int,
        image: Optional[str] = None,
        disk_gb: int = 10,
        network_volume_id: Optional[int] = None,
        mount_path: str = "/workspace/data",
    ) -> Optional[int]:
        """Create instance and return instance ID (alias for create)."""
        if self.create(
            offer_id,
            disk_gb=disk_gb,
            image=image,
            network_volume_id=network_volume_id,
            mount_path=mount_path,
        ):
            return self.instance_id
        return None

    def wait_for_ready(self, timeout: int = 600) -> bool:
        """Alias for wait_until_ready()."""
        return self.wait_until_ready(timeout=timeout)

    # =========================================================================
    # SSH Operations
    # =========================================================================

    def run_ssh(
        self,
        command: str,
        timeout: int = 300,
        capture: bool = True,
    ) -> Tuple[int, str, str]:
        """
        Run command on remote instance via SSH.

        Args:
            command: Shell command to run
            timeout: Command timeout in seconds
            capture: Whether to capture output

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        if not self.ssh_host or not self.ssh_port:
            return 1, "", "No SSH connection info"

        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-o", "ServerAliveInterval=60",
            "-p", str(self.ssh_port),
            f"root@{self.ssh_host}",
            command,
        ]

        try:
            if capture:
                result = subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    timeout=timeout,
                )
                # Decode with UTF-8 and replace errors to handle special chars
                stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
                stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""
                return result.returncode, stdout, stderr
            else:
                result = subprocess.run(ssh_cmd, timeout=timeout)
                return result.returncode, "", ""
        except subprocess.TimeoutExpired:
            return 1, "", f"SSH command timed out after {timeout}s"
        except Exception as e:
            return 1, "", str(e)

    def upload_file(self, local_path: Path, remote_path: str, show_progress: bool = True, max_retries: int = 3) -> bool:
        """Upload a file via SCP with retry logic."""
        if not self.ssh_host or not self.ssh_port:
            return False

        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-P", str(self.ssh_port),
            str(local_path),
            f"root@{self.ssh_host}:{remote_path}",
        ]

        if show_progress:
            print(f"  Uploading {local_path.name}...")

        for attempt in range(1, max_retries + 1):
            result = subprocess.run(scp_cmd, capture_output=True)
            if result.returncode == 0:
                return True
            if attempt < max_retries:
                print(f"    Upload failed (attempt {attempt}/{max_retries}), retrying in 5s...")
                time.sleep(5)

        if show_progress:
            print(f"    Upload failed after {max_retries} attempts")
        return False

    def upload_directory(self, local_dir: Path, remote_dir: str, show_progress: bool = True, max_retries: int = 5) -> bool:
        """Upload a directory via SCP with retry logic and longer timeout."""
        import platform

        if not self.ssh_host or not self.ssh_port:
            return False

        if show_progress:
            print(f"  Uploading {local_dir.name}/...")

        # Use SCP with longer timeouts for large transfers
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=120",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=10",
            "-P", str(self.ssh_port),
            "-r",
            str(local_dir),
            f"root@{self.ssh_host}:{remote_dir}",
        ]

        for attempt in range(1, max_retries + 1):
            try:
                # Longer timeout for large directory uploads (30 min)
                result = subprocess.run(scp_cmd, capture_output=True, timeout=1800)
                if result.returncode == 0:
                    return True
                if show_progress and result.stderr:
                    print(f"    SCP error: {result.stderr.decode()[:200]}")
            except subprocess.TimeoutExpired:
                if show_progress:
                    print(f"    Upload timed out after 30 min")

            if attempt < max_retries:
                wait_time = 15 * attempt  # Exponential backoff
                print(f"    Upload failed (attempt {attempt}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)

        if show_progress:
            print(f"    Upload failed after {max_retries} attempts")
        return False

    def download_file(self, remote_path: str, local_path: Path, show_progress: bool = True) -> bool:
        """Download a file via SCP."""
        if not self.ssh_host or not self.ssh_port:
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)

        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.ssh_port),
            f"root@{self.ssh_host}:{remote_path}",
            str(local_path),
        ]

        if show_progress:
            print(f"  Downloading to {local_path.name}...")

        result = subprocess.run(scp_cmd, capture_output=not show_progress)
        return result.returncode == 0

    def download_glob(self, remote_pattern: str, local_dir: Path) -> List[Path]:
        """Download files matching a pattern."""
        if not self.ssh_host or not self.ssh_port:
            return []

        # First, list files matching pattern
        ret, stdout, _ = self.run_ssh(f"ls {remote_pattern} 2>/dev/null", timeout=30)

        if ret != 0 or not stdout.strip():
            return []

        downloaded = []
        local_dir.mkdir(parents=True, exist_ok=True)

        for remote_path in stdout.strip().split("\n"):
            remote_path = remote_path.strip()
            if remote_path:
                filename = Path(remote_path).name
                local_path = local_dir / filename
                if self.download_file(remote_path, local_path, show_progress=True):
                    downloaded.append(local_path)

        return downloaded

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def connect(self, instance_id: int) -> bool:
        """Connect to an existing instance by ID."""
        self.instance_id = instance_id
        info = self.get_info()

        if not info:
            return False

        self.ssh_host = info.get("public_ipaddr", "")
        ports = info.get("ports", {})
        ssh_ports = ports.get("22/tcp", [])

        if ssh_ports:
            self.ssh_port = int(ssh_ports[0].get("HostPort", 22))
        else:
            self.ssh_port = info.get("ssh_port", 22)

        return bool(self.ssh_host and self.ssh_port)

    @property
    def ssh_command(self) -> str:
        """Get SSH command string for manual connection."""
        if self.ssh_host and self.ssh_port:
            return f"ssh -p {self.ssh_port} root@{self.ssh_host}"
        return ""

    def is_process_running(self, pattern: str) -> bool:
        """Check if a process matching pattern is running."""
        ret, _, _ = self.run_ssh(f"pgrep -f '{pattern}'", timeout=30)
        return ret == 0

    def tail_log(self, log_path: str, lines: int = 50) -> str:
        """Get last N lines from a log file."""
        ret, stdout, _ = self.run_ssh(f"tail -n {lines} {log_path} 2>/dev/null", timeout=30)
        return stdout if ret == 0 else ""
