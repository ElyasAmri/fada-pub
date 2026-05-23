"""
RunPod GPU instance management using official SDK.
"""

import os
import time
import subprocess
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

import runpod

# Load environment from project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env.local")
runpod.api_key = os.environ.get("RUNPOD_API_KEY")


class RunPodInstance:
    """Manages a single RunPod GPU pod."""

    def __init__(self):
        if not runpod.api_key:
            raise ValueError("RUNPOD_API_KEY not set")

        self.pod_id: Optional[str] = None
        self.ssh_host: Optional[str] = None
        self.ssh_port: Optional[int] = None

    def create_pod(
        self,
        gpu_type: str = "NVIDIA GeForce RTX 4090",
        image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        disk_gb: int = 50,
        name: str = "benchmark",
        env_vars: Optional[dict] = None
    ) -> Optional[str]:
        """Create a new pod with optional environment variables."""
        try:
            create_kwargs = {
                "name": name,
                "image_name": image,
                "gpu_type_id": gpu_type,
                "gpu_count": 1,
                "volume_in_gb": 0,
                "container_disk_in_gb": disk_gb,
                "ports": "22/tcp",
                "cloud_type": "SECURE",
            }

            # Add environment variables if provided
            if env_vars:
                create_kwargs["env"] = env_vars
                print(f"Setting env vars: {list(env_vars.keys())}")

            pod = runpod.create_pod(**create_kwargs)
            self.pod_id = pod["id"]
            print(f"Created pod {self.pod_id}")
            return self.pod_id
        except Exception as e:
            print(f"Failed to create pod: {e}")
            return None

    def wait_for_ready(self, timeout: int = 180) -> bool:
        """Wait for pod to be ready with SSH access."""
        if not self.pod_id:
            return False

        start = time.time()
        while time.time() - start < timeout:
            try:
                status = runpod.get_pod(self.pod_id)
                runtime = status.get("runtime")

                if runtime and runtime.get("ports"):
                    for port in runtime["ports"]:
                        if port.get("privatePort") == 22 and port.get("isIpPublic"):
                            self.ssh_host = port["ip"]
                            self.ssh_port = port["publicPort"]
                            print(f"Pod ready: {self.ssh_host}:{self.ssh_port}")
                            return True

                desired = status.get("desiredStatus", "unknown")
                print(f"Waiting for pod... (status: {desired})")
            except Exception as e:
                print(f"Status check failed: {e}")

            time.sleep(10)

        return False

    def run_ssh(self, command: str, timeout: int = 120) -> tuple:
        """Run SSH command on the pod."""
        if not self.ssh_host or not self.ssh_port:
            return 1, "", "SSH not available"

        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-o", "LogLevel=ERROR",
            "-o", "BatchMode=yes",
            "-o", "PasswordAuthentication=no",
            "-i", os.path.expanduser("~/.ssh/id_rsa"),
            "-p", str(self.ssh_port),
            f"root@{self.ssh_host}",
            command
        ]

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                timeout=timeout
            )
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')
            return result.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)

    def destroy_pod(self):
        """Terminate the pod."""
        if not self.pod_id:
            return

        try:
            runpod.terminate_pod(self.pod_id)
            print(f"Destroyed pod {self.pod_id}")
        except Exception as e:
            print(f"Failed to destroy pod: {e}")


def list_all_pods():
    """List all pods."""
    return runpod.get_pods()


def destroy_all_pods():
    """Destroy all pods."""
    pods = list_all_pods()
    for pod in pods:
        print(f"Destroying pod {pod['id']}...")
        runpod.terminate_pod(pod["id"])
    print(f"Destroyed {len(pods)} pods")


if __name__ == "__main__":
    print("Testing RunPod connection...")

    # List GPUs
    gpus = runpod.get_gpus()
    rtx4090 = [g for g in gpus if "4090" in g.get("displayName", "")]
    if rtx4090:
        print(f"RTX 4090 available: {rtx4090[0]['id']}")

    # Check pods
    pods = list_all_pods()
    print(f"Existing pods: {len(pods)}")
