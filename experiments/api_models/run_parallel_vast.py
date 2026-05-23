#!/usr/bin/env python3
"""
Parallel MedGemma-4B Inference on Multiple Vast.ai RTX 5090s

This script automates:
1. Searching for cheapest RTX 5090 machines
2. Renting N machines
3. Setting up vLLM with MedGemma-4B on each
4. Creating SSH tunnels
5. Launching parallel inference with sharding
6. Monitoring progress
7. Merging results
8. Terminating machines

Usage:
    python run_parallel_vast.py --num-machines 6 --start-from 7830 --total-images 19019
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class VastManager:
    """Manage Vast.ai instances for parallel inference"""

    def __init__(self, vastai_path: str = "vastai"):
        self.vastai = vastai_path
        self.instances: List[Dict] = []
        self.ssh_processes: List[subprocess.Popen] = []
        self.inference_processes: List[subprocess.Popen] = []

    def search_offers(self, gpu_name: str = "RTX 5090", num_gpus: int = 1,
                      min_ram: int = 30, max_price: float = 0.50) -> List[Dict]:
        """Search for available GPU offers"""
        # Query format: gpu_name=RTX_5090 num_gpus=1 gpu_ram>=30 dph<0.50 inet_down>100
        query = f"gpu_name={gpu_name.replace(' ', '_')} num_gpus={num_gpus} gpu_ram>={min_ram} dph<{max_price} inet_down>100 reliability>0.95"

        cmd = [self.vastai, "search", "offers", query, "--raw"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error searching offers: {result.stderr}")
            return []

        try:
            offers = json.loads(result.stdout)
            # Sort by price
            offers.sort(key=lambda x: x.get('dph_total', 999))
            return offers
        except json.JSONDecodeError:
            print(f"Failed to parse offers: {result.stdout[:500]}")
            return []

    def create_instance(self, offer_id: int, disk_gb: int = 100) -> Optional[int]:
        """Create a Vast.ai instance"""
        cmd = [
            self.vastai, "create", "instance", str(offer_id),
            "--image", "vastai/pytorch:@vastai-automatic-tag",
            "--disk", str(disk_gb),
            "--onstart-cmd", "entrypoint.sh",
            "--jupyter", "--ssh", "--direct"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error creating instance: {result.stderr}")
            return None

        # Parse instance ID from output
        # Output format: "Started. {'success': True, 'new_contract': 12345678}"
        try:
            # Look for contract ID in output
            import re
            match = re.search(r"'new_contract':\s*(\d+)", result.stdout)
            if match:
                return int(match.group(1))
            # Alternative format
            match = re.search(r"(\d+)", result.stdout)
            if match:
                return int(match.group(1))
        except Exception as e:
            print(f"Failed to parse instance ID: {e}")

        print(f"Create output: {result.stdout}")
        return None

    def wait_for_instance(self, instance_id: int, timeout: int = 600) -> bool:
        """Wait for instance to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            cmd = [self.vastai, "show", "instance", str(instance_id), "--raw"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                try:
                    instance = json.loads(result.stdout)
                    if isinstance(instance, list) and len(instance) > 0:
                        instance = instance[0]

                    status = instance.get('actual_status', '')
                    if status == 'running':
                        # Get SSH info
                        ssh_host = instance.get('ssh_host', '')
                        ssh_port = instance.get('ssh_port', '')
                        if ssh_host and ssh_port:
                            return True
                    elif status in ['error', 'terminated', 'failed']:
                        print(f"Instance {instance_id} failed with status: {status}")
                        return False
                except json.JSONDecodeError:
                    pass

            print(f"Waiting for instance {instance_id}... (status check)")
            time.sleep(15)

        print(f"Timeout waiting for instance {instance_id}")
        return False

    def get_instance_ssh(self, instance_id: int) -> Tuple[str, int]:
        """Get SSH host and port for instance (using direct connection)"""
        cmd = [self.vastai, "show", "instance", str(instance_id), "--raw"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            try:
                instance = json.loads(result.stdout)
                if isinstance(instance, list) and len(instance) > 0:
                    instance = instance[0]

                # Try direct connection first (public IP + direct port)
                public_ip = instance.get('public_ipaddr', '')
                ports = instance.get('ports', {})
                ssh_ports = ports.get('22/tcp', [])

                if public_ip and ssh_ports:
                    # Get the direct SSH port
                    direct_port = int(ssh_ports[0].get('HostPort', 22))
                    return public_ip, direct_port

                # Fallback to proxy SSH
                return instance.get('ssh_host', ''), instance.get('ssh_port', 22)
            except json.JSONDecodeError:
                pass

        return '', 0

    def attach_ssh_key(self, instance_id: int, ssh_key_path: str = "~/.ssh/id_rsa.pub") -> bool:
        """Attach SSH key to instance"""
        key_path = os.path.expanduser(ssh_key_path)
        if not os.path.exists(key_path):
            print(f"SSH key not found: {key_path}")
            return False

        with open(key_path, 'r') as f:
            ssh_key = f.read().strip()

        cmd = [self.vastai, "attach", "ssh", str(instance_id), ssh_key]
        result = subprocess.run(cmd, capture_output=True, text=True)

        return result.returncode == 0

    def setup_vllm(self, ssh_host: str, ssh_port: int, hf_token: str,
                   model: str = "google/medgemma-4b-it", port: int = 8000) -> bool:
        """Setup vLLM on remote instance via SSH"""
        setup_commands = f"""
        # Install vLLM
        pip install -q vllm

        # Login to HuggingFace
        export HF_TOKEN='{hf_token}'
        huggingface-cli login --token $HF_TOKEN

        # Start vLLM server in background
        nohup python -m vllm.entrypoints.openai.api_server \\
            --model {model} \\
            --host 0.0.0.0 \\
            --port {port} \\
            --trust-remote-code \\
            --max-model-len 4096 \\
            --gpu-memory-utilization 0.9 \\
            > /workspace/vllm.log 2>&1 &

        echo "vLLM started on port {port}"
        """

        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-p", str(ssh_port),
            f"root@{ssh_host}",
            "bash", "-c", f"'{setup_commands}'"
        ]

        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0

    def wait_for_vllm(self, ssh_host: str, ssh_port: int, vllm_port: int = 8000,
                      timeout: int = 600) -> bool:
        """Wait for vLLM to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            check_cmd = f"curl -s http://localhost:{vllm_port}/health"
            ssh_cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-p", str(ssh_port),
                f"root@{ssh_host}",
                check_cmd
            ]

            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return True

            print(f"Waiting for vLLM on {ssh_host}...")
            time.sleep(30)

        return False

    def create_ssh_tunnel(self, ssh_host: str, ssh_port: int,
                          local_port: int, remote_port: int = 8000) -> subprocess.Popen:
        """Create SSH tunnel to vLLM"""
        tunnel_cmd = [
            "ssh", "-N", "-L", f"{local_port}:localhost:{remote_port}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-p", str(ssh_port),
            f"root@{ssh_host}"
        ]

        proc = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)  # Give tunnel time to establish
        return proc

    def destroy_instance(self, instance_id: int) -> bool:
        """Terminate an instance"""
        cmd = [self.vastai, "destroy", "instance", str(instance_id)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def cleanup(self):
        """Clean up all resources"""
        # Kill SSH tunnels
        for proc in self.ssh_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

        # Kill inference processes
        for proc in self.inference_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

        # Destroy instances
        for instance in self.instances:
            self.destroy_instance(instance['id'])


def calculate_shards(start_index: int, total_images: int, num_machines: int) -> List[Tuple[int, int]]:
    """Calculate shard ranges for each machine"""
    remaining = total_images - start_index
    shard_size = remaining // num_machines
    extra = remaining % num_machines

    shards = []
    current = start_index

    for i in range(num_machines):
        size = shard_size + (1 if i < extra else 0)
        shards.append((current, current + size))
        current += size

    return shards


def run_inference(python_path: str, script_path: str, start_idx: int, end_idx: int,
                  local_port: int, output_log: str) -> subprocess.Popen:
    """Launch inference process for a shard"""
    cmd = [
        python_path, script_path,
        "--provider", "vllm",
        "--model", "google/medgemma-4b-it",
        "--vllm-url", f"http://localhost:{local_port}/v1",
        "--start-index", str(start_idx),
        "--end-index", str(end_idx),
        "--batch-size", "1"
    ]

    with open(output_log, 'w') as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    return proc


def merge_checkpoints(results_dir: Path, output_file: Path, main_checkpoint: Optional[Path] = None):
    """Merge all shard checkpoints into one"""
    merged = {
        "completed_images": {},
        "model": "google/medgemma-4b-it",
        "provider": "vllm"
    }

    # Load main checkpoint if exists
    if main_checkpoint and main_checkpoint.exists():
        with open(main_checkpoint, 'r') as f:
            main_data = json.load(f)
            merged["completed_images"].update(main_data.get("completed_images", {}))

    # Load all shard checkpoints
    for shard_file in results_dir.glob("checkpoint_vllm_google_medgemma-4b-it_shard*.json"):
        with open(shard_file, 'r') as f:
            shard_data = json.load(f)
            merged["completed_images"].update(shard_data.get("completed_images", {}))

    # Save merged
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

    return len(merged["completed_images"])


def main():
    parser = argparse.ArgumentParser(description="Parallel MedGemma inference on Vast.ai")
    parser.add_argument("--num-machines", type=int, default=6, help="Number of machines to rent")
    parser.add_argument("--start-from", type=int, default=7830, help="Starting image index")
    parser.add_argument("--total-images", type=int, default=19019, help="Total number of images")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--vastai-path", type=str, default="vastai", help="Path to vastai CLI")
    parser.add_argument("--gpu-name", type=str, default="RTX 5090", help="GPU type to search for")
    parser.add_argument("--max-price", type=float, default=0.50, help="Max price per hour")
    parser.add_argument("--base-port", type=int, default=8001, help="Base local port for tunnels")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing checkpoints")
    args = parser.parse_args()

    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token and not args.dry_run and not args.merge_only:
        print("Error: HuggingFace token required. Set HF_TOKEN or use --hf-token")
        sys.exit(1)

    # Setup paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    test_script = script_dir / "test_api_vlm.py"
    python_path = sys.executable

    # Calculate shards
    shards = calculate_shards(args.start_from, args.total_images, args.num_machines)

    print("\n=== Parallel MedGemma-4B Inference Plan ===")
    print(f"Total images: {args.total_images}")
    print(f"Starting from: {args.start_from}")
    print(f"Remaining: {args.total_images - args.start_from}")
    print(f"Machines: {args.num_machines}")
    print(f"GPU type: {args.gpu_name}")
    print(f"Max price: ${args.max_price}/hr")
    print("\nShard distribution:")
    for i, (start, end) in enumerate(shards):
        print(f"  Machine {i+1}: images {start} - {end} ({end - start} images)")

    if args.dry_run:
        print("\n[DRY RUN] Would execute the above plan")
        return

    if args.merge_only:
        print("\nMerging existing checkpoints...")
        main_ckpt = results_dir / "checkpoint_vllm_google_medgemma-4b-it.json"
        output_ckpt = results_dir / "checkpoint_vllm_google_medgemma-4b-it_merged.json"
        count = merge_checkpoints(results_dir, output_ckpt, main_ckpt)
        print(f"Merged {count} images to {output_ckpt}")
        return

    # Initialize manager
    manager = VastManager(args.vastai_path)

    try:
        # Step 1: Search for offers
        print("\n[1/7] Searching for GPU offers...")
        offers = manager.search_offers(args.gpu_name, max_price=args.max_price)

        if len(offers) < args.num_machines:
            print(f"Error: Only found {len(offers)} offers, need {args.num_machines}")
            print("Try increasing --max-price or reducing --num-machines")
            sys.exit(1)

        print(f"Found {len(offers)} offers. Using cheapest {args.num_machines}:")
        for i, offer in enumerate(offers[:args.num_machines]):
            print(f"  {i+1}. ID {offer['id']}: ${offer['dph_total']:.3f}/hr, "
                  f"{offer.get('gpu_name', 'GPU')}, {offer.get('gpu_ram', '?')}GB")

        # Step 2: Create instances
        print(f"\n[2/7] Creating {args.num_machines} instances...")
        for i, offer in enumerate(offers[:args.num_machines]):
            instance_id = manager.create_instance(offer['id'])
            if instance_id:
                manager.instances.append({
                    'id': instance_id,
                    'offer_id': offer['id'],
                    'price': offer['dph_total'],
                    'shard': shards[i]
                })
                print(f"  Created instance {instance_id} for shard {i+1}")
            else:
                print(f"  Failed to create instance for offer {offer['id']}")

        if len(manager.instances) < args.num_machines:
            print(f"Warning: Only created {len(manager.instances)} of {args.num_machines} instances")

        # Step 3: Wait for instances
        print("\n[3/7] Waiting for instances to be ready...")
        ready_instances = []
        for instance in manager.instances:
            print(f"  Waiting for instance {instance['id']}...")
            if manager.wait_for_instance(instance['id']):
                # Attach SSH key
                manager.attach_ssh_key(instance['id'])
                ssh_host, ssh_port = manager.get_instance_ssh(instance['id'])
                instance['ssh_host'] = ssh_host
                instance['ssh_port'] = ssh_port
                ready_instances.append(instance)
                print(f"  Instance {instance['id']} ready: {ssh_host}:{ssh_port}")
            else:
                print(f"  Instance {instance['id']} failed")

        manager.instances = ready_instances

        if len(manager.instances) == 0:
            print("Error: No instances ready")
            sys.exit(1)

        # Recalculate shards if fewer instances
        if len(manager.instances) < args.num_machines:
            shards = calculate_shards(args.start_from, args.total_images, len(manager.instances))
            for i, instance in enumerate(manager.instances):
                instance['shard'] = shards[i]

        # Step 4: Setup vLLM on each
        print("\n[4/7] Setting up vLLM on each instance...")
        for instance in manager.instances:
            print(f"  Setting up vLLM on {instance['id']}...")
            manager.setup_vllm(instance['ssh_host'], instance['ssh_port'], hf_token)

        # Step 5: Wait for vLLM servers
        print("\n[5/7] Waiting for vLLM servers to start...")
        for instance in manager.instances:
            print(f"  Waiting for vLLM on {instance['id']}...")
            if not manager.wait_for_vllm(instance['ssh_host'], instance['ssh_port']):
                print(f"  Warning: vLLM may not be ready on {instance['id']}")

        # Step 6: Create SSH tunnels and launch inference
        print("\n[6/7] Creating tunnels and launching inference...")
        for i, instance in enumerate(manager.instances):
            local_port = args.base_port + i

            # Create tunnel
            tunnel = manager.create_ssh_tunnel(
                instance['ssh_host'], instance['ssh_port'], local_port
            )
            manager.ssh_processes.append(tunnel)
            instance['local_port'] = local_port

            # Launch inference
            start_idx, end_idx = instance['shard']
            log_file = results_dir / f"inference_shard{start_idx}-{end_idx}.log"

            proc = run_inference(
                python_path, str(test_script),
                start_idx, end_idx, local_port, str(log_file)
            )
            manager.inference_processes.append(proc)

            print(f"  Instance {instance['id']}: port {local_port}, "
                  f"images {start_idx}-{end_idx}, log: {log_file.name}")

        # Step 7: Monitor progress
        print("\n[7/7] Monitoring progress... (Ctrl+C to abort)")
        print("=" * 60)

        all_done = False
        while not all_done:
            all_done = True
            status_lines = []

            for i, proc in enumerate(manager.inference_processes):
                if proc.poll() is None:
                    all_done = False
                    status = "running"
                else:
                    status = f"done (exit {proc.returncode})"

                instance = manager.instances[i]
                start_idx, end_idx = instance['shard']

                # Check checkpoint progress
                ckpt_file = results_dir / f"checkpoint_vllm_google_medgemma-4b-it_shard{start_idx}-{end_idx}.json"
                progress = 0
                if ckpt_file.exists():
                    try:
                        with open(ckpt_file, 'r') as f:
                            ckpt = json.load(f)
                            progress = len(ckpt.get("completed_images", {}))
                    except Exception:
                        pass

                total = end_idx - start_idx
                pct = (progress / total * 100) if total > 0 else 0
                status_lines.append(f"  Shard {i+1}: {progress}/{total} ({pct:.1f}%) - {status}")

            print(f"\r{' ' * 80}\r", end="")  # Clear line
            print("\n".join(status_lines))
            print(f"\nPress Ctrl+C to abort and clean up")

            if not all_done:
                time.sleep(60)
                # Move cursor up for refresh
                print(f"\033[{len(status_lines) + 2}A", end="")

        print("\n\nAll inference processes completed!")

        # Merge results
        print("\nMerging checkpoints...")
        main_ckpt = results_dir / "checkpoint_vllm_google_medgemma-4b-it.json"
        output_ckpt = results_dir / "checkpoint_vllm_google_medgemma-4b-it_merged.json"
        count = merge_checkpoints(results_dir, output_ckpt, main_ckpt)
        print(f"Merged {count} total images to {output_ckpt}")

    except KeyboardInterrupt:
        print("\n\nAborted by user. Cleaning up...")

    finally:
        # Cleanup
        print("\nCleaning up resources...")
        manager.cleanup()
        print("Done.")


if __name__ == "__main__":
    main()
