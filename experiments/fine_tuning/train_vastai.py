#!/usr/bin/env python3
"""
Vast.ai VLM Fine-tuning Orchestrator

Automates the complete fine-tuning workflow on Vast.ai:
1. Search for and rent GPU instance (A100/H100)
2. Upload training data
3. Setup environment and run training
4. Monitor progress
5. Download results
6. Terminate instance

Usage:
    # Full automated workflow
    python train_vastai.py --model qwen3-vl-8b --epochs 2

    # Just search for available GPUs
    python train_vastai.py --search-only

    # Connect to existing instance
    python train_vastai.py --instance-id 12345678 --upload-and-train

    # Download results from existing instance
    python train_vastai.py --instance-id 12345678 --download-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "vlm_training"
SETUP_SCRIPT = SCRIPT_DIR / "vastai_train_setup.sh"


class VastaiTrainer:
    """Manage Vast.ai instance for VLM fine-tuning"""

    def __init__(self, vastai_path: str = "vastai"):
        self.vastai = vastai_path
        self.instance_id: Optional[int] = None
        self.ssh_host: Optional[str] = None
        self.ssh_port: Optional[int] = None

    def search_gpus(
        self,
        gpu_name: str = "A100",
        min_vram: int = 40,
        max_price: float = 2.0,
        min_reliability: float = 0.95,
    ) -> List[Dict]:
        """Search for available GPU offers"""
        # Build query
        query_parts = [
            f"gpu_ram>={min_vram}",
            f"dph<{max_price}",
            f"reliability>{min_reliability}",
            "inet_down>100",
            "num_gpus=1",
        ]

        # Add GPU name filter
        if gpu_name:
            query_parts.append(f"gpu_name={gpu_name.replace(' ', '_')}")

        query = " ".join(query_parts)

        cmd = [self.vastai, "search", "offers", query, "--raw"]
        print(f"Searching: {' '.join(cmd)}")

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

    def display_offers(self, offers: List[Dict], limit: int = 10) -> None:
        """Display available offers in a table"""
        print(f"\n{'='*80}")
        print(f"{'ID':>10} | {'GPU':^20} | {'VRAM':^8} | {'$/hr':^8} | {'DL':^8} | {'Rel':^6}")
        print(f"{'='*80}")

        for offer in offers[:limit]:
            gpu = offer.get('gpu_name', 'Unknown')[:20]
            vram = offer.get('gpu_ram', 0)
            price = offer.get('dph_total', 0)
            dl_speed = offer.get('inet_down', 0)
            reliability = offer.get('reliability', 0)

            print(f"{offer['id']:>10} | {gpu:^20} | {vram:^8.0f} | ${price:^7.3f} | {dl_speed:^8.0f} | {reliability:^6.2f}")

        print(f"{'='*80}")
        print(f"Showing {min(limit, len(offers))} of {len(offers)} offers")

    def create_instance(
        self,
        offer_id: int,
        disk_gb: int = 100,
        image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
    ) -> Optional[int]:
        """Create a Vast.ai instance"""
        cmd = [
            self.vastai, "create", "instance", str(offer_id),
            "--image", image,
            "--disk", str(disk_gb),
            "--ssh", "--direct",
        ]

        print(f"Creating instance: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error creating instance: {result.stderr}")
            return None

        # Parse instance ID
        try:
            match = re.search(r"'new_contract':\s*(\d+)", result.stdout)
            if match:
                return int(match.group(1))
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
            info = self.get_instance_info(instance_id)
            if info:
                status = info.get('actual_status', '')
                if status == 'running':
                    self.ssh_host = info.get('public_ipaddr', '')
                    ports = info.get('ports', {})
                    ssh_ports = ports.get('22/tcp', [])
                    if ssh_ports:
                        self.ssh_port = int(ssh_ports[0].get('HostPort', 22))
                    else:
                        self.ssh_port = info.get('ssh_port', 22)

                    if self.ssh_host and self.ssh_port:
                        return True

                elif status in ['error', 'terminated', 'failed']:
                    print(f"Instance {instance_id} failed: {status}")
                    return False

            print(f"Waiting for instance {instance_id}... (status: {status if info else 'unknown'})")
            time.sleep(15)

        print(f"Timeout waiting for instance {instance_id}")
        return False

    def get_instance_info(self, instance_id: int) -> Optional[Dict]:
        """Get instance info"""
        cmd = [self.vastai, "show", "instance", str(instance_id), "--raw"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                return data
            except json.JSONDecodeError:
                pass
        return None

    def run_ssh_command(
        self,
        command: str,
        timeout: int = 300,
        capture: bool = True,
    ) -> Tuple[int, str, str]:
        """Run command on remote instance via SSH"""
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-p", str(self.ssh_port),
            f"root@{self.ssh_host}",
            command,
        ]

        if capture:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(ssh_cmd, timeout=timeout)
            return result.returncode, "", ""

    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload file via SCP"""
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.ssh_port),
            str(local_path),
            f"root@{self.ssh_host}:{remote_path}",
        ]

        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        return result.returncode == 0

    def upload_directory(self, local_dir: Path, remote_dir: str) -> bool:
        """Upload directory via SCP"""
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.ssh_port),
            "-r",
            str(local_dir),
            f"root@{self.ssh_host}:{remote_dir}",
        ]

        print(f"Uploading {local_dir} to {remote_dir}...")
        result = subprocess.run(scp_cmd)
        return result.returncode == 0

    def download_directory(self, remote_dir: str, local_dir: Path) -> bool:
        """Download directory via SCP"""
        local_dir.mkdir(parents=True, exist_ok=True)

        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.ssh_port),
            "-r",
            f"root@{self.ssh_host}:{remote_dir}",
            str(local_dir),
        ]

        print(f"Downloading {remote_dir} to {local_dir}...")
        result = subprocess.run(scp_cmd)
        return result.returncode == 0

    def setup_environment(self) -> bool:
        """Setup training environment on remote"""
        print("\nSetting up training environment...")

        # Install dependencies
        setup_cmds = """
        pip install -q --upgrade pip
        pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
        pip install -q accelerate bitsandbytes peft trl datasets pillow pandas
        pip install -q git+https://github.com/huggingface/transformers.git
        mkdir -p /workspace/fada-finetune/data
        mkdir -p /workspace/fada-finetune/output
        """

        ret, stdout, stderr = self.run_ssh_command(setup_cmds, timeout=600)
        if ret != 0:
            print(f"Setup failed: {stderr}")
            return False

        print("Environment setup complete.")
        return True

    def upload_training_data(self, data_dir: Path) -> bool:
        """Upload training data to remote"""
        print(f"\nUploading training data from {data_dir}...")

        # Upload JSONL files
        for jsonl_file in data_dir.glob("*.jsonl"):
            print(f"  Uploading {jsonl_file.name}...")
            if not self.upload_file(jsonl_file, f"/workspace/fada-finetune/data/{jsonl_file.name}"):
                print(f"  Failed to upload {jsonl_file.name}")
                return False

        print("Training data uploaded.")
        return True

    def upload_training_script(self) -> bool:
        """Upload the training script"""
        # Create training script content
        train_script = SCRIPT_DIR / "train_qwen3vl_lora.py"
        if train_script.exists():
            return self.upload_file(train_script, "/workspace/fada-finetune/train.py")
        return False

    def start_training(
        self,
        model: str = "qwen3-vl-8b",
        epochs: int = 2,
        batch_size: int = 2,
        grad_accum: int = 4,
        learning_rate: float = 2e-4,
        lora_r: int = 32,
        lora_alpha: int = 64,
        use_4bit: bool = False,
        use_screen: bool = True,
    ) -> bool:
        """Start training on remote"""
        print(f"\nStarting training: {model}, {epochs} epochs...")

        train_cmd = f"""
        cd /workspace/fada-finetune && python train.py \\
            --model {model} \\
            --train-data data/gt_train.jsonl \\
            --val-data data/gt_val.jsonl \\
            --epochs {epochs} \\
            --batch-size {batch_size} \\
            --gradient-accumulation {grad_accum} \\
            --learning-rate {learning_rate} \\
            --lora-r {lora_r} \\
            --lora-alpha {lora_alpha} \\
            {'--use-4bit' if use_4bit else ''} \\
            --output-dir ./output
        """

        if use_screen:
            # Run in screen session for persistence
            screen_cmd = f'screen -dmS training bash -c "{train_cmd}; exec bash"'
            ret, _, stderr = self.run_ssh_command(screen_cmd, timeout=60)
            if ret == 0:
                print("Training started in screen session 'training'")
                print("  To attach: screen -r training")
                print("  To detach: Ctrl+A, D")
                return True
            else:
                print(f"Failed to start screen session: {stderr}")
                return False
        else:
            # Run directly (will disconnect if SSH drops)
            ret, _, _ = self.run_ssh_command(train_cmd, timeout=86400, capture=False)
            return ret == 0

    def check_training_status(self) -> Dict:
        """Check training status"""
        # Check if training is running
        ret, stdout, _ = self.run_ssh_command("pgrep -f 'python.*train.py'", timeout=30)
        is_running = ret == 0

        # Get latest log output
        ret, log_output, _ = self.run_ssh_command(
            "tail -20 /workspace/fada-finetune/output/*/trainer_state.json 2>/dev/null || echo 'No logs yet'",
            timeout=30,
        )

        # Check for output directories
        ret, dirs, _ = self.run_ssh_command(
            "ls -la /workspace/fada-finetune/output/ 2>/dev/null || echo 'No output yet'",
            timeout=30,
        )

        return {
            "is_running": is_running,
            "log_output": log_output,
            "output_dirs": dirs,
        }

    def destroy_instance(self, instance_id: int) -> bool:
        """Terminate instance"""
        cmd = [self.vastai, "destroy", "instance", str(instance_id)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Vast.ai VLM Fine-tuning Orchestrator")

    # Instance selection
    parser.add_argument("--instance-id", type=int, help="Use existing instance ID")
    parser.add_argument("--gpu", type=str, default="A100", help="GPU type to search (A100, H100)")
    parser.add_argument("--min-vram", type=int, default=40, help="Minimum VRAM in GB")
    parser.add_argument("--max-price", type=float, default=2.0, help="Maximum price $/hr")

    # Training config
    parser.add_argument("--model", type=str, default="qwen3-vl-8b",
                        choices=["qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b",
                                 "qwen2.5-vl-3b", "qwen2.5-vl-7b", "qwen2-vl-2b", "qwen2-vl-7b"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--use-4bit", action="store_true")

    # Data paths
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))

    # Workflow options
    parser.add_argument("--search-only", action="store_true", help="Only search and display offers")
    parser.add_argument("--upload-only", action="store_true", help="Only upload data to existing instance")
    parser.add_argument("--train-only", action="store_true", help="Only start training on existing instance")
    parser.add_argument("--status", action="store_true", help="Check training status")
    parser.add_argument("--download-only", action="store_true", help="Only download results")
    parser.add_argument("--destroy", action="store_true", help="Destroy instance after completion")

    # Other
    parser.add_argument("--vastai-path", type=str, default="vastai")
    parser.add_argument("--disk", type=int, default=100, help="Disk size in GB")

    args = parser.parse_args()

    trainer = VastaiTrainer(args.vastai_path)
    data_dir = Path(args.data_dir)

    # Search only mode
    if args.search_only:
        print(f"Searching for {args.gpu} GPUs (min {args.min_vram}GB, max ${args.max_price}/hr)...")
        offers = trainer.search_gpus(args.gpu, args.min_vram, args.max_price)
        trainer.display_offers(offers)
        return

    # If instance ID provided, use it
    if args.instance_id:
        trainer.instance_id = args.instance_id
        info = trainer.get_instance_info(args.instance_id)
        if info:
            trainer.ssh_host = info.get('public_ipaddr', '')
            ports = info.get('ports', {})
            ssh_ports = ports.get('22/tcp', [])
            if ssh_ports:
                trainer.ssh_port = int(ssh_ports[0].get('HostPort', 22))
            print(f"Using instance {args.instance_id}: {trainer.ssh_host}:{trainer.ssh_port}")
        else:
            print(f"Could not find instance {args.instance_id}")
            return

    # Status check mode
    if args.status:
        if not trainer.ssh_host:
            print("Error: Need --instance-id to check status")
            return
        status = trainer.check_training_status()
        print(f"\nTraining running: {status['is_running']}")
        print(f"\nOutput directories:\n{status['output_dirs']}")
        print(f"\nRecent logs:\n{status['log_output']}")
        return

    # Download only mode
    if args.download_only:
        if not trainer.ssh_host:
            print("Error: Need --instance-id to download")
            return
        output_dir = SCRIPT_DIR / "vastai_results"
        trainer.download_directory("/workspace/fada-finetune/output", output_dir)
        print(f"\nResults downloaded to: {output_dir}")
        return

    # Full workflow or specific steps
    try:
        # Step 1: Get or create instance
        if not trainer.instance_id:
            print(f"\n[1/6] Searching for {args.gpu} GPUs...")
            offers = trainer.search_gpus(args.gpu, args.min_vram, args.max_price)

            if not offers:
                print("No suitable offers found. Try adjusting --max-price or --gpu")
                return

            trainer.display_offers(offers, limit=5)

            # Select cheapest offer
            offer = offers[0]
            print(f"\nSelected offer {offer['id']}: {offer['gpu_name']}, ${offer['dph_total']:.3f}/hr")

            confirm = input("Create instance? [y/N]: ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return

            print(f"\n[2/6] Creating instance...")
            trainer.instance_id = trainer.create_instance(offer['id'], disk_gb=args.disk)

            if not trainer.instance_id:
                print("Failed to create instance")
                return

            print(f"Instance created: {trainer.instance_id}")

            print(f"\n[3/6] Waiting for instance to be ready...")
            if not trainer.wait_for_instance(trainer.instance_id):
                print("Instance failed to start")
                return

            print(f"Instance ready: {trainer.ssh_host}:{trainer.ssh_port}")

        # Step 2: Setup environment
        if not args.train_only:
            print(f"\n[4/6] Setting up environment...")
            if not trainer.setup_environment():
                print("Environment setup failed")
                return

            # Upload training script
            print("Uploading training script...")
            trainer.upload_training_script()

        # Step 3: Upload data
        if not args.train_only:
            print(f"\n[5/6] Uploading training data...")
            if not trainer.upload_training_data(data_dir):
                print("Data upload failed")
                return

        if args.upload_only:
            print("\nData uploaded. Run with --train-only to start training.")
            print(f"  SSH: ssh -p {trainer.ssh_port} root@{trainer.ssh_host}")
            return

        # Step 4: Start training
        print(f"\n[6/6] Starting training...")
        if not trainer.start_training(
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            use_4bit=args.use_4bit,
        ):
            print("Failed to start training")
            return

        print("\n" + "="*60)
        print("  Training Started!")
        print("="*60)
        print(f"\nInstance ID: {trainer.instance_id}")
        print(f"SSH: ssh -p {trainer.ssh_port} root@{trainer.ssh_host}")
        print(f"\nMonitor training:")
        print(f"  python train_vastai.py --instance-id {trainer.instance_id} --status")
        print(f"\nDownload results when complete:")
        print(f"  python train_vastai.py --instance-id {trainer.instance_id} --download-only")
        print(f"\nDestroy instance:")
        print(f"  python train_vastai.py --instance-id {trainer.instance_id} --destroy")

    except KeyboardInterrupt:
        print("\n\nAborted by user.")

    finally:
        if args.destroy and trainer.instance_id:
            print(f"\nDestroying instance {trainer.instance_id}...")
            trainer.destroy_instance(trainer.instance_id)


if __name__ == "__main__":
    main()
