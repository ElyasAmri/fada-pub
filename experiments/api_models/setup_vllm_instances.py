#!/usr/bin/env python3
"""Setup vLLM on running Vast.ai instances and start parallel inference"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_hf_token():
    """Get HuggingFace token from cache"""
    cache_path = Path.home() / ".cache" / "huggingface" / "token"
    if cache_path.exists():
        return cache_path.read_text().strip()
    return os.environ.get("HF_TOKEN", "")


def get_instance_ssh(vastai_path: str, instance_id: int):
    """Get direct SSH connection info"""
    cmd = [vastai_path, "show", "instance", str(instance_id), "--raw"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        instance = json.loads(result.stdout)
        if isinstance(instance, list) and len(instance) > 0:
            instance = instance[0]

        public_ip = instance.get('public_ipaddr', '')
        ports = instance.get('ports', {})
        ssh_ports = ports.get('22/tcp', [])

        if public_ip and ssh_ports:
            direct_port = int(ssh_ports[0].get('HostPort', 22))
            return public_ip, direct_port

    return None, None


def run_ssh_command(host: str, port: int, command: str, timeout: int = 300):
    """Run command via SSH"""
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", f"ConnectTimeout=30",
        "-p", str(port),
        f"root@{host}",
        command
    ]

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"


def setup_vllm_on_instance(host: str, port: int, hf_token: str, instance_id: int):
    """Setup vLLM on a single instance"""
    print(f"  [{instance_id}] Installing vLLM...")

    # Install vLLM
    success, out, err = run_ssh_command(host, port, "pip install -q vllm huggingface_hub", timeout=300)
    if not success:
        print(f"  [{instance_id}] Failed to install vLLM: {err[:100]}")
        return False

    # Login to HuggingFace
    print(f"  [{instance_id}] Logging into HuggingFace...")
    login_cmd = f"huggingface-cli login --token {hf_token}"
    success, out, err = run_ssh_command(host, port, login_cmd, timeout=60)

    # Start vLLM server
    print(f"  [{instance_id}] Starting vLLM server...")
    start_cmd = """nohup python -m vllm.entrypoints.openai.api_server \
        --model google/medgemma-4b-it \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.9 \
        > /workspace/vllm.log 2>&1 &"""

    success, out, err = run_ssh_command(host, port, start_cmd, timeout=60)
    print(f"  [{instance_id}] vLLM server started")
    return True


def wait_for_vllm(host: str, port: int, instance_id: int, timeout: int = 600):
    """Wait for vLLM to be ready"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        success, out, err = run_ssh_command(host, port, "curl -s http://localhost:8000/health", timeout=10)
        if success and out.strip():
            return True
        print(f"  [{instance_id}] Waiting for vLLM...")
        time.sleep(30)

    return False


def create_ssh_tunnel(host: str, ssh_port: int, local_port: int, remote_port: int = 8000):
    """Create SSH tunnel"""
    tunnel_cmd = [
        "ssh", "-N", "-L", f"{local_port}:localhost:{remote_port}",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(ssh_port),
        f"root@{host}"
    ]

    proc = subprocess.Popen(tunnel_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)
    return proc


def main():
    # Configuration
    vastai_path = "./venv/Scripts/vastai.exe"
    instance_ids = [29252197, 29252198, 29252199, 29252200, 29252201, 29252202]

    # Shards based on current progress
    shards = [
        (7870, 9729),    # 1859 images
        (9729, 11587),   # 1858 images
        (11587, 13445),  # 1858 images
        (13445, 15303),  # 1858 images
        (15303, 17161),  # 1858 images
        (17161, 19019),  # 1858 images
    ]

    hf_token = get_hf_token()
    if not hf_token:
        print("Error: HF_TOKEN not found")
        sys.exit(1)

    print(f"HF Token found: {hf_token[:8]}...")
    print(f"\n=== Setting up {len(instance_ids)} instances ===\n")

    # Get SSH info for all instances
    instances = []
    for i, instance_id in enumerate(instance_ids):
        host, port = get_instance_ssh(vastai_path, instance_id)
        if host and port:
            instances.append({
                'id': instance_id,
                'host': host,
                'port': port,
                'shard': shards[i],
                'local_port': 8001 + i
            })
            print(f"Instance {instance_id}: {host}:{port}")
        else:
            print(f"Instance {instance_id}: Failed to get SSH info")

    if not instances:
        print("No instances available")
        sys.exit(1)

    # Setup vLLM on each instance in parallel
    print(f"\n=== Setting up vLLM on {len(instances)} instances ===\n")

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(setup_vllm_on_instance, inst['host'], inst['port'], hf_token, inst['id']): inst
            for inst in instances
        }
        for future in as_completed(futures):
            inst = futures[future]
            try:
                success = future.result()
                if success:
                    print(f"  [{inst['id']}] Setup complete")
            except Exception as e:
                print(f"  [{inst['id']}] Setup failed: {e}")

    # Wait for vLLM servers
    print(f"\n=== Waiting for vLLM servers ===\n")

    for inst in instances:
        print(f"Waiting for vLLM on {inst['id']}...")
        if wait_for_vllm(inst['host'], inst['port'], inst['id'], timeout=600):
            print(f"  [{inst['id']}] vLLM ready!")
        else:
            print(f"  [{inst['id']}] vLLM not ready (timeout)")

    # Create SSH tunnels
    print(f"\n=== Creating SSH tunnels ===\n")

    tunnels = []
    for inst in instances:
        tunnel = create_ssh_tunnel(inst['host'], inst['port'], inst['local_port'])
        tunnels.append(tunnel)
        print(f"Tunnel: localhost:{inst['local_port']} -> {inst['host']}:8000")

    # Launch inference processes
    print(f"\n=== Launching inference processes ===\n")

    script_dir = Path(__file__).parent
    test_script = script_dir / "test_api_vlm.py"
    python_path = Path(__file__).parent.parent.parent / "venv" / "Scripts" / "python.exe"

    inference_procs = []
    for inst in instances:
        start_idx, end_idx = inst['shard']
        log_file = script_dir / "results" / f"inference_shard{start_idx}-{end_idx}.log"

        cmd = [
            str(python_path), str(test_script),
            "--provider", "vllm",
            "--model", "google/medgemma-4b-it",
            "--vllm-url", f"http://localhost:{inst['local_port']}/v1",
            "--start-index", str(start_idx),
            "--end-index", str(end_idx),
            "--batch-size", "1"
        ]

        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            inference_procs.append((proc, inst, log_file))

        print(f"Launched: images {start_idx}-{end_idx} on port {inst['local_port']}")

    print(f"\n=== All {len(inference_procs)} inference processes launched ===")
    print("Monitor progress with: tail -f experiments/api_models/results/inference_shard*.log")
    print("Press Ctrl+C to stop all processes\n")

    try:
        # Monitor progress
        while True:
            all_done = True
            for proc, inst, log_file in inference_procs:
                if proc.poll() is None:
                    all_done = False

            if all_done:
                print("\nAll inference processes completed!")
                break

            time.sleep(60)

    except KeyboardInterrupt:
        print("\nStopping all processes...")

    finally:
        # Cleanup
        for proc, _, _ in inference_procs:
            proc.terminate()
        for tunnel in tunnels:
            tunnel.terminate()


if __name__ == "__main__":
    main()
