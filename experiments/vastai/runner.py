#!/usr/bin/env python3
"""
Unified Vast.ai CLI for VLM Testing and Fine-Tuning.

Usage:
    python -m experiments.vastai status                    # List all jobs
    python -m experiments.vastai test InternVL3-2B         # Quick test (20 samples)
    python -m experiments.vastai test InternVL3-2B --full  # Full evaluation (600 samples)
    python -m experiments.vastai batch-test A B C          # Test multiple models
    python -m experiments.vastai finetune InternVL3-2B     # Fine-tune a model
    python -m experiments.vastai finetune InternVL3-2B --epochs 3 --auto-destroy
    python -m experiments.vastai destroy job-abc123        # Destroy instance
    python -m experiments.vastai destroy --all             # Destroy all instances
    python -m experiments.vastai models                    # List supported models
    python -m experiments.vastai presets                   # List GPU presets
"""

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from .jobs import JobDatabase, JobStatus, JobType, Job, print_jobs_table
from .instance import VastInstance
from .presets import (
    PRESETS, MODEL_CONFIGS, resolve_model_id, get_preset_for_model,
    print_presets, print_model_list
)
from .transfer import (
    DataTransfer, setup_environment, setup_environment_with_retry,
    check_gpu, validate_environment, REMOTE_WORKSPACE
)


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "vastai"
TEST_SUBSET = PROJECT_ROOT / "outputs" / "evaluation" / "test_subset.jsonl"
IMAGES_DIR = DATA_DIR / "Fetal Ultrasound"
TEMPLATES_DIR = Path(__file__).parent / "templates"


class RuntimeWatchdog:
    """
    Background watchdog that destroys instances after max runtime.

    Provides billing protection by ensuring instances don't run forever.
    """

    def __init__(
        self,
        instance: VastInstance,
        db: 'JobDatabase',
        job_id: str,
        max_hours: float,
        check_interval: int = 60,
    ):
        """
        Initialize watchdog.

        Args:
            instance: VastInstance to monitor
            db: Job database for status updates
            job_id: Job ID for status updates
            max_hours: Maximum runtime in hours
            check_interval: Seconds between checks
        """
        self.instance = instance
        self.db = db
        self.job_id = job_id
        self.max_seconds = max_hours * 3600
        self.check_interval = check_interval
        self.start_time = time.time()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        """Start the watchdog thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"Watchdog started: will destroy instance after {self.max_seconds/3600:.1f} hours")

    def stop(self):
        """Stop the watchdog thread."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        """Background thread that monitors runtime."""
        while not self._stop.is_set():
            elapsed = time.time() - self.start_time

            if elapsed >= self.max_seconds:
                print(f"\n[WATCHDOG] Maximum runtime ({self.max_seconds/3600:.1f} hours) exceeded!")
                print(f"[WATCHDOG] Destroying instance {self.instance.instance_id}...")

                # Update job status
                self.db.set_status(
                    self.job_id,
                    JobStatus.CANCELLED,
                    f"Watchdog: max runtime {self.max_seconds/3600:.1f}h exceeded"
                )

                # Destroy instance
                if self.instance.instance_id:
                    self.instance.destroy()

                print("[WATCHDOG] Instance destroyed. Billing protection activated.")
                break

            # Log progress every 30 minutes
            if int(elapsed) % 1800 == 0 and elapsed > 0:
                remaining = (self.max_seconds - elapsed) / 3600
                print(f"[WATCHDOG] Runtime: {elapsed/3600:.1f}h, remaining: {remaining:.1f}h")

            self._stop.wait(self.check_interval)


class VastaiRunner:
    """Main orchestrator for Vast.ai operations."""

    def __init__(self):
        self.db = JobDatabase()
        self.instance = VastInstance()

    # =========================================================================
    # Status Commands
    # =========================================================================

    def cmd_status(self, job_id: str = None):
        """Show job status."""
        if job_id:
            job = self.db.get_job(job_id)
            if not job:
                print(f"Job not found: {job_id}")
                return 1

            print(f"\nJob: {job.job_id}")
            print(f"  Type: {job.job_type}")
            print(f"  Model: {job.model_id}")
            print(f"  Status: {job.status}")
            print(f"  Instance: {job.instance_id or 'None'}")
            if job.ssh_host:
                print(f"  SSH: ssh -p {job.ssh_port} root@{job.ssh_host}")
            print(f"  Created: {job.created_at}")
            if job.output_path:
                print(f"  Output: {job.output_path}")
            if job.error_message:
                print(f"  Error: {job.error_message}")

            # If instance is running, check remote status
            if job.instance_id and job.status == JobStatus.RUNNING.value:
                if self.instance.connect(job.instance_id):
                    if self.instance.is_process_running("run_inference.py"):
                        print(f"  Remote: Inference running")
                    else:
                        print(f"  Remote: No inference process found")
            return 0

        # List all jobs
        jobs = self.db.list_jobs(limit=20)
        print_jobs_table(jobs)

        # Show active instances
        active = self.db.get_active_jobs()
        if active:
            print(f"\nActive instances: {len(active)}")
            for job in active:
                if job.ssh_host:
                    print(f"  {job.job_id}: ssh -p {job.ssh_port} root@{job.ssh_host}")

        return 0

    # =========================================================================
    # Test Command
    # =========================================================================

    def cmd_test(
        self,
        model_id: str,
        samples: int = 20,
        full: bool = False,
        preset: str = None,
        dry_run: bool = False,
        auto_destroy: bool = False,
        yes: bool = False,
        skip_images: bool = False,
        max_hours: float = None,
    ):
        """
        Test a VLM model on the evaluation dataset.

        Args:
            model_id: Model name, alias, or HuggingFace ID
            samples: Number of samples (default 20, ignored if --full)
            full: Run full 600-sample evaluation
            preset: GPU preset override
            max_hours: Maximum runtime hours (billing protection)
            dry_run: Show plan without executing
            auto_destroy: Destroy instance after completion
            yes: Skip confirmation prompts
        """
        # Resolve model ID
        model_id = resolve_model_id(model_id)

        # Get preset
        if preset:
            gpu_preset = PRESETS.get(preset)
            if not gpu_preset:
                print(f"Unknown preset: {preset}")
                print_presets()
                return 1
        else:
            gpu_preset = get_preset_for_model(model_id)

        # Sample count
        if full:
            samples = None  # All samples
            sample_desc = "all (600)"
        else:
            sample_desc = str(samples)

        print("\n" + "=" * 60)
        print("VLM Test Configuration")
        print("=" * 60)
        print(f"Model:    {model_id}")
        print(f"Samples:  {sample_desc}")
        print(f"Preset:   {gpu_preset.name} ({gpu_preset.description})")
        print(f"GPU:      {gpu_preset.gpu_name}")
        print(f"Max $/hr: ${gpu_preset.max_price:.2f}")

        if dry_run:
            print("\n[DRY RUN] Would execute the above configuration")
            return 0

        # Check test data exists
        if not TEST_SUBSET.exists():
            print(f"\nError: Test data not found: {TEST_SUBSET}")
            print("Run: python experiments/evaluation/create_test_subset.py")
            return 1

        # Check images exist
        if not IMAGES_DIR.exists():
            print(f"\nError: Images not found: {IMAGES_DIR}")
            return 1

        # Confirmation
        if not yes:
            confirm = input("\nProceed with test? [y/N]: ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return 0

        # Create job record
        job = self.db.create_job(
            job_type=JobType.TEST,
            model_id=model_id,
            preset=gpu_preset.name,
            samples=samples,
            metadata={"full": full, "auto_destroy": auto_destroy}
        )
        print(f"\nJob created: {job.job_id}")

        try:
            # Step 1: Search for GPU
            print(f"\n[1/5] Searching for {gpu_preset.gpu_name}...")
            offers = self.instance.search_offers(
                gpu_name=gpu_preset.gpu_name,
                min_vram=gpu_preset.min_vram,
                max_price=gpu_preset.max_price,
                min_reliability=0.99,  # Higher reliability to avoid SSH failures
                min_download=500,  # Higher min download for faster uploads
            )

            if not offers:
                print("No suitable GPU offers found. Try:")
                print(f"  - Increase --max-price (current: ${gpu_preset.max_price})")
                print(f"  - Use a different preset")
                self.db.set_status(job.job_id, JobStatus.FAILED, "No GPU offers found")
                return 1

            self.instance.print_offers(offers, limit=5)
            offer = offers[0]
            print(f"\nSelected: {offer.gpu_name} @ ${offer.price_per_hour:.3f}/hr")

            # Step 2: Create instance
            print(f"\n[2/5] Creating instance...")
            if not self.instance.create(offer.offer_id, disk_gb=gpu_preset.disk_gb):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Failed to create instance")
                return 1

            print(f"Instance created: {self.instance.instance_id}")
            self.db.set_status(job.job_id, JobStatus.CREATING)

            # Step 3: Wait for instance
            print(f"\n[3/5] Waiting for instance...")
            if not self.instance.wait_until_ready(timeout=900):  # 15 min for Docker pull
                self.db.set_status(job.job_id, JobStatus.FAILED, "Instance failed to start")
                return 1

            self.db.set_instance(
                job.job_id,
                self.instance.instance_id,
                self.instance.ssh_host,
                self.instance.ssh_port
            )
            print(f"Instance ready: {self.instance.ssh_command}")

            # Brief delay to let SSH stabilize
            print("Waiting for SSH to stabilize...")
            time.sleep(20)

            # Check GPU
            gpu_info = check_gpu(self.instance)
            if gpu_info:
                print(f"GPU: {gpu_info['name']}, VRAM: {gpu_info['memory_total']}")
            else:
                print("Warning: Could not query GPU info")

            # Validate instance meets requirements
            print("Validating instance...")
            validation = self.instance.validate_instance()
            if validation.get("driver_version"):
                print(f"  Driver version: {validation['driver_version']}")
            if validation.get("cuda_version"):
                print(f"  CUDA version: {validation['cuda_version']}")
            if not validation.get("valid", False):
                errors = validation.get("errors", ["Unknown validation error"])
                print(f"VALIDATION FAILED: {errors}")
                self.db.set_status(job.job_id, JobStatus.FAILED, f"Instance validation failed: {errors}")
                if self.instance.instance_id:
                    self.instance.destroy()
                return 1
            print("  Validation passed")

            # Step 4: Setup environment and upload data
            print(f"\n[4/5] Setting up environment...")
            self.db.set_status(job.job_id, JobStatus.UPLOADING)

            if not setup_environment_with_retry(self.instance, max_retries=3):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Environment setup/validation failed after 3 attempts")
                return 1

            transfer = DataTransfer(self.instance)
            if not transfer.setup_workspace():
                self.db.set_status(job.job_id, JobStatus.FAILED, "Workspace setup failed")
                return 1

            print("Uploading data...")
            if not transfer.upload_test_data(TEST_SUBSET):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Failed to upload test data")
                return 1

            if skip_images:
                print("  Skipping image upload (--skip-images)")
            elif not transfer.upload_test_images(TEST_SUBSET):
                # Fall back to full upload if test images fail
                print("  Falling back to full image upload...")
                if not transfer.upload_images(IMAGES_DIR):
                    self.db.set_status(job.job_id, JobStatus.FAILED, "Failed to upload images")
                    return 1

            # Upload inference script
            inference_script = TEMPLATES_DIR / "run_inference.py"
            if not transfer.upload_script(inference_script):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Failed to upload script")
                return 1

            # Step 5: Run inference
            print(f"\n[5/5] Starting inference...")
            self.db.set_status(job.job_id, JobStatus.RUNNING)

            samples_arg = f"--samples {samples}" if samples else ""
            inference_cmd = f"""
            cd {REMOTE_WORKSPACE} && python scripts/run_inference.py \
                --model {model_id} \
                --test-data data/test_subset.jsonl \
                --output outputs \
                {samples_arg}
            """

            # Run in foreground (wait for completion)
            print("Running inference (this may take a while)...")
            ret, stdout, stderr = self.instance.run_ssh(
                inference_cmd,
                timeout=7200,  # 2 hour timeout
                capture=True
            )

            if ret != 0:
                print(f"Inference failed: {stderr}")
                self.db.set_status(job.job_id, JobStatus.FAILED, f"Inference failed: {stderr[:200]}")
                return 1

            print("Inference complete!")

            # Step 6: Download results
            print("\nDownloading results...")
            self.db.set_status(job.job_id, JobStatus.DOWNLOADING)

            output_dir = OUTPUTS_DIR / job.job_id
            predictions_path = transfer.download_predictions(output_dir)

            if predictions_path:
                print(f"Predictions saved to: {predictions_path}")
                self.db.set_output(job.job_id, str(predictions_path))

                # Score locally
                print("\nScoring predictions...")
                self.db.set_status(job.job_id, JobStatus.SCORING)
                score = self._score_predictions(predictions_path)
                if score:
                    print(f"\nSimilarity Score: {score:.4f}")
            else:
                print("Warning: Could not download predictions")

            self.db.set_status(job.job_id, JobStatus.COMPLETE)
            print(f"\nJob complete: {job.job_id}")

        except KeyboardInterrupt:
            print("\n\nAborted by user.")
            self.db.set_status(job.job_id, JobStatus.CANCELLED, "User abort")

        except Exception as e:
            print(f"\nError: {e}")
            self.db.set_status(job.job_id, JobStatus.FAILED, str(e))
            return 1

        finally:
            if auto_destroy and self.instance.instance_id:
                print(f"\nDestroying instance {self.instance.instance_id}...")
                self.instance.destroy()

        return 0

    def _score_predictions(self, predictions_path: Path) -> float:
        """Score predictions locally using embedding similarity."""
        try:
            # Import scorer from evaluation module
            import sys
            eval_dir = PROJECT_ROOT / "experiments" / "evaluation"
            sys.path.insert(0, str(eval_dir))
            from embedding_scorer import EmbeddingScorer

            import json
            results = []
            with open(predictions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

            predictions = [r['prediction'] for r in results]
            # Backward-compatible: accept both 'reference_response' and legacy 'ground_truth'
            references = [r.get('reference_response', r.get('ground_truth', '')) for r in results]

            scorer = EmbeddingScorer(device="cpu")
            similarities = scorer.compute_similarity(predictions, references)
            metrics = scorer.compute_aggregate_metrics(similarities)

            return metrics['mean_similarity']

        except Exception as e:
            print(f"Scoring failed: {e}")
            return None

    # =========================================================================
    # Batch Test Command
    # =========================================================================

    def cmd_batch_test(
        self,
        models: list,
        samples: int = 20,
        auto_destroy: bool = True,
        yes: bool = False,
        max_hours: float = None,
    ):
        """
        Test multiple models in sequence.

        Args:
            models: List of model names/IDs
            samples: Samples per model
            auto_destroy: Destroy instance after each model
            yes: Skip confirmations
            max_hours: Maximum hours per model
        """
        print("\n" + "=" * 60)
        print("Batch Test Configuration")
        print("=" * 60)
        print(f"Models: {len(models)}")
        for m in models:
            resolved = resolve_model_id(m)
            preset = get_preset_for_model(resolved)
            print(f"  - {resolved} ({preset.name})")
        print(f"Samples per model: {samples}")
        print(f"Auto-destroy: {auto_destroy}")
        if max_hours:
            print(f"Max hours per model: {max_hours}")

        if not yes:
            confirm = input("\nProceed with batch test? [y/N]: ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return 0

        results = []
        for i, model in enumerate(models, 1):
            print(f"\n{'='*60}")
            print(f"Model {i}/{len(models)}: {model}")
            print("=" * 60)

            ret = self.cmd_test(
                model_id=model,
                samples=samples,
                auto_destroy=auto_destroy,
                yes=True,
                max_hours=max_hours,
            )

            results.append({
                "model": resolve_model_id(model),
                "success": ret == 0,
            })

        # Summary
        print("\n" + "=" * 60)
        print("Batch Test Summary")
        print("=" * 60)
        for r in results:
            status = "OK" if r["success"] else "FAILED"
            print(f"  {r['model']}: {status}")

        return 0

    # =========================================================================
    # Finetune Command
    # =========================================================================

    def cmd_finetune(
        self,
        model_id: str,
        train_data: str = None,
        val_data: str = None,
        epochs: int = 2,
        batch_size: int = 1,
        gradient_accumulation: int = 8,
        learning_rate: float = 2e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        preset: str = None,
        dry_run: bool = False,
        auto_destroy: bool = False,
        yes: bool = False,
        max_hours: float = None,
    ):
        """
        Fine-tune a VLM model on the training dataset.

        Args:
            model_id: Model name, alias, or HuggingFace ID
            train_data: Path to training JSONL file
            val_data: Path to validation JSONL file
            epochs: Number of training epochs
            batch_size: Per-device batch size
            gradient_accumulation: Gradient accumulation steps
            learning_rate: Learning rate
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            preset: GPU preset override
            dry_run: Show plan without executing
            auto_destroy: Destroy instance after completion
            yes: Skip confirmation prompts
            max_hours: Maximum runtime hours (billing protection)
        """
        # Resolve model ID
        model_id = resolve_model_id(model_id)

        # Get preset - finetune needs more VRAM, prefer medium or large
        if preset:
            gpu_preset = PRESETS.get(preset)
            if not gpu_preset:
                print(f"Unknown preset: {preset}")
                print_presets()
                return 1
        else:
            gpu_preset = get_preset_for_model(model_id)
            # For fine-tuning, prefer A100 if model fits in 24GB but could benefit from more
            _, vram_est, preset_name = MODEL_CONFIGS.get(model_id, (7.0, 16, "medium"))
            if preset_name == "small" and vram_est > 10:
                gpu_preset = PRESETS.get("medium", gpu_preset)

        # Default training data paths
        if train_data is None:
            train_data = PROJECT_ROOT / "data" / "vlm_training" / "gt_train.jsonl"
        else:
            train_data = Path(train_data)

        if val_data is None:
            val_data = PROJECT_ROOT / "data" / "vlm_training" / "gt_val.jsonl"
        else:
            val_data = Path(val_data)

        print("\n" + "=" * 60)
        print("VLM Fine-Tuning Configuration")
        print("=" * 60)
        print(f"Model:         {model_id}")
        print(f"Train data:    {train_data}")
        print(f"Val data:      {val_data}")
        print(f"Epochs:        {epochs}")
        print(f"Batch size:    {batch_size} x {gradient_accumulation} grad accum")
        print(f"Learning rate: {learning_rate}")
        print(f"LoRA:          r={lora_r}, alpha={lora_alpha}")
        print(f"GPU Preset:    {gpu_preset.name} ({gpu_preset.description})")
        print(f"Max $/hr:      ${gpu_preset.max_price:.2f}")
        if max_hours:
            print(f"Max runtime:   {max_hours:.1f} hours")

        if dry_run:
            print("\n[DRY RUN] Would execute the above configuration")
            return 0

        # Check training data exists
        if not train_data.exists():
            print(f"\nError: Training data not found: {train_data}")
            return 1

        # Check images exist
        if not IMAGES_DIR.exists():
            print(f"\nError: Images not found: {IMAGES_DIR}")
            return 1

        # Confirmation
        if not yes:
            confirm = input("\nProceed with fine-tuning? [y/N]: ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return 0

        # Create job record
        job = self.db.create_job(
            job_type=JobType.FINETUNE,
            model_id=model_id,
            preset=gpu_preset.name,
            metadata={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "auto_destroy": auto_destroy,
                "max_hours": max_hours,
            }
        )
        print(f"\nJob created: {job.job_id}")

        try:
            # Step 1: Search for GPU
            print(f"\n[1/6] Searching for {gpu_preset.gpu_name}...")
            offers = self.instance.search_offers(
                gpu_name=gpu_preset.gpu_name,
                min_vram=gpu_preset.min_vram,
                max_price=gpu_preset.max_price,
                min_reliability=0.99,  # Higher reliability to avoid SSH failures
                min_download=500,  # Higher min download for faster uploads
            )

            if not offers:
                print("No suitable GPU offers found. Try:")
                print(f"  - Increase --max-price (current: ${gpu_preset.max_price})")
                print(f"  - Use a different preset")
                self.db.set_status(job.job_id, JobStatus.FAILED, "No GPU offers found")
                return 1

            self.instance.print_offers(offers, limit=5)
            offer = offers[0]
            print(f"\nSelected: {offer.gpu_name} @ ${offer.price_per_hour:.3f}/hr")

            # Step 2: Create instance
            print(f"\n[2/6] Creating instance...")
            if not self.instance.create(offer.offer_id, disk_gb=gpu_preset.disk_gb):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Failed to create instance")
                return 1

            print(f"Instance created: {self.instance.instance_id}")
            self.db.set_status(job.job_id, JobStatus.CREATING)

            # Step 3: Wait for instance
            print(f"\n[3/6] Waiting for instance...")
            if not self.instance.wait_until_ready(timeout=900):  # 15 min for Docker pull
                self.db.set_status(job.job_id, JobStatus.FAILED, "Instance failed to start")
                return 1

            self.db.set_instance(
                job.job_id,
                self.instance.instance_id,
                self.instance.ssh_host,
                self.instance.ssh_port
            )
            print(f"Instance ready: {self.instance.ssh_command}")

            # Brief delay to let SSH stabilize
            print("Waiting for SSH to stabilize...")
            time.sleep(20)

            # Check GPU
            gpu_info = check_gpu(self.instance)
            if gpu_info:
                print(f"GPU: {gpu_info['name']}, VRAM: {gpu_info['memory_total']}")
            else:
                print("Warning: Could not query GPU info")

            # Validate instance meets requirements
            print("Validating instance...")
            validation = self.instance.validate_instance()
            if validation.get("driver_version"):
                print(f"  Driver version: {validation['driver_version']}")
            if validation.get("cuda_version"):
                print(f"  CUDA version: {validation['cuda_version']}")
            if not validation.get("valid", False):
                errors = validation.get("errors", ["Unknown validation error"])
                print(f"VALIDATION FAILED: {errors}")
                self.db.set_status(job.job_id, JobStatus.FAILED, f"Instance validation failed: {errors}")
                if self.instance.instance_id:
                    self.instance.destroy()
                return 1
            print("  Validation passed")

            # Step 4: Setup environment and upload data
            print(f"\n[4/6] Setting up environment...")
            self.db.set_status(job.job_id, JobStatus.UPLOADING)

            if not setup_environment_with_retry(self.instance, max_retries=3, use_flash_attn=False):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Environment setup/validation failed after 3 attempts")
                return 1

            transfer = DataTransfer(self.instance)
            if not transfer.setup_workspace():
                self.db.set_status(job.job_id, JobStatus.FAILED, "Workspace setup failed")
                return 1

            # Upload training data
            print("Uploading training data...")
            if not transfer.upload_training_data(train_data, val_data):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Failed to upload training data")
                return 1

            # Upload images
            print("Uploading images...")
            if not transfer.upload_training_images(train_data, val_data):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Failed to upload images")
                return 1

            # Upload finetune script
            finetune_script = TEMPLATES_DIR / "run_finetune.py"
            if not transfer.upload_script(finetune_script):
                self.db.set_status(job.job_id, JobStatus.FAILED, "Failed to upload script")
                return 1

            # Step 5: Run fine-tuning
            print(f"\n[5/6] Starting fine-tuning...")
            self.db.set_status(job.job_id, JobStatus.RUNNING)

            # Start watchdog if max_hours specified
            watchdog = None
            if max_hours:
                watchdog = RuntimeWatchdog(
                    instance=self.instance,
                    db=self.db,
                    job_id=job.job_id,
                    max_hours=max_hours,
                )
                watchdog.start()

            finetune_cmd = f"""
            cd {REMOTE_WORKSPACE} && python scripts/run_finetune.py \
                --model {model_id} \
                --train-data data/train.jsonl \
                --val-data data/val.jsonl \
                --epochs {epochs} \
                --batch-size {batch_size} \
                --gradient-accumulation {gradient_accumulation} \
                --learning-rate {learning_rate} \
                --lora-r {lora_r} \
                --lora-alpha {lora_alpha} \
                --output outputs
            """

            # Run in foreground (wait for completion)
            print("Running fine-tuning (this may take several hours)...")
            print(f"SSH: {self.instance.ssh_command}")

            # Calculate timeout based on max_hours or default to 8 hours
            timeout = int((max_hours or 8) * 3600)

            ret, stdout, stderr = self.instance.run_ssh(
                finetune_cmd,
                timeout=timeout,
                capture=True
            )

            # Stop watchdog
            if watchdog:
                watchdog.stop()

            if ret != 0:
                error_msg = stderr or stdout or "Unknown error"
                # Clean non-ASCII chars for Windows console
                error_msg_clean = error_msg.encode('ascii', errors='replace').decode('ascii')
                print(f"Fine-tuning failed: {error_msg_clean}")
                # Truncate safely
                error_summary = error_msg_clean[:200] if error_msg_clean else "No error details"
                self.db.set_status(job.job_id, JobStatus.FAILED, f"Training failed: {error_summary}")
                return 1

            print("Fine-tuning complete!")

            # Step 6: Download results
            print("\n[6/6] Downloading adapter...")
            self.db.set_status(job.job_id, JobStatus.DOWNLOADING)

            output_dir = OUTPUTS_DIR / job.job_id
            adapter_path = transfer.download_adapter(output_dir)

            if adapter_path:
                print(f"Adapter saved to: {adapter_path}")
                self.db.set_output(job.job_id, str(adapter_path))
            else:
                print("Warning: Could not download adapter")

            self.db.set_status(job.job_id, JobStatus.COMPLETE)
            print(f"\nJob complete: {job.job_id}")

        except KeyboardInterrupt:
            print("\n\nAborted by user.")
            self.db.set_status(job.job_id, JobStatus.CANCELLED, "User abort")

        except Exception as e:
            print(f"\nError: {e}")
            self.db.set_status(job.job_id, JobStatus.FAILED, str(e))
            return 1

        finally:
            if auto_destroy and self.instance.instance_id:
                print(f"\nDestroying instance {self.instance.instance_id}...")
                self.instance.destroy()

        return 0

    # =========================================================================
    # Destroy Command
    # =========================================================================

    def cmd_destroy(self, job_id: str = None, all_instances: bool = False):
        """Destroy instances."""
        if all_instances:
            active = self.db.get_active_jobs()
            if not active:
                print("No active instances to destroy.")
                return 0

            print(f"Destroying {len(active)} instances...")
            for job in active:
                if job.instance_id:
                    print(f"  Destroying {job.job_id} (instance {job.instance_id})...")
                    self.instance.destroy(job.instance_id)
                    self.db.set_status(job.job_id, JobStatus.CANCELLED, "User destroyed")
            return 0

        if not job_id:
            print("Specify job ID or --all")
            return 1

        job = self.db.get_job(job_id)
        if not job:
            print(f"Job not found: {job_id}")
            return 1

        if job.instance_id:
            print(f"Destroying instance {job.instance_id}...")
            self.instance.destroy(job.instance_id)
            self.db.set_status(job.job_id, JobStatus.CANCELLED, "User destroyed")
            print("Done.")
        else:
            print("No instance associated with this job.")

        return 0

    # =========================================================================
    # Download Command
    # =========================================================================

    def cmd_download(self, job_id: str):
        """Download results from a running job."""
        job = self.db.get_job(job_id)
        if not job:
            print(f"Job not found: {job_id}")
            return 1

        if not job.instance_id:
            print("No instance for this job.")
            return 1

        if not self.instance.connect(job.instance_id):
            print(f"Could not connect to instance {job.instance_id}")
            return 1

        print(f"Connected to {self.instance.ssh_command}")

        transfer = DataTransfer(self.instance)
        output_dir = OUTPUTS_DIR / job_id

        print("Downloading results...")
        downloaded = transfer.download_results(output_dir)

        if downloaded:
            print(f"\nDownloaded {len(downloaded)} files to {output_dir}")
            for f in downloaded:
                print(f"  - {f.name}")
        else:
            print("No results found to download.")

        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified Vast.ai CLI for VLM Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.vastai status
  python -m experiments.vastai test InternVL3-2B --samples 20
  python -m experiments.vastai test Qwen2.5-VL-3B --full --auto-destroy
  python -m experiments.vastai batch-test InternVL3-2B Qwen2.5-VL-3B SmolVLM2-2B
  python -m experiments.vastai destroy job-abc123
  python -m experiments.vastai models  # List supported models
  python -m experiments.vastai presets # List GPU presets
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show job status")
    status_parser.add_argument("job_id", nargs="?", help="Job ID (optional)")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test a VLM model")
    test_parser.add_argument("model", help="Model name or HuggingFace ID")
    test_parser.add_argument("--samples", type=int, default=20, help="Number of samples")
    test_parser.add_argument("--full", action="store_true", help="Full 600-sample evaluation")
    test_parser.add_argument("--preset", help="GPU preset override")
    test_parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    test_parser.add_argument("--auto-destroy", action="store_true", help="Destroy after completion")
    test_parser.add_argument("--max-hours", type=float, help="Maximum runtime hours (billing protection)")
    test_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmations")
    test_parser.add_argument("--skip-images", action="store_true", help="Skip image upload (if already on instance)")

    # Batch test command
    batch_parser = subparsers.add_parser("batch-test", help="Test multiple models")
    batch_parser.add_argument("models", nargs="+", help="Model names")
    batch_parser.add_argument("--samples", type=int, default=20, help="Samples per model")
    batch_parser.add_argument("--no-auto-destroy", action="store_true")
    batch_parser.add_argument("--max-hours", type=float, help="Maximum hours per model")
    batch_parser.add_argument("-y", "--yes", action="store_true")

    # Destroy command
    destroy_parser = subparsers.add_parser("destroy", help="Destroy instances")
    destroy_parser.add_argument("job_id", nargs="?", help="Job ID")
    destroy_parser.add_argument("--all", action="store_true", help="Destroy all instances")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download results")
    download_parser.add_argument("job_id", help="Job ID")

    # Models command
    subparsers.add_parser("models", help="List supported models")

    # Presets command
    subparsers.add_parser("presets", help="List GPU presets")

    # Finetune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune a VLM model")
    finetune_parser.add_argument("model", help="Model name or HuggingFace ID")
    finetune_parser.add_argument("--train-data", help="Path to training JSONL file")
    finetune_parser.add_argument("--val-data", help="Path to validation JSONL file")
    finetune_parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    finetune_parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    finetune_parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps")
    finetune_parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    finetune_parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    finetune_parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    finetune_parser.add_argument("--preset", help="GPU preset override")
    finetune_parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    finetune_parser.add_argument("--auto-destroy", action="store_true", help="Destroy after completion")
    finetune_parser.add_argument("--max-hours", type=float, help="Maximum runtime hours (billing protection)")
    finetune_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmations")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    runner = VastaiRunner()

    if args.command == "status":
        return runner.cmd_status(args.job_id)

    elif args.command == "test":
        return runner.cmd_test(
            model_id=args.model,
            samples=args.samples,
            full=args.full,
            preset=args.preset,
            dry_run=args.dry_run,
            auto_destroy=args.auto_destroy,
            yes=args.yes,
            skip_images=args.skip_images,
            max_hours=args.max_hours,
        )

    elif args.command == "batch-test":
        return runner.cmd_batch_test(
            models=args.models,
            samples=args.samples,
            auto_destroy=not args.no_auto_destroy,
            yes=args.yes,
            max_hours=args.max_hours,
        )

    elif args.command == "destroy":
        return runner.cmd_destroy(args.job_id, all_instances=args.all)

    elif args.command == "download":
        return runner.cmd_download(args.job_id)

    elif args.command == "models":
        print_model_list()
        return 0

    elif args.command == "presets":
        print_presets()
        return 0

    elif args.command == "finetune":
        return runner.cmd_finetune(
            model_id=args.model,
            train_data=args.train_data,
            val_data=args.val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            preset=args.preset,
            dry_run=args.dry_run,
            auto_destroy=args.auto_destroy,
            yes=args.yes,
            max_hours=args.max_hours,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
