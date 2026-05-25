"""
Gemini Batch API Annotation Script - Process fetal ultrasound images using batch API

This script uses the Gemini Batch API for 50% cost savings and to bypass daily rate limits.
Supports multiple modes: estimate, prepare, submit, status, download, process

Usage:
    # Token estimation
    python batch_gemini_annotation.py --mode estimate

    # Prepare JSONL files for batch submission
    python batch_gemini_annotation.py --mode prepare --batch-size 125

    # Submit batch job
    python batch_gemini_annotation.py --mode submit --batch-index 0

    # Check status
    python batch_gemini_annotation.py --mode status --job-id <job_id>

    # Download results
    python batch_gemini_annotation.py --mode download --job-id <job_id>

    # Process results and merge with checkpoint
    python batch_gemini_annotation.py --mode process
"""

import os
import sys
import json
import base64
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
from dotenv import load_dotenv

from src.data.question_loader import QuestionLoader
from experiments.evaluation.config import API_SYSTEM_PROMPT

# Constants
DEFAULT_MODEL = "gemini-3-flash-preview"
DATA_DIR = project_root / "data" / "Fetal Ultrasound"
RESULTS_DIR = Path(__file__).parent / "results"
BATCH_DIR = RESULTS_DIR / "batch_files"
JOBS_FILE = RESULTS_DIR / "batch_jobs.json"

SYSTEM_PROMPT = API_SYSTEM_PROMPT


def load_env():
    """Load environment variables"""
    env_path = project_root / '.env.local'
    load_dotenv(env_path)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment")
    return api_key


def get_genai_client():
    """Get Google GenAI client"""
    from google import genai
    api_key = load_env()
    client = genai.Client(api_key=api_key)
    return client


def image_to_base64(image_path: Path) -> Tuple[str, str]:
    """Convert image to base64 string and detect mime type"""
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        b64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return b64_string, "image/jpeg"


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load existing checkpoint to identify completed images"""
    if not checkpoint_path.exists():
        return {"completed_images": {}}
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def save_jobs_state(jobs: Dict[str, Any]):
    """Save batch job state for resume"""
    with open(JOBS_FILE, 'w') as f:
        json.dump(jobs, f, indent=2)


def load_jobs_state() -> Dict[str, Any]:
    """Load batch job state"""
    if JOBS_FILE.exists():
        with open(JOBS_FILE, 'r') as f:
            return json.load(f)
    return {"jobs": [], "created_at": datetime.now().isoformat()}


def get_all_images() -> List[Tuple[str, Path]]:
    """Get all images from data directory with their categories"""
    images = []
    for category_dir in sorted(DATA_DIR.iterdir()):
        if category_dir.is_dir():
            category = category_dir.name
            for img_path in sorted(category_dir.glob("*.png")):
                images.append((category, img_path))
            for img_path in sorted(category_dir.glob("*.jpg")):
                images.append((category, img_path))
    return images


def get_remaining_images(checkpoint: Dict, all_images: List[Tuple[str, Path]]) -> List[Tuple[str, Path]]:
    """Filter out completed images"""
    completed = set(checkpoint.get("completed_images", {}).keys())
    remaining = []
    for category, img_path in all_images:
        key = f"{category}/{img_path.name}"
        if key not in completed:
            remaining.append((category, img_path))
    return remaining


def estimate_tokens(args):
    """Estimate token count for batch processing"""
    print("=" * 60)
    print("TOKEN ESTIMATION MODE")
    print("=" * 60)

    client = get_genai_client()
    question_loader = QuestionLoader(str(DATA_DIR))
    questions = question_loader.get_questions()

    # Load checkpoint to find remaining images
    checkpoint_path = RESULTS_DIR / args.checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    all_images = get_all_images()
    remaining = get_remaining_images(checkpoint, all_images)

    print(f"\nDataset status:")
    print(f"  Total images: {len(all_images)}")
    print(f"  Completed: {len(all_images) - len(remaining)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Questions per image: {len(questions)}")
    print(f"  Total requests needed: {len(remaining) * len(questions)}")

    # Sample a few images for token counting
    sample_size = min(5, len(remaining))
    if sample_size == 0:
        print("\nNo remaining images to process!")
        return

    print(f"\nSampling {sample_size} images for token estimation...")

    total_tokens = 0
    samples = remaining[:sample_size]

    for idx, (category, img_path) in enumerate(samples):
        b64_data, mime_type = image_to_base64(img_path)

        # Count tokens for first question as representative
        question = questions[0]
        prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}"

        try:
            # Use count_tokens API
            from google.genai import types
            result = client.models.count_tokens(
                model=f"models/{args.model}",
                contents=[
                    types.Content(
                        parts=[
                            types.Part(inline_data=types.Blob(mime_type=mime_type, data=base64.b64decode(b64_data))),
                            types.Part(text=prompt)
                        ]
                    )
                ]
            )
            tokens = result.total_tokens
            total_tokens += tokens
            print(f"  Sample {idx+1} ({category}/{img_path.name}): {tokens} tokens")
        except Exception as e:
            print(f"  Sample {idx+1} error: {e}")
            # Fallback estimate: ~258 for image + ~150 for prompt
            total_tokens += 408
            print(f"  Using fallback estimate: 408 tokens")

    avg_tokens = total_tokens / sample_size
    total_estimate = avg_tokens * len(remaining) * len(questions)

    print(f"\nToken estimation:")
    print(f"  Average tokens per request: {avg_tokens:.0f}")
    print(f"  Total estimated tokens: {total_estimate:,.0f}")

    # Tier requirements
    print(f"\nTier requirements:")
    print(f"  Tier 1 (3M tokens):   {total_estimate / 3_000_000:.1f} batches needed")
    print(f"  Tier 2 (400M tokens): {total_estimate / 400_000_000:.1f} batches needed")

    # Cost estimate (input only, 50% batch discount)
    # Flash pricing: $0.075 per 1M input tokens (standard) -> $0.0375 (batch)
    cost_estimate = (total_estimate / 1_000_000) * 0.0375
    print(f"\nCost estimate (batch pricing):")
    print(f"  ~${cost_estimate:.2f} USD (input tokens only)")


def prepare_batch(args):
    """Generate JSONL files for batch submission"""
    print("=" * 60)
    print("PREPARE BATCH MODE")
    print("=" * 60)

    question_loader = QuestionLoader(str(DATA_DIR))
    questions = question_loader.get_questions()

    # Load checkpoint to find remaining images
    checkpoint_path = RESULTS_DIR / args.checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    all_images = get_all_images()
    remaining = get_remaining_images(checkpoint, all_images)

    print(f"\nRemaining images: {len(remaining)}")
    print(f"Questions per image: {len(questions)}")
    print(f"Batch size (images): {args.batch_size}")

    # Create batch directory
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate batches
    requests_per_batch = args.batch_size * len(questions)
    num_batches = (len(remaining) + args.batch_size - 1) // args.batch_size

    print(f"Requests per batch: {requests_per_batch}")
    print(f"Number of batches: {num_batches}")

    # Process batches
    for batch_idx in range(num_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(remaining))
        batch_images = remaining[start:end]

        jsonl_path = BATCH_DIR / f"batch_{batch_idx:03d}.jsonl"
        print(f"\nGenerating batch {batch_idx} ({len(batch_images)} images)...")

        with open(jsonl_path, 'w') as f:
            for category, img_path in batch_images:
                # Encode image once per image
                b64_data, mime_type = image_to_base64(img_path)
                image_key = f"{category}/{img_path.name}"

                # Create request for each question
                for q_idx, question in enumerate(questions):
                    request_key = f"{image_key}_q{q_idx}"
                    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}"

                    request_obj = {
                        "key": request_key,
                        "request": {
                            "contents": [{
                                "parts": [
                                    {
                                        "inline_data": {
                                            "mime_type": mime_type,
                                            "data": b64_data
                                        }
                                    },
                                    {"text": prompt}
                                ]
                            }]
                        }
                    }

                    f.write(json.dumps(request_obj) + "\n")

        file_size = jsonl_path.stat().st_size / (1024 * 1024)
        print(f"  Written: {jsonl_path.name} ({file_size:.1f} MB)")

    print(f"\nGenerated {num_batches} batch file(s) in {BATCH_DIR}")


def submit_batch(args):
    """Upload JSONL and submit batch job"""
    print("=" * 60)
    print("SUBMIT BATCH MODE")
    print("=" * 60)

    from google.genai import types

    client = get_genai_client()
    jobs_state = load_jobs_state()

    # Find batch files to submit
    if args.batch_index is not None:
        batch_files = [BATCH_DIR / f"batch_{args.batch_index:03d}.jsonl"]
    elif args.all_batches:
        batch_files = sorted(BATCH_DIR.glob("batch_*.jsonl"))
    else:
        # Submit next unsubmitted batch
        submitted = {j.get("batch_file") for j in jobs_state.get("jobs", [])}
        batch_files = [f for f in sorted(BATCH_DIR.glob("batch_*.jsonl"))
                       if f.name not in submitted][:1]

    if not batch_files:
        print("No batch files to submit")
        return

    for batch_file in batch_files:
        if not batch_file.exists():
            print(f"Batch file not found: {batch_file}")
            continue

        print(f"\nSubmitting: {batch_file.name}")

        # Upload file
        print("  Uploading JSONL file...")
        uploaded_file = client.files.upload(
            file=str(batch_file),
            config=types.UploadFileConfig(
                display_name=batch_file.stem,
                mime_type='application/jsonl'
            )
        )
        print(f"  Uploaded: {uploaded_file.name}")

        # Create batch job
        print(f"  Creating batch job with model: {args.model}")
        batch_job = client.batches.create(
            model=f"models/{args.model}",
            src=uploaded_file.name,
            config={
                'display_name': f"fetal-ultrasound-{batch_file.stem}",
            },
        )
        print(f"  Job created: {batch_job.name}")
        print(f"  State: {batch_job.state.name}")

        # Save job info
        jobs_state.setdefault("jobs", []).append({
            "job_name": batch_job.name,
            "batch_file": batch_file.name,
            "uploaded_file": uploaded_file.name,
            "submitted_at": datetime.now().isoformat(),
            "state": batch_job.state.name
        })
        save_jobs_state(jobs_state)

    print(f"\nSubmitted {len(batch_files)} batch(es). Use --mode status to check progress.")


def check_status(args):
    """Check status of submitted batch jobs"""
    print("=" * 60)
    print("STATUS CHECK MODE")
    print("=" * 60)

    client = get_genai_client()
    jobs_state = load_jobs_state()

    if args.job_id:
        # Check specific job
        job_names = [args.job_id]
    else:
        # Check all jobs
        job_names = [j["job_name"] for j in jobs_state.get("jobs", [])]

    if not job_names:
        print("No jobs to check. Submit a batch first.")
        return

    completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}

    for job_name in job_names:
        print(f"\nJob: {job_name}")
        try:
            batch_job = client.batches.get(name=job_name)
            state = batch_job.state.name
            print(f"  State: {state}")

            # Update stored state
            for j in jobs_state.get("jobs", []):
                if j["job_name"] == job_name:
                    j["state"] = state
                    if state == "JOB_STATE_SUCCEEDED" and batch_job.dest:
                        if hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
                            j["result_file"] = batch_job.dest.file_name
                    break

            if state == "JOB_STATE_FAILED":
                print(f"  Error: {batch_job.error}")
            elif state == "JOB_STATE_SUCCEEDED":
                if batch_job.dest and hasattr(batch_job.dest, 'file_name'):
                    print(f"  Result file: {batch_job.dest.file_name}")

        except Exception as e:
            print(f"  Error checking job: {e}")

    save_jobs_state(jobs_state)


def download_results(args):
    """Download completed batch results"""
    print("=" * 60)
    print("DOWNLOAD RESULTS MODE")
    print("=" * 60)

    client = get_genai_client()
    jobs_state = load_jobs_state()

    # Find completed jobs with results
    if args.job_id:
        jobs_to_download = [j for j in jobs_state.get("jobs", [])
                           if j["job_name"] == args.job_id]
    else:
        jobs_to_download = [j for j in jobs_state.get("jobs", [])
                           if j.get("state") == "JOB_STATE_SUCCEEDED"
                           and not j.get("downloaded")]

    if not jobs_to_download:
        print("No completed jobs to download. Check status first.")
        return

    results_subdir = RESULTS_DIR / "batch_results"
    results_subdir.mkdir(parents=True, exist_ok=True)

    for job_info in jobs_to_download:
        job_name = job_info["job_name"]
        print(f"\nDownloading: {job_name}")

        try:
            batch_job = client.batches.get(name=job_name)

            if batch_job.state.name != "JOB_STATE_SUCCEEDED":
                print(f"  Job not succeeded (state: {batch_job.state.name})")
                continue

            if not batch_job.dest or not hasattr(batch_job.dest, 'file_name'):
                print("  No result file found")
                continue

            result_file_name = batch_job.dest.file_name
            print(f"  Result file: {result_file_name}")

            # Download result file
            file_content = client.files.download(file=result_file_name)
            output_path = results_subdir / f"{job_info['batch_file'].replace('.jsonl', '_results.jsonl')}"

            with open(output_path, 'wb') as f:
                f.write(file_content)

            print(f"  Downloaded to: {output_path}")

            # Mark as downloaded
            job_info["downloaded"] = True
            job_info["result_path"] = str(output_path)

        except Exception as e:
            print(f"  Error downloading: {e}")

    save_jobs_state(jobs_state)


def process_results(args):
    """Process downloaded results and merge with checkpoint"""
    print("=" * 60)
    print("PROCESS RESULTS MODE")
    print("=" * 60)

    results_subdir = RESULTS_DIR / "batch_results"

    # Find result files
    result_files = list(results_subdir.glob("*_results.jsonl"))
    if not result_files:
        print("No result files found. Download results first.")
        return

    print(f"Found {len(result_files)} result file(s)")

    # Load existing checkpoint
    checkpoint_path = RESULTS_DIR / args.checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    completed_images = checkpoint.get("completed_images", {})

    print(f"Existing checkpoint has {len(completed_images)} images")

    # Process each result file
    new_responses = {}  # key -> {question_idx -> response}

    for result_file in result_files:
        print(f"\nProcessing: {result_file.name}")
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    result = json.loads(line)
                    key = result.get("key", "")

                    # Parse key: "Category/image.png_q0"
                    if "_q" not in key:
                        continue

                    image_key, q_part = key.rsplit("_q", 1)
                    question_idx = int(q_part)

                    # Extract response text
                    response_text = ""
                    error = None

                    if "response" in result:
                        resp = result["response"]
                        if "candidates" in resp and resp["candidates"]:
                            candidate = resp["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                for part in candidate["content"]["parts"]:
                                    if "text" in part:
                                        response_text = part["text"]
                                        break
                    elif "error" in result:
                        error = str(result["error"])

                    if image_key not in new_responses:
                        new_responses[image_key] = {}

                    new_responses[image_key][question_idx] = {
                        "response": response_text,
                        "error": error
                    }

                except Exception as e:
                    print(f"  Error parsing line: {e}")
                    continue

    print(f"\nProcessed responses for {len(new_responses)} images")

    # Load questions for short names
    question_loader = QuestionLoader(str(DATA_DIR))
    questions = question_loader.get_questions()

    # Convert to checkpoint format
    for image_key, question_responses in new_responses.items():
        if image_key in completed_images:
            # Skip if already in checkpoint
            continue

        # Parse category and image name
        parts = image_key.split("/", 1)
        if len(parts) != 2:
            continue
        category, image_name = parts

        # Build question results
        question_results = []
        for q_idx in range(len(questions)):
            resp_data = question_responses.get(q_idx, {"response": "", "error": "Missing"})
            question_results.append({
                "question_idx": q_idx,
                "question": questions[q_idx][:50] + "..." if len(questions[q_idx]) > 50 else questions[q_idx],
                "response": resp_data["response"],
                "time": 0,  # Not applicable for batch
                "evaluation": None,  # Can be added later
                "error": resp_data["error"]
            })

        completed_images[image_key] = {
            "image": image_name,
            "category": category,
            "questions": question_results
        }

    # Save updated checkpoint
    checkpoint["completed_images"] = completed_images
    checkpoint["model_name"] = args.model
    checkpoint["config"] = {"batch_mode": True, "model": args.model}
    checkpoint["saved_at"] = datetime.now().isoformat()

    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\nCheckpoint updated: {checkpoint_path}")
    print(f"Total completed images: {len(completed_images)}")

    # Summary
    all_images = get_all_images()
    remaining = get_remaining_images(checkpoint, all_images)
    print(f"Remaining images: {len(remaining)}")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini Batch API Annotation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--mode",
        choices=["estimate", "prepare", "submit", "status", "download", "process"],
        required=True,
        help="Operation mode"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_gemini_gemini-3-flash-preview.json",
        help="Checkpoint file name in results directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=125,
        help="Number of images per batch (default: 125 = 1000 requests)"
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=None,
        help="Specific batch index to submit"
    )
    parser.add_argument(
        "--all-batches",
        action="store_true",
        help="Submit all prepared batches"
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Specific job ID for status/download"
    )

    args = parser.parse_args()

    # Route to appropriate function
    if args.mode == "estimate":
        estimate_tokens(args)
    elif args.mode == "prepare":
        prepare_batch(args)
    elif args.mode == "submit":
        submit_batch(args)
    elif args.mode == "status":
        check_status(args)
    elif args.mode == "download":
        download_results(args)
    elif args.mode == "process":
        process_results(args)


if __name__ == "__main__":
    main()
