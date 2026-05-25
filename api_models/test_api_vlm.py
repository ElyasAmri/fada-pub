"""
Parallel API VLM Response Quality Test - Evaluate vision models with concurrent requests and rate limiting

Note: Uses "response quality score" not "accuracy" since VLMs produce free-text responses
evaluated by rubrics, not classification against ground-truth labels.

Usage:
    python test_api_vlm_parallel.py --models grok --images-per-category 3 --max-rpm 60
    python test_api_vlm_parallel.py --models gemini --max-rpm 30 --max-concurrent 5
    python test_api_vlm_parallel.py --models all --quiet  # Minimal output
"""
import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'  # Suppress glog INFO/WARNING

import sys
sys.stdout.reconfigure(line_buffering=True)

import logging
import argparse
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
from tqdm.asyncio import tqdm_asyncio

from src.data.question_loader import QuestionLoader
from src.data.dataset_splits import load_splits, SPLITS_FILE
from experiments.evaluation.config import API_SYSTEM_PROMPT, GENERATION_TEMPERATURE as EVAL_TEMPERATURE


# RPD (Requests Per Day) limit error patterns
RPD_ERROR_PATTERNS = [
    'quota exceeded',
    'rate limit',
    'too many requests',
    'resource exhausted',
    'resourceexhausted',
    'daily limit',
    'requests per day',
    '429',
    # Vertex AI specific (lowercase to match is_rpd_error() which lowercases the error string)
    'resource_exhausted',
    'quota_exceeded',
    'ratelimitexceeded',
]


class RPDLimitError(Exception):
    """Raised when API rate/quota limit is hit"""
    pass



def is_rpd_error(error: Exception) -> bool:
    """Check if an error is a rate/quota limit error"""
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in RPD_ERROR_PATTERNS)


def save_checkpoint(checkpoint_path: Path, completed_images: Dict[str, Dict],
                    model_name: str, config: Dict, partial_results: List[Dict]) -> None:
    """Save checkpoint for resuming later"""
    checkpoint = {
        'model_name': model_name,
        'config': config,
        'completed_images': completed_images,  # {image_key: result}
        'partial_results': partial_results,
        'saved_at': datetime.now().isoformat()
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    """Load checkpoint if it exists"""
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


@dataclass
class RateLimiter:
    """Token bucket rate limiter for API requests"""
    max_rpm: int  # Maximum requests per minute
    max_concurrent: int = 10  # Maximum concurrent requests

    _tokens: float = field(default=0, init=False)
    _last_update: float = field(default_factory=time.time, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _semaphore: asyncio.Semaphore = field(default=None, init=False)

    def __post_init__(self):
        self._tokens = self.max_rpm / 60  # Start with some tokens
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def acquire(self):
        """Acquire a token, waiting if necessary"""
        while True:
            wait_time = 0.0
            async with self._lock:
                now = time.time()
                # Add tokens based on time elapsed
                elapsed = now - self._last_update
                self._tokens = min(
                    self.max_rpm / 60,  # Max 1 minute worth of tokens
                    self._tokens + elapsed * (self.max_rpm / 60)
                )
                self._last_update = now

                if self._tokens >= 1:
                    self._tokens -= 1
                    break  # Token acquired, exit loop
                else:
                    wait_time = (1 - self._tokens) / (self.max_rpm / 60)

            # Sleep outside the lock so other coroutines can proceed
            await asyncio.sleep(wait_time)

        # Also respect max concurrent
        await self._semaphore.acquire()

    def release(self):
        """Release the semaphore"""
        self._semaphore.release()


class AsyncGrokVLM:
    """Async wrapper for Grok VLM"""

    BASE_URL = "https://api.x.ai/v1"

    def __init__(self, model_name: str = "grok-4", api_key: Optional[str] = None):
        import os
        from dotenv import load_dotenv

        env_path = project_root / '.env.local'
        load_dotenv(env_path)

        self.model_name = model_name
        self.api_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        self._client = None

        if not self.api_key:
            raise ValueError("XAI_API_KEY or GROK_API_KEY not found")

    async def load(self):
        """Initialize async client"""
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL
        )

    async def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image asynchronously"""
        from src.utils.image_processing import to_base64_data_url

        if self._client is None:
            await self.load()

        image_url = to_base64_data_url(image)

        messages = [
            {
                "role": "system",
                "content": API_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                    {"type": "text", "text": question}
                ]
            }
        ]

        response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=EVAL_TEMPERATURE
        )

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return "No response generated."


class AsyncOpenAIVLM:
    """Async wrapper for OpenAI VLM (GPT-5.2, GPT-4o)"""

    def __init__(self, model_name: str = "gpt-5.2-chat-latest", api_key: Optional[str] = None):
        import os
        from dotenv import load_dotenv

        env_path = project_root / '.env.local'
        load_dotenv(env_path)

        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")

    async def load(self):
        """Initialize async client"""
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=self.api_key)

    async def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image asynchronously with retry for vision routing errors"""
        from src.utils.image_processing import to_base64_data_url

        if self._client is None:
            await self.load()

        image_url = to_base64_data_url(image)

        messages = [
            {
                "role": "system",
                "content": API_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                    {"type": "text", "text": question}
                ]
            }
        ]

        # Retry logic for GPT-5.2 vision routing issues
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # GPT-5.x does not support custom temperature; results use API default (~1.0). See issue #38.
                if self.model_name.startswith("gpt-5"):
                    response = await self._client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_completion_tokens=1024
                    )
                else:
                    response = await self._client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=1024,
                        temperature=EVAL_TEMPERATURE
                    )

                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                return "No response generated."

            except Exception as e:
                last_error = e
                error_str = str(e)
                # Retry on vision routing errors (GPT-5.2 intermittent issue)
                if "image_url is only supported" in error_str or "Invalid content type" in error_str:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))  # Backoff
                        continue
                raise  # Re-raise non-retryable errors immediately

        # All retries exhausted
        raise last_error if last_error else RuntimeError("Request failed")


class AsyncGeminiVLM:
    """Async wrapper for Gemini VLM"""

    def __init__(self, model_name: str = "gemini-3-flash-preview", api_key: Optional[str] = None):
        import os
        from dotenv import load_dotenv

        env_path = project_root / '.env.local'
        load_dotenv(env_path)

        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._client = None

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found")

    async def load(self):
        """Initialize Gemini client"""
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(self.model_name)

    async def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image"""
        if self._client is None:
            await self.load()

        prompt = f"""{API_SYSTEM_PROMPT}

Question: {question}"""

        # Gemini's generate_content is sync, run in executor
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.generate_content(
                [prompt, image],
                generation_config={"temperature": EVAL_TEMPERATURE}
            )
        )

        # Guard against safety-blocked or empty responses
        try:
            text = response.text
        except Exception:
            text = None
        if text:
            return text.strip()
        return "No response generated."


class AsyncVertexAIVLM:
    """Async wrapper for Vertex AI MedGemma endpoint"""

    def __init__(
        self,
        model_name: str = "medgemma-27b-mm-it",
        project_id: Optional[str] = None,
        location: str = "us-central1",
        endpoint_id: Optional[str] = None
    ):
        import os
        from dotenv import load_dotenv

        env_path = project_root / '.env.local'
        load_dotenv(env_path)

        self.model_name = model_name
        self.project_id = project_id or os.getenv("VERTEX_AI_PROJECT_ID")
        self.location = location or os.getenv("VERTEX_AI_LOCATION", "us-central1")
        self.endpoint_id = endpoint_id or os.getenv("VERTEX_AI_ENDPOINT_ID")

        self._credentials = None
        self._endpoint_url = None
        self._session = None

        if not self.project_id:
            raise ValueError("VERTEX_AI_PROJECT_ID not found in environment")
        if not self.endpoint_id:
            raise ValueError("VERTEX_AI_ENDPOINT_ID not found in environment")

    async def load(self):
        """Initialize credentials and aiohttp session"""
        import google.auth
        import google.auth.transport.requests
        import aiohttp

        # Get Application Default Credentials
        self._credentials, _ = google.auth.default()

        # Build endpoint URL
        self._endpoint_url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.location}/"
            f"endpoints/{self.endpoint_id}:predict"
        )

        # Create aiohttp session
        self._session = aiohttp.ClientSession()

    async def _get_access_token(self) -> str:
        """Get fresh access token, refreshing if needed"""
        import google.auth.transport.requests

        # Run refresh in executor (it's a sync operation)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._credentials.refresh(google.auth.transport.requests.Request())
        )
        return self._credentials.token

    async def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image asynchronously"""
        from src.utils.image_processing import to_base64_data_url

        if self._session is None:
            await self.load()

        image_url = to_base64_data_url(image)

        # Build request body in vLLM OpenAI-compatible format
        request_body = {
            "instances": [
                {
                    "@requestFormat": "chatCompletions",
                    "messages": [
                        {
                            "role": "system",
                            "content": API_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url}
                                },
                                {
                                    "type": "text",
                                    "text": question
                                }
                            ]
                        }
                    ],
                    "max_tokens": 1024,
                    "temperature": EVAL_TEMPERATURE
                }
            ]
        }

        # Get fresh token
        token = await self._get_access_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        async with self._session.post(
            self._endpoint_url,
            headers=headers,
            json=request_body,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Vertex AI API error {response.status}: {error_text}")

            result = await response.json()
            return self._extract_response_text(result)

    def _extract_response_text(self, response: dict) -> str:
        """Extract text from Vertex AI response"""
        try:
            # vLLM chat completions format
            predictions = response.get("predictions", [])
            if predictions:
                choices = predictions[0].get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    if content:
                        return content.strip()

            # Fallback: try direct text extraction
            if "predictions" in response and response["predictions"]:
                pred = response["predictions"][0]
                if isinstance(pred, str):
                    return pred.strip()
                if isinstance(pred, dict) and "content" in pred:
                    return pred["content"].strip()

            return "No response generated."
        except Exception as e:
            return f"Error parsing response: {e}"

    async def close(self):
        """Close the aiohttp session"""
        if self._session:
            await self._session.close()
            self._session = None


class AsyncVLLM:
    """Async wrapper for local vLLM server (OpenAI-compatible API)"""

    def __init__(self, base_url: str = "http://localhost:8000", model_name: str = "google/medgemma-27b-it"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self._client = None

    async def load(self):
        """Initialize async OpenAI client pointing to vLLM"""
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key="not-needed",  # vLLM doesn't require API key
            base_url=f"{self.base_url}/v1"
        )

    async def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image using vLLM server"""
        from src.utils.image_processing import to_base64_data_url

        if self._client is None:
            await self.load()

        image_url = to_base64_data_url(image)

        messages = [
            {
                "role": "system",
                "content": API_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": question}
                ]
            }
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=EVAL_TEMPERATURE
            )

            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return "No response generated."
        except Exception as e:
            raise RuntimeError(f"vLLM API error: {e}")


def evaluate_response(response: str, category: str, question_idx: int) -> Dict[str, Any]:
    """Evaluate a model response quality using rubric-based scoring"""
    response_lower = response.lower()

    fetal_keywords = ['fetal', 'fetus', 'ultrasound', 'pregnancy', 'prenatal',
                      'gestational', 'amniotic', 'placenta', 'umbilical']

    # Keywords shared across multiple categories -- excluded from hallucination penalty
    shared_keywords = {
        'heart', 'cardiac', 'ventricle', 'atrium', 'nuchal', 'translucency',
        'nt', 'cervical', 'crown-rump',
    }

    anatomy_keywords = {
        'Abdomen': ['abdomen', 'stomach', 'liver', 'kidney', 'bowel', 'intestine', 'bladder', 'abdominal'],
        'Aorta': ['aorta', 'heart', 'cardiac', 'vessel', 'artery', 'ventricle', 'atrium'],
        'CRL-View': ['crown-rump', 'crl', 'embryo', 'crown', 'rump'],
        'Cervical': ['cervical', 'cervix', 'neck', 'spine'],
        'Cervix': ['cervix', 'cervical', 'uterus'],
        'Femur': ['femur', 'bone', 'leg', 'limb', 'thigh', 'long bone', 'skeletal'],
        'NT-View': ['nuchal translucency', 'nuchal', 'nt', 'translucency'],
        'Non_standard_NT': ['nuchal', 'translucency', 'nt', 'non-standard', 'non standard'],
        'Public_Symphysis_fetal_head': ['pubic symphysis', 'symphysis', 'head engagement', 'head position', 'fetal head'],
        'Standard_NT': ['nuchal', 'translucency', 'nt', 'standard'],
        'Thorax': ['thorax', 'chest', 'lung', 'rib', 'diaphragm', 'thoracic'],
        'Trans-cerebellum': ['cerebellum', 'posterior fossa', 'cisterna magna', 'vermis', 'cerebellar'],
        'Trans-thalamic': ['thalamus', 'thalamic', 'cavum septum pellucidum', 'falx', 'biparietal'],
        'Trans-ventricular': ['lateral ventricle', 'choroid plexus', 'atrium', 'ventricular'],
    }

    has_fetal_context = any(kw in response_lower for kw in fetal_keywords)
    category_kws = anatomy_keywords.get(category, [category.lower()])
    has_correct_anatomy = any(kw in response_lower for kw in category_kws)

    word_count = len(response.split())
    if word_count < 5:
        detail_score = 0.2
    elif word_count < 15:
        detail_score = 0.5
    elif word_count < 30:
        detail_score = 0.8
    else:
        detail_score = 1.0

    hallucination_penalty = 0
    wrong_categories = [cat for cat in anatomy_keywords.keys() if cat != category]
    for wrong_cat in wrong_categories:
        wrong_kws = anatomy_keywords.get(wrong_cat, [])
        # Only penalize non-shared keywords to avoid false positives
        unique_kws = [kw for kw in wrong_kws[:2] if kw not in shared_keywords]
        if any(kw in response_lower and category.lower() not in response_lower for kw in unique_kws):
            hallucination_penalty += 0.1

    overall_score = (
        (1.0 if has_fetal_context else 0.0) * 0.3 +
        (1.0 if has_correct_anatomy else 0.0) * 0.4 +
        detail_score * 0.2 +
        max(0, 1.0 - hallucination_penalty) * 0.1
    )

    return {
        'has_fetal_context': has_fetal_context,
        'has_correct_anatomy': has_correct_anatomy,
        'word_count': word_count,
        'detail_score': detail_score,
        'hallucination_penalty': hallucination_penalty,
        'overall_score': overall_score
    }


async def process_single_question(
    model,
    image: Image.Image,
    question: str,
    question_idx: int,
    category: str,
    rate_limiter: RateLimiter
) -> Dict[str, Any]:
    """Process a single question with rate limiting"""
    await rate_limiter.acquire()

    start_time = time.time()
    try:
        response = await model.answer_question(image, question)
        elapsed_time = time.time() - start_time
        evaluation = evaluate_response(response, category, question_idx)

        return {
            'question_idx': question_idx,
            'question': question[:50] + '...' if len(question) > 50 else question,
            'response': response,
            'time': elapsed_time,
            'evaluation': evaluation,
            'error': None
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        # Check for RPD limit errors - raise to stop processing
        # (finally block handles semaphore release, no explicit release here)
        if is_rpd_error(e):
            raise RPDLimitError(f"Rate/quota limit hit: {e}")
        return {
            'question_idx': question_idx,
            'question': question[:50] + '...' if len(question) > 50 else question,
            'response': f"ERROR: {str(e)}",
            'time': elapsed_time,
            'evaluation': {
                'has_fetal_context': False,
                'has_correct_anatomy': False,
                'word_count': 0,
                'detail_score': 0,
                'hallucination_penalty': 0,
                'overall_score': 0
            },
            'error': str(e)
        }
    finally:
        rate_limiter.release()


async def process_single_image(
    model,
    image_path: Path,
    questions: List[str],
    category: str,
    rate_limiter: RateLimiter
) -> Dict[str, Any]:
    """Process all questions for a single image in parallel"""
    with Image.open(image_path) as img:
        image = img.convert('RGB')

    # Create tasks for all questions
    tasks = [
        process_single_question(model, image, question, q_idx, category, rate_limiter)
        for q_idx, question in enumerate(questions)
    ]

    # Run all questions in parallel (rate limiter controls actual concurrency)
    question_results = await asyncio.gather(*tasks)

    return {
        'image': image_path.name,
        'category': category,
        'questions': sorted(question_results, key=lambda x: x['question_idx'])
    }


async def run_parallel_evaluation(
    model,
    model_name: str,
    test_images: List[Dict[str, Any]],
    questions: List[str],
    rate_limiter: RateLimiter,
    output_dir: Path,
    quiet: bool = False,
    no_progress: bool = False,
    logger: Optional[logging.Logger] = None,
    checkpoint_path: Optional[Path] = None,
    completed_images: Optional[Dict[str, Dict]] = None,
    max_requests: int = 0
) -> Dict[str, Any]:
    """Run evaluation with parallel requests and checkpoint support

    Args:
        max_requests: Maximum total requests before stopping (0=unlimited)
    """

    completed_images = completed_images or {}
    requests_per_image = len(questions)
    total_requests_made = 0  # Only count NEW requests this session (not resumed)
    results = list(completed_images.values())  # Start with already completed

    # Filter out already completed images
    remaining_images = [
        img for img in test_images
        if f"{img['category']}/{img['name']}" not in completed_images
    ]

    if logger:
        logger.info(f"Starting: {model_name} ({len(remaining_images)} remaining of {len(test_images)} total)")

    if not remaining_images:
        if logger:
            logger.info("All images already completed from checkpoint")
    else:
        await model.load()

        # Process images with checkpoint saving
        rpd_hit = False
        last_checkpoint_count = len(completed_images)
        pbar = None
        if not no_progress and not quiet:
            pbar = tqdm_asyncio(total=len(test_images), desc=model_name, initial=len(completed_images))

        # Process images in PARALLEL batches
        CHECKPOINT_INTERVAL = 50  # Save checkpoint every N images
        IMAGE_BATCH_SIZE = rate_limiter.max_concurrent  # Process this many images concurrently
        max_requests_hit = False
        stop_processing = False

        async def process_image_wrapper(img):
            """Wrapper to process a single image and return result with metadata"""
            nonlocal total_requests_made, stop_processing

            if stop_processing:
                return None

            try:
                result = await process_single_image(
                    model, img['path'], questions, img['category'], rate_limiter
                )
                return {'img': img, 'result': result, 'error': None}
            except RPDLimitError as e:
                return {'img': img, 'result': None, 'error': ('rpd', e)}
            except Exception as e:
                return {'img': img, 'result': None, 'error': ('other', e)}

        # Process in batches of IMAGE_BATCH_SIZE images concurrently
        for batch_start in range(0, len(remaining_images), IMAGE_BATCH_SIZE):
            if stop_processing:
                break

            # Check max_requests limit before processing batch
            batch_end = min(batch_start + IMAGE_BATCH_SIZE, len(remaining_images))
            batch = remaining_images[batch_start:batch_end]
            batch_requests = len(batch) * requests_per_image

            if max_requests > 0 and total_requests_made + batch_requests > max_requests:
                # Reduce batch size to fit within limit
                remaining_quota = max_requests - total_requests_made
                max_images_in_batch = remaining_quota // requests_per_image
                if max_images_in_batch <= 0:
                    max_requests_hit = True
                    if logger:
                        logger.info(f"MAX REQUESTS LIMIT: Stopping at {total_requests_made} requests (limit: {max_requests})")
                    print(f"\n[MAX REQUESTS] Reached limit of {max_requests} requests ({total_requests_made} made)")
                    break
                batch = batch[:max_images_in_batch]

            # Process batch concurrently
            tasks = [process_image_wrapper(img) for img in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for res in batch_results:
                if res is None:
                    continue
                if isinstance(res, Exception):
                    if logger:
                        logger.error(f"Unexpected error in batch: {res}")
                    if pbar:
                        pbar.update(1)
                    continue

                img = res['img']
                result = res['result']
                error = res['error']

                if error:
                    error_type, error_obj = error
                    if error_type == 'rpd':
                        rpd_hit = True
                        stop_processing = True
                        if logger:
                            logger.error(f"RPD LIMIT HIT: {error_obj}")
                        print(f"\n[RPD LIMIT] {error_obj}")
                    else:
                        if logger:
                            logger.error(f"Error processing {img['category']}/{img['name']}: {error_obj}")
                    if pbar:
                        pbar.update(1)
                    continue

                results.append(result)
                total_requests_made += requests_per_image

                # Track completed
                img_key = f"{img['category']}/{img['name']}"
                completed_images[img_key] = result

                if pbar:
                    pbar.update(1)

            # Save checkpoint after each batch (counter-based to avoid modular arithmetic issues)
            if checkpoint_path and (len(completed_images) - last_checkpoint_count) >= CHECKPOINT_INTERVAL:
                last_checkpoint_count = len(completed_images)
                save_checkpoint(checkpoint_path, completed_images, model_name,
                              {'total_images': len(test_images)}, results)
                if logger:
                    logger.info(f"Checkpoint saved: {len(completed_images)}/{len(test_images)} images ({total_requests_made} requests)")

            if rpd_hit:
                # Save checkpoint immediately on RPD hit
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, completed_images, model_name,
                                  {'total_images': len(test_images)}, results)
                    print(f"[CHECKPOINT] Saved {len(completed_images)}/{len(test_images)} images to {checkpoint_path.name}")
                    print(f"[RESUME] Run with --resume {checkpoint_path.name} to continue")
                break

        if pbar:
            pbar.close()

        # Final checkpoint save
        if checkpoint_path and not rpd_hit:
            save_checkpoint(checkpoint_path, completed_images, model_name,
                          {'total_images': len(test_images)}, results)

        if max_requests_hit and checkpoint_path:
            print(f"[CHECKPOINT] Saved {len(completed_images)}/{len(test_images)} images to {checkpoint_path.name}")
            print(f"[RESUME] Run with --resume {checkpoint_path.name} to continue tomorrow")

    # Calculate metrics from all results (completed + new)
    all_scores = []
    per_category_scores = {}
    per_question_scores = {i: [] for i in range(len(questions))}
    total_time = 0
    error_count = 0

    for result in results:
        category = result['category']
        image_name = result['image']
        if category not in per_category_scores:
            per_category_scores[category] = []

        image_scores = []
        for q_result in result['questions']:
            score = q_result['evaluation']['overall_score']
            all_scores.append(score)
            image_scores.append(score)
            per_category_scores[category].append(score)
            q_idx = q_result['question_idx']
            if q_idx in per_question_scores:
                per_question_scores[q_idx].append(score)
            total_time += q_result['time']

            # Log errors to file
            if q_result.get('error') and logger:
                logger.warning(f"ERROR {category}/{image_name} Q{q_result['question_idx']}: {q_result['error']}")
                error_count += 1

        # Log per-image summary
        if logger:
            avg_score = sum(image_scores) / len(image_scores) if image_scores else 0
            logger.debug(f"{category}/{image_name}: {avg_score:.2%}")

    if logger and error_count > 0:
        logger.warning(f"Total errors: {error_count}")

    question_loader = QuestionLoader(str(project_root / 'data' / 'Fetal Ultrasound'))
    question_names = question_loader.get_question_short_names()

    return {
        'model': model_name,
        'results': results,
        'completed_count': len(completed_images),
        'total_count': len(test_images),
        'metrics': {
            'avg_response_score': sum(all_scores) / len(all_scores) if all_scores else 0,
            'total_images': len(results),
            'total_questions': len(all_scores),
            'total_time': total_time,
            'avg_time_per_question': total_time / len(all_scores) if all_scores else 0,
            'per_category': {
                cat: sum(scores) / len(scores) if scores else 0
                for cat, scores in per_category_scores.items()
            },
            'per_question': {
                question_names[i]: sum(scores) / len(scores) if scores else 0
                for i, scores in per_question_scores.items()
            }
        }
    }


def get_test_images(
    question_loader: QuestionLoader,
    images_per_category: int = 5,
    categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Get test images from the dataset"""
    test_images = []

    available_categories = question_loader.get_categories()
    if categories:
        available_categories = [c for c in categories if c in available_categories]

    for category in available_categories:
        images = question_loader.get_category_images(category)
        for img_path in images[:images_per_category]:
            test_images.append({
                'path': img_path,
                'category': category,
                'name': img_path.name
            })

    return test_images


def get_images_from_split(split: str, data_root: Path) -> List[Dict[str, Any]]:
    """Get images from a dataset split (train/val/test)"""
    splits_data = load_splits()

    if split not in splits_data['splits']:
        raise ValueError(f"Invalid split '{split}'. Must be one of: train, val, test")

    split_data = splits_data['splits'][split]
    test_images = []

    for category, image_paths in split_data.items():
        for rel_path in image_paths:
            img_path = data_root / rel_path
            test_images.append({
                'path': img_path,
                'category': category,
                'name': img_path.name
            })

    return test_images


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Parallel API VLM evaluation with rate limiting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --models openai --openai-model gpt-5.2-chat-latest --max-rpm 60
  %(prog)s --models grok --max-rpm 60 --images-per-category 3
  %(prog)s --models gemini --max-rpm 30 --max-concurrent 5
  %(prog)s --models all --max-rpm 30
        """
    )

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=["openai", "gemini", "grok", "vertex-ai", "vllm", "all"],
        default=["openai"],
        help="Models to evaluate (default: openai)"
    )

    parser.add_argument(
        "--images-per-category", "-n",
        type=int,
        default=3,
        help="Number of images per category to test (default: 3)"
    )

    parser.add_argument(
        "--split", "-s",
        choices=["train", "val", "test"],
        default=None,
        help="Use images from dataset split (overrides --images-per-category)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for results"
    )

    parser.add_argument(
        "--openai-model",
        default="gpt-5.2-chat-latest",
        help="OpenAI model to use (default: gpt-5.2-chat-latest)"
    )

    parser.add_argument(
        "--gemini-model",
        default="gemini-3-flash-preview",
        help="Gemini model to use (default: gemini-3-flash-preview)"
    )

    parser.add_argument(
        "--grok-model",
        default="grok-4",
        help="Grok model to use (default: grok-4)"
    )

    parser.add_argument(
        "--vertex-ai-model",
        default="medgemma-27b-mm-it",
        help="Vertex AI model name (default: medgemma-27b-mm-it)"
    )

    parser.add_argument(
        "--vertex-ai-project",
        default=None,
        help="Vertex AI project ID (default: VERTEX_AI_PROJECT_ID env var)"
    )

    parser.add_argument(
        "--vertex-ai-endpoint",
        default=None,
        help="Vertex AI endpoint ID (default: VERTEX_AI_ENDPOINT_ID env var)"
    )

    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)"
    )

    parser.add_argument(
        "--vllm-model",
        default="google/medgemma-27b-it",
        help="vLLM model name (default: google/medgemma-27b-it)"
    )

    parser.add_argument(
        "--max-rpm",
        type=int,
        default=60,
        help="Maximum requests per minute (default: 60)"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - hide progress bar, show only final result"
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide progress bar"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file (filename in results dir)"
    )

    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Maximum total requests before stopping (0=unlimited). Useful for API quota limits like 10K RPD."
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start index for image list (0-based). For splitting dataset across machines."
    )

    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index for image list (exclusive). If not specified, processes to end."
    )

    return parser.parse_args()


def setup_file_logger(output_dir: Path, timestamp: str) -> logging.Logger:
    """Setup file logging for detailed output (file only, no console)"""
    log_file = output_dir / f'vlm_test_{timestamp}.log'

    logger = logging.getLogger('vlm_test')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Remove any existing handlers
    logger.propagate = False  # Don't propagate to root logger (prevents console output)

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    return logger, log_file


async def main():
    """Main async entry point"""
    args = parse_args()
    quiet = args.quiet

    # Suppress library logging
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='google')
    for logger_name in ['httpx', 'openai', 'google', 'urllib3', 'PIL', 'asyncio',
                        'google.generativeai', 'absl', 'grpc']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Setup output directory
    output_dir = args.output_dir or Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    # Handle resume from checkpoint
    checkpoint_data = None
    if args.resume:
        checkpoint_file = output_dir / args.resume
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            print(f"[RESUME] Loaded checkpoint: {len(checkpoint_data.get('completed_images', {}))} images completed")
        else:
            print(f"[WARN] Checkpoint file not found: {checkpoint_file}")

    # Setup file logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger, log_file = setup_file_logger(output_dir, timestamp)

    # Log configuration
    logger.info(f"Models: {args.models}")
    logger.info(f"Split: {args.split}" if args.split else f"Images per category: {args.images_per_category}")
    logger.info(f"Max RPM: {args.max_rpm}, Concurrent: {args.max_concurrent}")
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")

    # Load questions and images
    data_root = project_root / 'data' / 'Fetal Ultrasound'
    question_loader = QuestionLoader(str(data_root))
    questions = question_loader.get_questions()

    # Get images from split or by count
    if args.split:
        test_images = get_images_from_split(args.split, data_root)
    else:
        test_images = get_test_images(question_loader, args.images_per_category)

    # Apply sharding if specified (for parallel processing across machines)
    total_images_before_shard = len(test_images)
    shard_suffix = ""
    if args.start_index is not None or args.end_index is not None:
        start = args.start_index or 0
        end = args.end_index or len(test_images)
        shard_suffix = f"_shard{start}-{end}"
        print(f"[SHARD] Processing images {start} to {end} (of {total_images_before_shard} total)")
        logger.info(f"Sharding: images {start} to {end} of {total_images_before_shard}")
        test_images = test_images[start:end]

    total_requests = len(test_images) * len(questions)
    logger.info(f"Total: {len(test_images)} images x {len(questions)} questions = {total_requests} requests")

    # Console: just show what we're doing
    if not quiet:
        print(f"{len(test_images)} images, {total_requests} requests")

    # Create rate limiter
    rate_limiter = RateLimiter(max_rpm=args.max_rpm, max_concurrent=args.max_concurrent)

    # Determine which models to run
    run_openai = "all" in args.models or "openai" in args.models
    run_gemini = "all" in args.models or "gemini" in args.models
    run_grok = "all" in args.models or "grok" in args.models
    run_vertex_ai = "all" in args.models or "vertex-ai" in args.models
    run_vllm = "vllm" in args.models  # Not included in "all" - requires local server

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'split': args.split,
            'images_per_category': args.images_per_category if not args.split else None,
            'total_images': len(test_images),
            'max_rpm': args.max_rpm,
            'max_concurrent': args.max_concurrent,
            'questions': question_loader.get_question_short_names()
        },
        'models': {}
    }

    no_progress = args.no_progress

    # Run OpenAI
    if run_openai:
        try:
            # Create checkpoint path for this model
            checkpoint_path = output_dir / f'checkpoint_openai_{args.openai_model.replace("/", "_")}{shard_suffix}.json'
            completed_images = {}
            if checkpoint_data and checkpoint_data.get('model_name', '').startswith('OpenAI'):
                completed_images = checkpoint_data.get('completed_images', {})

            openai_vlm = AsyncOpenAIVLM(model_name=args.openai_model)
            result = await run_parallel_evaluation(
                openai_vlm, f"OpenAI ({args.openai_model})",
                test_images, questions, rate_limiter, output_dir, quiet, no_progress, logger,
                checkpoint_path=checkpoint_path, completed_images=completed_images,
                max_requests=args.max_requests
            )
            all_results['models'][f'OpenAI ({args.openai_model})'] = result
        except Exception as e:
            logger.error(f"OpenAI failed: {e}")
            print(f"[SKIP] OpenAI: {e}")

    # Run Grok
    if run_grok:
        try:
            checkpoint_path = output_dir / f'checkpoint_grok_{args.grok_model.replace("/", "_")}{shard_suffix}.json'
            completed_images = {}
            if checkpoint_data and checkpoint_data.get('model_name', '').startswith('Grok'):
                completed_images = checkpoint_data.get('completed_images', {})

            grok = AsyncGrokVLM(model_name=args.grok_model)
            result = await run_parallel_evaluation(
                grok, f"Grok ({args.grok_model})",
                test_images, questions, rate_limiter, output_dir, quiet, no_progress, logger,
                checkpoint_path=checkpoint_path, completed_images=completed_images,
                max_requests=args.max_requests
            )
            all_results['models'][f'Grok ({args.grok_model})'] = result
        except Exception as e:
            logger.error(f"Grok failed: {e}")
            print(f"[SKIP] Grok: {e}")

    # Run Gemini
    if run_gemini:
        try:
            checkpoint_path = output_dir / f'checkpoint_gemini_{args.gemini_model.replace("/", "_")}{shard_suffix}.json'
            completed_images = {}
            if checkpoint_data and checkpoint_data.get('model_name', '').startswith('Gemini'):
                completed_images = checkpoint_data.get('completed_images', {})

            gemini = AsyncGeminiVLM(model_name=args.gemini_model)
            result = await run_parallel_evaluation(
                gemini, f"Gemini ({args.gemini_model})",
                test_images, questions, rate_limiter, output_dir, quiet, no_progress, logger,
                checkpoint_path=checkpoint_path, completed_images=completed_images,
                max_requests=args.max_requests
            )
            all_results['models'][f'Gemini ({args.gemini_model})'] = result
        except Exception as e:
            logger.error(f"Gemini failed: {e}")
            print(f"[SKIP] Gemini: {e}")

    # Run Vertex AI
    if run_vertex_ai:
        vertex_ai = None
        try:
            checkpoint_path = output_dir / f'checkpoint_vertex_ai_{args.vertex_ai_model.replace("/", "_")}{shard_suffix}.json'
            completed_images = {}
            if checkpoint_data and checkpoint_data.get('model_name', '').startswith('VertexAI'):
                completed_images = checkpoint_data.get('completed_images', {})

            vertex_ai = AsyncVertexAIVLM(
                model_name=args.vertex_ai_model,
                project_id=args.vertex_ai_project,
                endpoint_id=args.vertex_ai_endpoint
            )
            result = await run_parallel_evaluation(
                vertex_ai, f"VertexAI ({args.vertex_ai_model})",
                test_images, questions, rate_limiter, output_dir, quiet, no_progress, logger,
                checkpoint_path=checkpoint_path, completed_images=completed_images,
                max_requests=args.max_requests
            )
            all_results['models'][f'VertexAI ({args.vertex_ai_model})'] = result
        except Exception as e:
            logger.error(f"Vertex AI failed: {e}")
            print(f"[SKIP] Vertex AI: {e}")
        finally:
            # Clean up aiohttp session even on error
            if vertex_ai is not None:
                await vertex_ai.close()

    # Run vLLM (local server)
    if run_vllm:
        try:
            checkpoint_path = output_dir / f'checkpoint_vllm_{args.vllm_model.replace("/", "_")}{shard_suffix}.json'
            completed_images = {}
            if checkpoint_data and checkpoint_data.get('model_name', '').startswith('vLLM'):
                completed_images = checkpoint_data.get('completed_images', {})

            # Health check: verify vLLM server is up and serving the expected model
            import aiohttp
            health_url = args.vllm_url.rstrip('/') + '/v1/models'
            try:
                async with aiohttp.ClientSession() as _hc_session:
                    async with _hc_session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as _hc_resp:
                        _hc_data = await _hc_resp.json()
                        _model_ids = [m.get('id', '') for m in _hc_data.get('data', [])]
                        if not any(args.vllm_model in mid for mid in _model_ids):
                            print(f"ERROR: vLLM server at {args.vllm_url} does not have model '{args.vllm_model}'. "
                                  f"Found: {_model_ids}")
                            import sys as _sys
                            _sys.exit(1)
            except Exception as _hc_err:
                print(f"ERROR: vLLM health check failed for {health_url}: {_hc_err}")
                import sys as _sys
                _sys.exit(1)

            vllm = AsyncVLLM(base_url=args.vllm_url, model_name=args.vllm_model)
            result = await run_parallel_evaluation(
                vllm, f"vLLM ({args.vllm_model})",
                test_images, questions, rate_limiter, output_dir, quiet, no_progress, logger,
                checkpoint_path=checkpoint_path, completed_images=completed_images,
                max_requests=args.max_requests
            )
            all_results['models'][f'vLLM ({args.vllm_model})'] = result
        except Exception as e:
            logger.error(f"vLLM failed: {e}")
            print(f"[SKIP] vLLM: {e}")

    # Save results JSON
    output_file = output_dir / f'vlm_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Log and print summary
    for model_name, model_data in all_results['models'].items():
        if 'error' in model_data:
            logger.error(f"{model_name}: ERROR - {model_data['error']}")
            print(f"{model_name}: ERROR")
        else:
            metrics = model_data['metrics']
            elapsed = metrics['total_time']
            rpm = (metrics['total_questions'] / elapsed * 60) if elapsed > 0 else 0
            summary = f"{model_name}: {metrics['avg_response_score']:.2%} ({metrics['total_questions']} in {elapsed:.0f}s, {rpm:.1f} RPM)"
            logger.info(summary)
            print(summary)

            # Log detailed breakdown to file only
            for cat, score in sorted(metrics['per_category'].items()):
                logger.debug(f"  {cat}: {score:.2%}")
            for q_name, score in metrics['per_question'].items():
                logger.debug(f"  Q:{q_name}: {score:.2%}")

    logger.info(f"Results: {output_file}")
    logger.info(f"Log: {log_file}")
    print(f"Log: {log_file.name}")


if __name__ == '__main__':
    asyncio.run(main())
