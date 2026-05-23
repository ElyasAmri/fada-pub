"""
API Client Utilities
Shared retry logic and error handling for API calls
"""

import time
import logging
from typing import Callable, TypeVar, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class APIError(Exception):
    """Base exception for API errors"""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded"""
    pass


class TimeoutError(APIError):
    """Request timed out"""
    pass


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff delay.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds

    Returns:
        Delay in seconds
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


def call_with_retry(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
) -> T:
    """
    Call a function with retry logic and exponential backoff.

    Args:
        func: Function to call (should be a callable with no arguments, use lambda/partial)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called on each retry with (attempt, exception)

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return func()
        except retryable_exceptions as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = exponential_backoff(attempt, base_delay)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                if on_retry:
                    on_retry(attempt, e)
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")

    if last_error:
        raise last_error
    raise RuntimeError("Unexpected state: no error but no result")


def retry_decorator(
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: tuple = (Exception,)
):
    """
    Decorator for adding retry logic to functions.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        retryable_exceptions: Tuple of exception types to retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return call_with_retry(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                base_delay=base_delay,
                retryable_exceptions=retryable_exceptions
            )
        return wrapper
    return decorator
