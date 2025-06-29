from typing import TypeVar, Callable, Optional, Type, Tuple, Any, Awaitable
import asyncio
from functools import wraps

T = TypeVar('T')


async def retry_async(
    func: Callable[..., Awaitable[T]],
    max_attempts: int = 3,
    backoff: str = "exponential",
    initial_delay: float = 0.5,
    max_delay: float = 30.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
    logger: Optional[Any] = None,
) -> T:
    """
    Retry an async function with configurable backoff.

    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        backoff: "exponential", "linear", or "constant"
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        retry_on: Tuple of exceptions to retry on
        logger: Optional logger for retry attempts

    Returns:
        Result from successful function call

    Raises:
        Last exception if all retries fail
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            return await func()
        except retry_on as e:
            last_exception = e

            if attempt < max_attempts - 1:
                # Calculate delay based on backoff strategy
                if backoff == "exponential":
                    delay = min(initial_delay * (2**attempt), max_delay)
                elif backoff == "linear":
                    delay = min(initial_delay * (attempt + 1), max_delay)
                else:  # constant
                    delay = initial_delay

                if logger:
                    logger.warning(
                        "Retry attempt failed",
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        delay=delay,
                        error=str(e),
                    )

                await asyncio.sleep(delay)
            else:
                if logger:
                    logger.error("All retry attempts failed", attempts=max_attempts, error=str(e))

    # Should never be None if we reach here, but satisfy type checker
    if last_exception is None:
        raise RuntimeError("No exception captured but all attempts failed")
    raise last_exception


# Decorator version
def with_retry(max_attempts: int = 3, backoff: str = "exponential", **kwargs):
    """Decorator to add retry logic to async functions."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **func_kwargs):
            return await retry_async(
                lambda: func(*args, **func_kwargs),
                max_attempts=max_attempts,
                backoff=backoff,
                **kwargs,
            )

        return wrapper

    return decorator
