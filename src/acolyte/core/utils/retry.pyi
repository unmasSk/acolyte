from typing import TypeVar, Callable, Optional, Type, Tuple, Any, Awaitable

T = TypeVar('T')

async def retry_async(
    func: Callable[..., Awaitable[T]],
    max_attempts: int = 3,
    backoff: str = "exponential",
    initial_delay: float = 0.5,
    max_delay: float = 30.0,
    retry_on: Tuple[Type[Exception], ...] = ...,
    logger: Optional[Any] = None,
) -> T: ...
def with_retry(
    max_attempts: int = 3, backoff: str = "exponential", **kwargs: Any
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]: ...
