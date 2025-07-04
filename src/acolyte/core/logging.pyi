from typing import Any, Optional, List, Pattern, Iterator
from queue import Queue
from logging.handlers import QueueHandler
from contextlib import contextmanager

class AsyncLogger:
    component: str
    debug_mode: bool
    queue: Queue[Any]
    handler: QueueHandler

    def __init__(self, component: str, debug_mode: bool = False) -> None: ...
    def log(self, level: str, message: str, **context: Any) -> None: ...
    def debug(self, message: str, **context: Any) -> None: ...
    def info(self, message: str, **context: Any) -> None: ...
    def warning(self, message: str, **context: Any) -> None: ...
    def error(self, message: str, include_trace: Optional[bool] = None, **context: Any) -> None: ...

class SensitiveDataMasker:
    patterns: List[Pattern[str]]

    def __init__(self, patterns: Optional[List[Pattern[str]]] = None) -> None: ...
    def mask(self, text: str) -> str: ...

class PerformanceLogger:
    logger: AsyncLogger

    def __init__(self) -> None: ...
    @contextmanager
    def measure(self, operation: str, **context: Any) -> Iterator[None]: ...

logger: AsyncLogger
