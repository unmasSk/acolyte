from typing import List, Optional, Callable, Awaitable, Union
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from fastapi import WebSocket

class EventType(Enum):
    PROGRESS = "progress"
    LOG = "log"
    STATUS = "status"
    ERROR = "error"
    INSIGHT = "insight"
    OPTIMIZATION_NEEDED = "optimization_needed"
    CACHE_INVALIDATE = "cache_invalidate"

class Event(ABC):
    id: str
    timestamp: datetime
    type: EventType
    source: str

    def __init__(self) -> None: ...
    @abstractmethod
    def to_json(self) -> str: ...
    @abstractmethod
    def validate(self) -> bool: ...

class CacheInvalidateEvent(Event):
    type: EventType
    source: str
    target_service: str
    key_pattern: str
    reason: str

    def __init__(
        self, source: str, target_service: str, key_pattern: str = ..., reason: str = ...
    ) -> None: ...
    def to_json(self) -> str: ...
    def validate(self) -> bool: ...

class ProgressEvent(Event):
    type: EventType
    source: str
    operation: str
    current: int
    total: int
    percentage: float
    message: str
    task_id: Optional[str]
    files_skipped: int
    chunks_created: int
    embeddings_generated: int
    errors: int
    current_file: Optional[str]

    def __init__(
        self,
        source: str,
        operation: str,
        current: int,
        total: int,
        message: str = ...,
        task_id: Optional[str] = None,
        files_skipped: int = ...,
        chunks_created: int = ...,
        embeddings_generated: int = ...,
        errors: int = ...,
        current_file: Optional[str] = None,
    ) -> None: ...
    def to_json(self) -> str: ...
    def validate(self) -> bool: ...

class EventBus:
    def __init__(self) -> None: ...
    async def publish(self, event: Event) -> None: ...
    def subscribe(
        self,
        event_type: EventType,
        handler: Union[Callable[[Event], None], Callable[[Event], Awaitable[None]]],
        filter: Optional[Callable[[Event], bool]] = None,
    ) -> Callable[[], None]: ...
    async def replay(
        self,
        from_timestamp: datetime,
        to_timestamp: Optional[datetime] = None,
        event_types: Optional[List[EventType]] = None,
    ) -> List[Event]: ...

class WebSocketManager:
    def __init__(self) -> None: ...
    async def connect(self, websocket: WebSocket) -> str: ...
    async def send_event(self, event: Event) -> None: ...
    async def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...

event_bus: EventBus
