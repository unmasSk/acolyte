"""Type stubs for reindex_service.py"""

from typing import List, Dict, Any, Optional
import asyncio

from acolyte.core.tracing import MetricsCollector
from acolyte.core.secure_config import Settings
from acolyte.core.events import CacheInvalidateEvent, Event
from acolyte.services.indexing_service import IndexingService

class ReindexService:
    metrics: MetricsCollector
    config: Settings
    indexing_service: IndexingService
    max_reindex_files: int
    batch_size: int
    pattern_cooldown: int
    _recent_patterns: Dict[str, float]
    _pattern_lock: asyncio.Lock
    _reindex_queue: asyncio.Queue[CacheInvalidateEvent]
    _queue_processor_task: Optional[asyncio.Task]
    _shutdown_event: asyncio.Event
    _unsubscribe: Any

    def __init__(self, indexing_service: Optional[IndexingService] = None) -> None: ...
    def _start_queue_processor(self) -> None: ...
    async def _process_queue(self) -> None: ...
    async def _handle_cache_invalidation(self, event: Event) -> None: ...
    async def _should_process_pattern(self, pattern: str) -> bool: ...
    async def _process_reindex_event(self, event: CacheInvalidateEvent) -> None: ...
    async def _reindex_in_batches(self, files: List[str], event: CacheInvalidateEvent) -> None: ...
    async def _filter_files_needing_reindex(self, files: List[str]) -> List[str]: ...
    def _find_files_cached(self, pattern: str, project_root: str) -> List[str]: ...
    async def _find_files_matching_pattern(self, pattern: str) -> List[str]: ...
    async def shutdown(self) -> None: ...
    def get_stats(self) -> Dict[str, Any]: ...
