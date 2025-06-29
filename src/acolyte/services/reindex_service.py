"""
Reindex Service - Automatic Re-indexing System.

Dedicated service for handling automatic re-indexation of files.
Separates re-indexing concerns from core indexing functionality.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import asyncio
import time

from acolyte.core.logging import logger
from acolyte.core.tracing import MetricsCollector
from acolyte.core.secure_config import Settings
from acolyte.core.events import event_bus, EventType, CacheInvalidateEvent, Event
from acolyte.core.id_generator import generate_id

# Import IndexingService for dependency injection
from acolyte.services.indexing_service import IndexingService


class ReindexService:
    """
    Service dedicated to automatic file re-indexation.

    Responsibilities:
    - Listen for cache invalidation events
    - Find files matching patterns
    - Coordinate with IndexingService to re-index
    - Manage re-indexing queue (future)
    - Track re-indexing specific metrics
    - Implement intelligent deduplication and batching
    """

    def __init__(self, indexing_service: Optional[IndexingService] = None):
        """
        Initialize ReindexService.

        Args:
            indexing_service: Optional IndexingService instance for dependency injection.
                            If not provided, a new instance will be created.
        """
        self.metrics = MetricsCollector()
        self.config = Settings()

        # Dependency injection for better testability
        self.indexing_service = indexing_service or IndexingService()

        # Re-indexing configuration
        self.max_reindex_files = self.config.get("indexing.max_reindex_files", 50)
        self.batch_size = self.config.get("reindexing.batch_size", 5)
        self.pattern_cooldown = self.config.get("reindexing.pattern_cooldown", 5)  # seconds

        # Pattern deduplication
        self._recent_patterns: Dict[str, float] = {}  # pattern -> timestamp
        self._pattern_lock = asyncio.Lock()

        # Re-indexing queue for future implementation
        self._reindex_queue: asyncio.Queue[CacheInvalidateEvent] = asyncio.Queue()
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Start queue processor
        self._start_queue_processor()

        # Subscribe to cache invalidation events
        self._unsubscribe = event_bus.subscribe(
            EventType.CACHE_INVALIDATE,
            self._handle_cache_invalidation,
            filter=lambda e: isinstance(e, CacheInvalidateEvent) and e.target_service == "indexing",
        )

        logger.info(
            "ReindexService initialized",
            max_files=self.max_reindex_files,
            batch_size=self.batch_size,
            cooldown=self.pattern_cooldown,
        )

    def _start_queue_processor(self):
        """Start the background queue processor task."""
        self._queue_processor_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """
        Process re-indexing queue in the background.

        This allows for better control over re-indexing operations:
        - Batching multiple related events
        - Rate limiting
        - Priority handling (future)
        """
        logger.info("ReindexService queue processor started")

        while not self._shutdown_event.is_set():
            try:
                # Wait for event with timeout to allow shutdown checks
                event = await asyncio.wait_for(self._reindex_queue.get(), timeout=1.0)

                # Process the event
                await self._process_reindex_event(event)

            except asyncio.TimeoutError:
                # Normal timeout, continue loop
                continue
            except Exception as e:
                logger.error("Error processing reindex queue", error=str(e))
                self.metrics.increment("reindexing.queue_errors")

    async def _handle_cache_invalidation(self, event: Event):
        """
        Handle cache invalidation events to trigger re-indexing.

        This is called when:
        1. GitService detects changes (after pull, checkout, etc.)
        2. GitService publishes CacheInvalidateEvent
        3. This handler queues files for re-indexing

        Args:
            event: Event with file pattern to re-index
        """
        if not isinstance(event, CacheInvalidateEvent):
            return

        try:
            logger.info(
                "Cache invalidation received for re-indexing",
                source=event.source,
                pattern=event.key_pattern,
                reason=event.reason,
            )

            # Check if we should process this pattern (deduplication)
            should_process = await self._should_process_pattern(event.key_pattern)
            if not should_process:
                logger.info(
                    "Skipping duplicate pattern within cooldown",
                    pattern=event.key_pattern,
                    cooldown=self.pattern_cooldown,
                )
                self.metrics.increment("reindexing.patterns_deduplicated")
                return

            # Queue the event for processing
            await self._reindex_queue.put(event)
            self.metrics.increment("reindexing.events_queued")
            logger.debug("Queued event for re-indexing", pattern=event.key_pattern)

        except Exception as e:
            logger.error("Error handling cache invalidation", error=str(e))
            self.metrics.increment("reindexing.handler_errors")

    async def _should_process_pattern(self, pattern: str) -> bool:
        """
        Check if a pattern should be processed based on cooldown.

        Prevents rapid re-indexing of the same pattern.

        Args:
            pattern: File pattern to check

        Returns:
            True if pattern should be processed, False if within cooldown
        """
        async with self._pattern_lock:
            now = time.time()

            # Clean old entries
            self._recent_patterns = {
                p: ts for p, ts in self._recent_patterns.items() if now - ts < self.pattern_cooldown
            }

            # Check if pattern was recently processed
            if pattern in self._recent_patterns:
                return False

            # Mark pattern as processing
            self._recent_patterns[pattern] = now
            return True

    async def _process_reindex_event(self, event: CacheInvalidateEvent):
        """
        Process a single re-index event from the queue.

        Args:
            event: Cache invalidation event to process
        """
        try:
            # Avoid re-indexing if indexing is already in progress
            if self.indexing_service._is_indexing:
                logger.warning(
                    "Skipping re-indexing - indexing already in progress",
                    pattern=event.key_pattern,
                )
                self.metrics.increment("reindexing.skipped_busy")
                # Re-queue the event for later
                await self._reindex_queue.put(event)
                return

            # Extract files from pattern
            if event.key_pattern and event.key_pattern != "*":
                # Pattern can be something like "*auth.py*"
                file_hint = event.key_pattern.strip("*")

                if file_hint:
                    # Find files matching the pattern
                    matching_files = await self._find_files_matching_pattern(file_hint)

                    # Check timestamp optimization
                    files_to_reindex = await self._filter_files_needing_reindex(matching_files)

                    if not files_to_reindex:
                        logger.info(
                            "No files need re-indexing (timestamps unchanged)",
                            pattern=file_hint,
                            total_matches=len(matching_files),
                        )
                        self.metrics.increment("reindexing.skipped_unchanged")
                        return

                    # Apply file limit
                    if len(files_to_reindex) > self.max_reindex_files:
                        logger.warning(
                            "Too many files match pattern, limiting re-indexing",
                            pattern=file_hint,
                            total_matches=len(files_to_reindex),
                            limit=self.max_reindex_files,
                        )
                        files_to_reindex = files_to_reindex[: self.max_reindex_files]

                    if files_to_reindex:
                        # Process in batches
                        await self._reindex_in_batches(files_to_reindex, event)
                    else:
                        logger.debug("No files found matching pattern", pattern=file_hint)
            else:
                logger.debug(
                    "Cache invalidation with wildcard pattern - skipping",
                    pattern=event.key_pattern,
                )

            self.metrics.increment("reindexing.events_processed")

        except Exception as e:
            logger.error(
                "Error processing re-index event",
                pattern=event.key_pattern,
                error=str(e),
            )
            self.metrics.increment("reindexing.processing_errors")

    async def _reindex_in_batches(self, files: List[str], event: CacheInvalidateEvent):
        """
        Re-index files in smaller batches to avoid blocking.

        Args:
            files: List of file paths to re-index
            event: Original cache invalidation event
        """
        total_files = len(files)
        total_chunks = 0
        total_duration = 0.0

        # Generate task ID for tracking
        task_id = f"reinx_{int(time.time())}_{generate_id()[:8]}"

        logger.info(
            "Starting batch re-indexing",
            task_id=task_id,
            total_files=total_files,
            batch_size=self.batch_size,
            reason=event.reason,
        )

        # Process in batches
        for i in range(0, total_files, self.batch_size):
            batch = files[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_files + self.batch_size - 1) // self.batch_size

            try:
                logger.debug(
                    "Processing re-index batch",
                    task_id=task_id,
                    batch=f"{batch_num}/{total_batches}",
                    files_count=len(batch),
                )

                # Call indexing service
                result = await self.indexing_service.index_files(
                    files=batch,
                    trigger="cache_invalidation",
                    task_id=f"{task_id}_b{batch_num}",
                )

                # Accumulate results
                if result["status"] in ["success", "partial"]:
                    total_chunks += result.get("chunks_created", 0)
                    total_duration += result.get("duration_seconds", 0)

                # Small delay between batches to avoid overwhelming the system
                if i + self.batch_size < total_files:
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(
                    "Batch re-indexing failed",
                    task_id=task_id,
                    batch=f"{batch_num}/{total_batches}",
                    error=str(e),
                )
                self.metrics.increment("reindexing.batch_errors")

        # Log summary
        logger.info(
            "Batch re-indexing completed",
            task_id=task_id,
            total_files=total_files,
            total_chunks=total_chunks,
            total_duration=f"{total_duration:.2f}s",
            avg_time_per_file=f"{total_duration/max(total_files, 1):.2f}s",
        )

        # Update metrics
        self.metrics.increment("reindexing.completed")
        self.metrics.gauge("reindexing.last_files_count", total_files)
        self.metrics.gauge("reindexing.last_chunks_created", total_chunks)
        self.metrics.record("reindexing.total_duration_ms", total_duration * 1000)

    async def _filter_files_needing_reindex(self, files: List[str]) -> List[str]:
        """
        Filter files that actually need re-indexing based on timestamps.

        TODO: Implement actual timestamp comparison with Weaviate.
        For now, returns all files.

        Args:
            files: List of file paths to check

        Returns:
            List of files that need re-indexing
        """
        # TODO: Implement:
        # 1. Query Weaviate for indexed_at timestamps
        # 2. Compare with file mtime using Path(file).stat().st_mtime
        # 3. Only return files where mtime > indexed_at
        # This will significantly reduce unnecessary re-indexing

        # For now, return all files (conservative approach)
        return files

    async def _find_files_matching_pattern(self, pattern: str) -> List[str]:
        """
        Find files in the project that match the given pattern.

        Args:
            pattern: File pattern to search (e.g., "auth.py" from "*auth.py*")

        Returns:
            List of absolute file paths that match the pattern and are supported
        """
        try:
            project_root = Path(self.config.get("project.path", ".")).resolve()
            matching_files = []

            # Check cache first (convert to sync for lru_cache)
            # Note: We don't use cache for now to ensure fresh results

            # Optimize: if pattern looks like a full filename, search more specifically
            if "." in pattern and not pattern.startswith("."):
                # Pattern like "auth.py" - search for exact filename match
                filename_pattern = pattern.lower()

                # Search for exact filename matches first
                for file_path in project_root.rglob(f"*{pattern}"):
                    if file_path.is_file() and self.indexing_service.is_supported_file(file_path):
                        if not self.indexing_service.should_ignore(str(file_path)):
                            matching_files.append(str(file_path))

                # If no exact matches, try partial matches
                if not matching_files:
                    for file_path in project_root.rglob("*"):
                        if file_path.is_file() and filename_pattern in file_path.name.lower():
                            if self.indexing_service.is_supported_file(
                                file_path
                            ) and not self.indexing_service.should_ignore(str(file_path)):
                                matching_files.append(str(file_path))
            else:
                # Pattern is partial (like "auth" from "*auth*")
                # Search for files containing the pattern in their name or path
                for file_path in project_root.rglob("*"):
                    if file_path.is_file():
                        # Check both filename and full path
                        file_str = str(file_path).lower()
                        filename = file_path.name.lower()

                        if pattern.lower() in filename or pattern.lower() in file_str:
                            if self.indexing_service.is_supported_file(
                                file_path
                            ) and not self.indexing_service.should_ignore(str(file_path)):
                                matching_files.append(str(file_path))

            # Remove duplicates while preserving order
            seen: Set[str] = set()
            unique_files = []
            for f in matching_files:
                if f not in seen:
                    seen.add(f)
                    unique_files.append(f)

            logger.info(
                "Found files matching pattern",
                pattern=pattern,
                files_count=len(unique_files),
                sample_files=unique_files[:5] if unique_files else [],
            )

            self.metrics.gauge("reindexing.pattern_matches", len(unique_files))

            return unique_files

        except Exception as e:
            logger.error(
                "Failed to find files matching pattern",
                pattern=pattern,
                error=str(e),
            )
            self.metrics.increment("reindexing.pattern_search_errors")
            return []

    async def shutdown(self):
        """
        Gracefully shutdown the service.

        - Unsubscribe from events
        - Stop queue processor
        - Process remaining items in queue
        """
        logger.info("Shutting down ReindexService")

        # Unsubscribe from events
        if self._unsubscribe:
            self._unsubscribe()

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for queue processor to finish
        if self._queue_processor_task:
            try:
                await asyncio.wait_for(self._queue_processor_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Queue processor did not shutdown gracefully")
                self._queue_processor_task.cancel()

        # Process remaining items in queue (best effort)
        remaining = self._reindex_queue.qsize()
        if remaining > 0:
            logger.warning(
                "ReindexService shutting down with items in queue",
                remaining_items=remaining,
            )

        logger.info("ReindexService shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get re-indexing statistics.

        Returns:
            Dictionary with re-indexing metrics
        """
        return {
            "queue_size": self._reindex_queue.qsize(),
            "recent_patterns": len(self._recent_patterns),
            "pattern_cooldown_seconds": self.pattern_cooldown,
            "max_reindex_files": self.max_reindex_files,
            "batch_size": self.batch_size,
        }
