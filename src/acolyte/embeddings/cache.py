"""
Cache with context for embeddings.

Implements an LRU cache with TTL that considers the complete context
of the code to avoid collisions and improve relevance.
"""

from dataclasses import dataclass
import hashlib
import time
from collections import OrderedDict
from typing import Optional, Union, TypedDict
from acolyte.core.logging import logger
from acolyte.embeddings.types import EmbeddingVector
from acolyte.embeddings.context import RichCodeContext
from acolyte.models.chunk import Chunk


class CacheStats(TypedDict):
    """TypedDict for cache statistics."""

    size: int
    max_size: int
    ttl_seconds: float
    oldest_entry_age: float
    newest_entry_age: float
    capacity_used: float
    cleanup_interval: int
    operations_until_cleanup: int


@dataclass
class CacheEntry:
    """Cache entry with TTL.

    Attributes:
        embedding: Stored embedding vector
        created_at: Creation timestamp (for TTL)
    """

    embedding: EmbeddingVector
    created_at: float


class ContextAwareCache:
    """Cache LRU with TTL that considers the complete context.

    The cache generates unique keys considering not only the text
    but also the context (language, imports, tags, etc.)
    to avoid collisions between the same code in different contexts.

    Uses OrderedDict for real O(1) LRU eviction.

    Attributes:
        max_size: Maximum number of entries in cache
        ttl_seconds: Time to live in seconds for each entry
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: float = 3600):
        """Initializes the cache with configurable limits.

        Args:
            max_size: Maximum number of entries (default: 10000)
            ttl_seconds: TTL in seconds (can be float, default: 3600 = 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Automatic periodic cleanup
        self._operations_count = 0
        self._cleanup_interval = 100  # Clean every 100 set() operations

        logger.info("ContextAwareCache initialized", max_size=max_size, ttl_seconds=ttl_seconds)

    def _generate_key(self, text: Union[str, Chunk], context: Optional[RichCodeContext]) -> str:
        """Generates a unique key considering text and context.

        Args:
            text: Text of the code or a Chunk object
            context: Optional context with additional metadata

        Returns:
            SHA256 hash as hexadecimal string
        """
        # Determine the base key based on the type
        if isinstance(text, Chunk):
            # For Chunks, use their unique ID that is stable
            text_key = f"chunk_{text.id}"
        else:
            # For strings, use the content directly
            text_key = text

        # If there is no context, hash only the text
        if context is None:
            return hashlib.sha256(text_key.encode()).hexdigest()

        # Key includes important context elements
        context_elements = [
            context.language,
            context.file_path,
            ",".join(sorted(context.imports)) if context.imports else "",
            ",".join(sorted(context.semantic_tags)) if context.semantic_tags else "",
        ]
        context_str = "|".join(context_elements)
        combined = f"{text_key}||{context_str}"

        return hashlib.sha256(combined.encode()).hexdigest()

    def get(
        self, text: Union[str, Chunk], context: Optional[RichCodeContext]
    ) -> Optional[EmbeddingVector]:
        """Gets embedding from cache if it exists and has not expired.

        Args:
            text: Text to search
            context: Context used to generate the key

        Returns:
            EmbeddingVector if it exists and is valid, None if it does not exist or expired
        """
        key = self._generate_key(text, context)

        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check TTL
        if time.time() - entry.created_at > self.ttl_seconds:
            # Expired, delete
            del self._cache[key]
            return None

        # Move to the end for LRU (most recently used)
        self._cache.move_to_end(key)
        return entry.embedding

    def set(
        self,
        text: Union[str, Chunk],
        context: Optional[RichCodeContext],
        embedding: EmbeddingVector,
    ):
        """Saves embedding in cache with LRU eviction if necessary.

        Args:
            text: Original text
            context: Context used
            embedding: Vector to save
        """
        key = self._generate_key(text, context)

        # If the cache is full and the key does not exist, do LRU eviction O(1)
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Delete the first element (least recently used)
            lru_key, _ = self._cache.popitem(last=False)

        # Create new entry (added automatically to the end)
        self._cache[key] = CacheEntry(embedding=embedding, created_at=time.time())

        # Automatic periodic cleanup every N operations
        self._operations_count += 1
        if self._operations_count >= self._cleanup_interval:
            self._operations_count = 0
            self.cleanup_expired()

    def clear(self):
        """Clears the cache."""
        size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared - removed {size} entries")

    def cleanup_expired(self):
        """Deletes expired entries from the cache.

        Automatically called every _cleanup_interval set() operations.
        Can also be called manually if necessary.

        Returns:
            Number of deleted entries
        """
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self._cache.items()
            if current_time - entry.created_at > self.ttl_seconds
        ]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.info("Cleaned up expired cache entries", count=len(expired_keys))

        return len(expired_keys)

    @property
    def size(self) -> int:
        """Current number of entries in cache."""
        return len(self._cache)

    def get_stats(self) -> CacheStats:
        """Returns cache statistics.

        Returns:
            CacheStats with all cache metrics
        """
        if not self._cache:
            return {
                "size": 0,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "oldest_entry_age": 0,
                "newest_entry_age": 0,
                "capacity_used": 0.0,
                "cleanup_interval": self._cleanup_interval,
                "operations_until_cleanup": self._cleanup_interval - self._operations_count,
            }

        current_time = time.time()
        ages = [current_time - entry.created_at for entry in self._cache.values()]

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "oldest_entry_age": max(ages) if ages else 0,
            "newest_entry_age": min(ages) if ages else 0,
            "capacity_used": len(self._cache) / self.max_size,
            "cleanup_interval": self._cleanup_interval,
            "operations_until_cleanup": self._cleanup_interval - self._operations_count,
        }
