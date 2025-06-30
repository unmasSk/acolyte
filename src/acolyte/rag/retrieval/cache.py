"""
Simple LRU cache for search results.

Provides caching for frequently used search queries to improve performance.
Mono-user system, so cache can be aggressive.
"""

import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

from acolyte.core.logging import logger
from acolyte.core.tracing import MetricsCollector
from acolyte.models.chunk import Chunk


class SearchCache:
    """
    LRU (Least Recently Used) cache for search results.

    Simple implementation optimized for mono-user local usage.
    Thread-safe not required since ACOLYTE is single-user.
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached queries
            ttl: Time to live in seconds (default 1 hour)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.metrics = MetricsCollector()

        logger.info("SearchCache initialized", max_size=max_size, ttl=ttl)

    def _hash_query(self, query: str, filters: Optional[Dict] = None) -> str:
        """
        Generate cache key from query and filters.

        Args:
            query: Search query
            filters: Optional filters applied

        Returns:
            MD5 hash as cache key
        """
        # Combine query and filters for unique key
        cache_input = f"{query}"
        if filters:
            # Sort keys for consistent hashing
            sorted_filters = sorted(filters.items())
            cache_input += f"|{sorted_filters}"

        return hashlib.md5(cache_input.encode()).hexdigest()

    def get(self, query: str, filters: Optional[Dict] = None) -> Optional[List[Chunk]]:
        """
        Get cached results for query.

        Args:
            query: Search query
            filters: Optional filters

        Returns:
            Cached chunks or None if not found/expired
        """
        key = self._hash_query(query, filters)

        if key not in self.cache:
            self.metrics.increment("rag.retrieval.cache.misses")
            return None

        results, timestamp = self.cache[key]

        # Check if expired
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            self.metrics.increment("rag.retrieval.cache.misses")
            self.metrics.increment("rag.retrieval.cache.expirations")
            logger.debug("Cache expired for query", query=query[:50])
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.metrics.increment("rag.retrieval.cache.hits")

        logger.debug(
            "Cache hit", query=query[:50], results=len(results), hit_rate=self.get_hit_rate()
        )

        return results

    def set(self, query: str, results: List[Chunk], filters: Optional[Dict] = None):
        """
        Cache search results.

        Args:
            query: Search query
            results: Chunks to cache
            filters: Optional filters applied
        """
        key = self._hash_query(query, filters)

        # Add to cache
        self.cache[key] = (results, time.time())
        self.cache.move_to_end(key)

        # Evict oldest if over limit
        if len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.metrics.increment("rag.retrieval.cache.evictions")
            logger.debug("Evicted oldest cache entry", size=len(self.cache))

        self.metrics.gauge("rag.retrieval.cache.size", len(self.cache))
        logger.debug(
            "Cached results", query=query[:50], results=len(results), cache_size=len(self.cache)
        )

    def invalidate_by_pattern(self, pattern: str):
        """
        Invalidate cache entries matching pattern.

        Useful when files are modified and cached results become stale.

        Args:
            pattern: Pattern to match in original queries
        """
        keys_to_remove = []

        for key in self.cache:
            # We need to track original queries to match patterns
            # For now, invalidate all - simple but works
            keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]

        if keys_to_remove:
            self.metrics.increment("rag.retrieval.cache.invalidations", len(keys_to_remove))
            logger.info("Invalidated cache entries", count=len(keys_to_remove), pattern=pattern)

    def invalidate_by_file(self, file_path: str):
        """
        Invalidate cache entries containing chunks from specific file.

        Called when a file is modified.

        Args:
            file_path: Path of modified file
        """
        keys_to_remove = []

        for key, (chunks, _) in self.cache.items():
            # Check if any chunk is from this file
            if any(chunk.metadata.file_path == file_path for chunk in chunks):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]

        if keys_to_remove:
            self.metrics.increment("rag.retrieval.cache.invalidations", len(keys_to_remove))
            logger.info(
                "Invalidated cache entries for file", count=len(keys_to_remove), file_path=file_path
            )

    def clear(self):
        """Clear entire cache."""
        size = len(self.cache)
        self.cache.clear()
        self.metrics.increment("rag.retrieval.cache.clears")
        self.metrics.gauge("rag.retrieval.cache.size", 0)
        logger.info("Cleared cache", entries_removed=size)

    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate as percentage (0.0 to 1.0)
        """
        metrics = self.metrics.get_metrics()
        hits = metrics.get("rag.retrieval.cache.hits", 0)
        misses = metrics.get("rag.retrieval.cache.misses", 0)
        total = hits + misses
        if total == 0:
            return 0.0
        return hits / total

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        metrics = self.metrics.get_metrics()
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": metrics.get("rag.retrieval.cache.hits", 0),
            "misses": metrics.get("rag.retrieval.cache.misses", 0),
            "hit_rate": self.get_hit_rate(),
            "ttl": self.ttl,
            "evictions": metrics.get("rag.retrieval.cache.evictions", 0),
            "expirations": metrics.get("rag.retrieval.cache.expirations", 0),
            "invalidations": metrics.get("rag.retrieval.cache.invalidations", 0),
        }

    def warm_cache(self, common_queries: List[Tuple[str, List[Chunk]]]):
        """
        Pre-populate cache with common queries.

        Useful for warming cache on startup with frequent searches.

        Args:
            common_queries: List of (query, results) tuples
        """
        warmed = 0
        for query, results in common_queries:
            if len(self.cache) < self.max_size:
                self.set(query, results)
                warmed += 1

        logger.info("Warmed cache", queries_count=warmed)
