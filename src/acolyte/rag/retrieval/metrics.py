"""
RAG-specific metrics collection.

Uses the core MetricsCollector to track retrieval performance,
cache efficiency, and search quality metrics.
"""

from typing import Dict, Any, List

from acolyte.core.tracing import MetricsCollector
from acolyte.core.logging import logger


class RAGMetrics:
    """
    Metrics collector for RAG operations.

    Tracks search performance, cache efficiency, and result quality.
    Uses core MetricsCollector via composition (NOT inheritance).
    """

    def __init__(self):
        """Initialize RAG metrics collector."""
        # Use MetricsCollector as composition, not inheritance
        self.collector = MetricsCollector()

        # Additional RAG-specific counters
        self.search_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.compression_count = 0
        self.total_chunks_returned = 0
        self.total_tokens_saved = 0

        # Track latencies manually since collector only has increment/gauge
        self.latencies = {
            "search_hybrid": [],
            "search_semantic": [],
            "search_lexical": [],
            "compression": [],
        }

        logger.info("RAGMetrics initialized")

    def record_search(
        self,
        query: str,
        latency_ms: float,
        cache_hit: bool,
        result_count: int,
        search_type: str = "hybrid",
    ):
        """
        Record a search operation.

        Args:
            query: Search query (for logging)
            latency_ms: Search latency in milliseconds
            cache_hit: Whether results came from cache
            result_count: Number of results returned
            search_type: Type of search (hybrid, semantic, lexical)
        """
        # Track latency manually
        key = f"search_{search_type}"
        if key not in self.latencies:
            self.latencies[key] = []
        self.latencies[key].append(latency_ms)

        # Update counters
        self.search_count += 1
        self.total_chunks_returned += result_count

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        # Use MetricsCollector's increment method
        self.collector.increment(f"{search_type}_searches", 1)
        self.collector.increment("total_chunks_returned", result_count)

        logger.debug(
            "Search recorded",
            search_type=search_type,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            result_count=result_count,
        )

    def record_compression_savings(
        self, original_tokens: int, compressed_tokens: int, compression_time_ms: float
    ):
        """
        Record compression metrics.

        Args:
            original_tokens: Token count before compression
            compressed_tokens: Token count after compression
            compression_time_ms: Time taken to compress
        """
        tokens_saved = original_tokens - compressed_tokens
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        # Track compression time
        if "compression" not in self.latencies:
            self.latencies["compression"] = []
        self.latencies["compression"].append(compression_time_ms)

        # Update totals
        self.compression_count += 1
        self.total_tokens_saved += tokens_saved

        # Use MetricsCollector's increment
        self.collector.increment("compression_count", 1)
        self.collector.increment("tokens_saved", tokens_saved)

        logger.debug(
            "Compression recorded",
            tokens_saved=tokens_saved,
            compression_ratio=compression_ratio,
            compression_time_ms=compression_time_ms,
        )

    def record_filter_impact(self, filter_type: str, before_count: int, after_count: int):
        """
        Record impact of filters on result count.

        Args:
            filter_type: Type of filter applied
            before_count: Results before filtering
            after_count: Results after filtering
        """
        reduction = before_count - after_count
        reduction_ratio = reduction / before_count if before_count > 0 else 0

        # Use gauge for current values
        self.collector.gauge(f"filter_{filter_type}_last_reduction", reduction)
        self.collector.gauge(f"filter_{filter_type}_last_ratio", reduction_ratio)

        logger.debug(
            "Filter impact",
            filter_type=filter_type,
            reduction=reduction,
            reduction_ratio=reduction_ratio,
        )

    def record_rerank_impact(self, strategy: str, score_changes: List[float]):
        """
        Record impact of re-ranking on scores.

        Args:
            strategy: Re-ranking strategy used
            score_changes: List of score deltas
        """
        if not score_changes:
            return

        avg_change = sum(score_changes) / len(score_changes)
        max_change = max(abs(c) for c in score_changes)

        # Use gauge for current values
        self.collector.gauge(f"rerank_{strategy}_avg_change", avg_change)
        self.collector.gauge(f"rerank_{strategy}_max_change", max_change)

    def get_cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate as percentage (0.0 to 1.0)
        """
        total = self.cache_hits + self.cache_misses
        if total == 0:
            logger.info("[UNTESTED PATH] Cache hit rate calculation with zero total")
            return 0.0
        return self.cache_hits / total

    def get_average_latency(self, operation: str = "search_hybrid") -> float:
        """
        Get average latency for an operation.

        Returns:
            Average latency in milliseconds
        """
        if operation in self.latencies and self.latencies[operation]:
            return sum(self.latencies[operation]) / len(self.latencies[operation])
        return 0.0

    def get_p95_latency(self, operation: str = "search_hybrid") -> float:
        """
        Get p95 latency for an operation.

        Returns:
            p95 latency in milliseconds
        """
        if operation in self.latencies and self.latencies[operation]:
            sorted_latencies = sorted(self.latencies[operation])
            index = int(len(sorted_latencies) * 0.95)
            return sorted_latencies[min(index, len(sorted_latencies) - 1)]
        return 0.0

    def get_compression_efficiency(self) -> Dict[str, float]:
        """
        Get compression efficiency metrics.

        Returns:
            Dict with compression statistics
        """
        avg_compression_time = self.get_average_latency("compression")

        return {
            "total_compressions": self.compression_count,
            "total_tokens_saved": self.total_tokens_saved,
            "avg_compression_time_ms": avg_compression_time,
        }

    def get_search_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive search metrics summary.

        Returns:
            Dict with all search-related metrics
        """
        # Get counts from MetricsCollector
        collector_metrics = self.collector.get_metrics()

        return {
            # Search performance
            "total_searches": self.search_count,
            "avg_latency_ms": self.get_average_latency("search_hybrid"),
            "p95_latency_ms": self.get_p95_latency("search_hybrid"),
            # Cache efficiency
            "cache_hit_rate": self.get_cache_hit_rate(),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            # Result metrics
            "total_chunks_returned": self.total_chunks_returned,
            "avg_results_per_search": (
                self.total_chunks_returned / self.search_count if self.search_count > 0 else 0
            ),
            # Compression metrics
            "compression_efficiency": self.get_compression_efficiency(),
            # Component breakdown from collector
            "semantic_searches": collector_metrics.get("semantic_searches", 0),
            "lexical_searches": collector_metrics.get("lexical_searches", 0),
            "hybrid_searches": collector_metrics.get("hybrid_searches", 0),
        }

    def log_summary(self):
        """Log a summary of RAG metrics."""
        summary = self.get_search_summary()

        logger.info(
            "RAG Metrics Summary",
            searches=summary['total_searches'],
            avg_latency_ms=summary['avg_latency_ms'],
            cache_hit_rate=summary['cache_hit_rate'],
            tokens_saved=summary['compression_efficiency']['total_tokens_saved'],
        )

    def reset_counters(self):
        """Reset all counters (useful for testing)."""
        self.search_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.compression_count = 0
        self.total_chunks_returned = 0
        self.total_tokens_saved = 0

        # Clear latency tracking
        for key in self.latencies:
            self.latencies[key] = []

        # MetricsCollector doesn't have reset, so create new instance
        self.collector = MetricsCollector()

        logger.info("RAG metrics reset")


# Singleton instance for easy access
_rag_metrics = None


def get_rag_metrics() -> RAGMetrics:
    """Get the global RAG metrics instance."""
    global _rag_metrics
    if _rag_metrics is None:
        _rag_metrics = RAGMetrics()
    return _rag_metrics
