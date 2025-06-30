"""
Specific metrics for the embeddings module.

Extends the base metrics system of Core to provide
specialized metrics for performance and search quality.

NOTE: This module MUST NOT import from reranker.py or unixcoder.py
to avoid circular dependencies.
"""

from typing import Dict, List, Tuple, TypedDict, Union, Optional
import time
import collections
import math

from acolyte.core.tracing import MetricsCollector
from acolyte.core.logging import logger


class OperationStats(TypedDict):
    """Detailed statistics by operation."""

    count: int
    avg: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float


class GlobalStats(TypedDict):
    """Global performance statistics."""

    total_operations: int
    avg: float
    p95: float
    meets_sla: bool


class PerformanceStatsDict(TypedDict, total=False):
    """Complete performance statistics dictionary.

    total=False allows fields to be optional since
    operations are added dynamically.
    """

    _global: GlobalStats
    encode: OperationStats
    encode_batch: OperationStats
    model_load: OperationStats
    cache_lookup: OperationStats
    tokenization: OperationStats


class SearchQualityReport(TypedDict):
    """Search quality report."""

    mrr: float
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    total_clicks: int
    unique_queries: int
    queries_with_feedback: int
    avg_search_time_ms: float  # Optional, only if data is present
    p95_search_time_ms: float  # Optional, only if data is present


class CacheMetrics(TypedDict):
    """Cache metrics."""

    hit_rate: float
    hits: int
    misses: int


class OperationsMetrics(TypedDict):
    """Operations metrics."""

    total_embeddings: int
    total_errors: int
    error_rate: float


class HealthMetrics(TypedDict):
    """System health metrics."""

    meets_sla: bool
    quality_threshold: bool


class EmbeddingsMetricsSummary(TypedDict):
    """Complete metrics summary."""

    performance: PerformanceStatsDict
    quality: SearchQualityReport
    cache: CacheMetrics
    operations: OperationsMetrics
    health: HealthMetrics


class PerformanceMetrics:
    """Performance metrics to meet p95 < 5s requirement.

    Tracks latencies of different operations to ensure
    the module meets performance requirements.
    """

    def __init__(self):
        """Initializes structures to track latencies."""
        self._latencies: Dict[str, List[float]] = {
            "encode": [],
            "encode_batch": [],
            "model_load": [],
            "cache_lookup": [],
            "tokenization": [],
        }
        self._start_times: Dict[str, float] = {}

    def start_operation(self, operation: str) -> str:
        """Marks the start of an operation.

        Args:
            operation: Name of the operation

        Returns:
            Unique operation ID for tracking
        """
        op_id = f"{operation}_{time.time()}"
        self._start_times[op_id] = time.time()
        return op_id

    def end_operation(self, op_id: str) -> float:
        """Marks the end of an operation and records latency.

        Args:
            op_id: ID returned by start_operation

        Returns:
            Latency in milliseconds
        """
        if op_id not in self._start_times:
            logger.warning(f"Operation ID {op_id} not found")
            return 0.0

        start_time = self._start_times.pop(op_id)
        latency_ms = (time.time() - start_time) * 1000

        # Extract operation name from ID
        operation = op_id.rsplit('_', 1)[0]
        self.record_latency(operation, latency_ms)

        return latency_ms

    def record_latency(self, operation: str, latency_ms: float):
        """Records latency for an operation.

        Args:
            operation: Operation type
            latency_ms: Latency in milliseconds
        """
        if operation not in self._latencies:
            self._latencies[operation] = []

        self._latencies[operation].append(latency_ms)

        # Keep only the last 1000 measurements per operation
        if len(self._latencies[operation]) > 1000:
            self._latencies[operation] = self._latencies[operation][-1000:]

    def get_p95(self, operation: Optional[str] = None) -> float:
        """Calculates p95 latency for an operation or all.

        Args:
            operation: Specific operation or None for all

        Returns:
            p95 latency in milliseconds
        """
        if operation and operation in self._latencies:
            times = self._latencies[operation]
        else:
            # Combine all operations
            times = [t for times in self._latencies.values() for t in times]

        if not times:
            return 0.0

        sorted_times = sorted(times)
        idx = math.ceil((len(sorted_times) * 0.95) - 1)
        idx = min(max(idx, 0), len(sorted_times) - 1)
        return sorted_times[idx]

    def get_stats(self) -> PerformanceStatsDict:
        """Returns complete performance statistics.

        Returns:
            PerformanceStatsDict with metrics per operation
        """
        stats: PerformanceStatsDict = {}

        for op, times in self._latencies.items():
            if times:
                sorted_times = sorted(times)
                stats[op] = {
                    "count": len(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "p50": sorted_times[len(sorted_times) // 2],
                    "p95": self.get_p95(op),
                    "p99": sorted_times[int(len(sorted_times) * 0.99)],
                }

        # Global metrics
        all_times = [t for times in self._latencies.values() for t in times]
        if all_times:
            stats["_global"] = {
                "total_operations": len(all_times),
                "avg": sum(all_times) / len(all_times),
                "p95": self.get_p95(),
                "meets_sla": self.get_p95() < 5000,  # < 5 seconds
            }

        return stats

    def check_sla_compliance(self) -> Tuple[bool, Dict[str, bool]]:
        """Checks if performance SLAs are met.

        Returns:
            Tuple of (global_compliance, per_operation_dict)
        """
        global_compliant = self.get_p95() < 5000  # 5 seconds

        operation_compliance = {}
        for op in self._latencies:
            p95 = self.get_p95(op)
            # Different SLAs per operation
            if op == "model_load":
                operation_compliance[op] = p95 < 30000  # 30s for initial load
            elif op == "encode_batch":
                operation_compliance[op] = p95 < 10000  # 10s for batch
            else:
                operation_compliance[op] = p95 < 5000  # 5s for the rest

        return global_compliant, operation_compliance


class SearchQualityMetrics:
    """Search quality metrics (MRR, Precision@K).

    Tracks the effectiveness of embeddings in search tasks
    using standard information retrieval metrics.
    """

    def __init__(self):
        """Initializes structures to track quality."""
        self._clicks: collections.deque[Tuple[str, int]] = collections.deque(
            maxlen=10_000
        )  # (query, position)
        self._relevance_scores: collections.deque[Tuple[str, List[bool]]] = collections.deque(
            maxlen=10_000
        )  # (query, [relevant?])
        self._query_results: Dict[str, List[str]] = {}  # For tracking
        self._search_times: collections.deque[float] = collections.deque(maxlen=10_000)

    def record_search_results(
        self, query: str, results: List[str], search_time_ms: Optional[float] = None
    ):
        """Records search results.

        Args:
            query: Search query
            results: List of result IDs/contents in order
            search_time_ms: Search time in ms (optional)
        """
        self._query_results[query] = results

        if search_time_ms is not None:
            self._search_times.append(search_time_ms)

    def record_click(self, query: str, clicked_result: str):
        """Records a click on a search result.

        Args:
            query: Original query
            clicked_result: ID/content of the clicked result
        """
        if query not in self._query_results:
            logger.warning(f"Query not found in results: {query}")
            return

        try:
            position = self._query_results[query].index(clicked_result) + 1
            self._clicks.append((query, position))
            logger.debug(f"Click recorded - query: {query[:50]}..., position: {position}")
        except ValueError:
            logger.warning(f"Click on result not found: {clicked_result[:50]}...")

    def record_relevance_feedback(self, query: str, relevance_list: List[bool]):
        """Records explicit relevance feedback.

        Args:
            query: Original query
            relevance_list: List of bool indicating if each result was relevant
        """
        self._relevance_scores.append((query, relevance_list))

    def calculate_mrr(self) -> float:
        """Calculates Mean Reciprocal Rank.

        MRR = average of (1 / position of first relevant result)

        Returns:
            MRR between 0.0 and 1.0
        """
        if not self._clicks:
            return 0.0

        reciprocal_ranks = []
        queries = {}

        # Group by query and take the highest position (first click)
        for query, position in self._clicks:
            if query not in queries or position < queries[query]:
                queries[query] = position

        # Calculate reciprocal ranks
        for query, position in queries.items():
            reciprocal_ranks.append(1.0 / position)

        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    def calculate_precision_at_k(self, k: int) -> float:
        """Calculates Precision@K.

        Precision@K = proportion of relevant results in the top K

        Args:
            k: Number of top results to consider

        Returns:
            Precision between 0.0 and 1.0
        """
        if not self._relevance_scores:
            return 0.0

        precisions = []

        for query, relevance_list in self._relevance_scores:
            # Take only the first k results
            top_k = relevance_list[:k]
            if top_k:
                precision = sum(top_k) / len(top_k)
                precisions.append(precision)

        return sum(precisions) / len(precisions) if precisions else 0.0

    def calculate_recall_at_k(self, k: int) -> float:
        """Calculates Recall@K.

        Recall@K = proportion of relevant items retrieved in top K

        Args:
            k: Number of top results to consider

        Returns:
            Recall between 0.0 and 1.0
        """
        if not self._relevance_scores:
            return 0.0

        recalls = []

        for query, relevance_list in self._relevance_scores:
            total_relevant = sum(relevance_list)
            if total_relevant == 0:
                continue

            relevant_in_top_k = sum(relevance_list[:k])
            recall = relevant_in_top_k / total_relevant
            recalls.append(recall)

        return sum(recalls) / len(recalls) if recalls else 0.0

    def get_quality_report(self) -> SearchQualityReport:
        """Returns complete quality report.

        Returns:
            SearchQualityReport with all quality metrics
        """
        report: Dict[str, Union[float, int]] = {
            "mrr": self.calculate_mrr(),
            "precision_at_1": self.calculate_precision_at_k(1),
            "precision_at_5": self.calculate_precision_at_k(5),
            "precision_at_10": self.calculate_precision_at_k(10),
            "recall_at_5": self.calculate_recall_at_k(5),
            "recall_at_10": self.calculate_recall_at_k(10),
            "total_clicks": len(self._clicks),
            "unique_queries": len(set(q for q, _ in self._clicks)),
            "queries_with_feedback": len(self._relevance_scores),
        }

        if self._search_times:
            report.update(
                {
                    "avg_search_time_ms": sum(self._search_times) / len(self._search_times),
                    "p95_search_time_ms": sorted(self._search_times)[
                        int(len(self._search_times) * 0.95)
                    ],
                }
            )

        # Cast to SearchQualityReport since we know the structure is correct
        return report  # type: ignore[return-value]


class EmbeddingsMetrics:
    """Combined metrics for the embeddings module.

    Combines the Core base system with specific metrics
    for performance and search quality.
    """

    def __init__(self):
        """Initializes all metrics components."""
        # Core base system (composition, not inheritance)
        self.collector = MetricsCollector()  # No namespace

        # Embeddings-specific metrics
        self.performance = PerformanceMetrics()
        self.search_quality = SearchQualityMetrics()

        # Internal counters
        self._cache_hits = 0
        self._cache_misses = 0
        self._embeddings_generated = 0
        self._errors = 0
        self._total_encoding_operations = 0

        logger.info("EmbeddingsMetrics initialized")

    def record_operation(self, operation: str, latency_ms: float, success: bool = True):
        """Records an embeddings operation.

        Args:
            operation: Operation type
            latency_ms: Latency in milliseconds
            success: Whether the operation was successful
        """
        # Record in Core
        self.collector.increment(f"embeddings.{operation}.count", 1)
        self.collector.gauge(f"embeddings.{operation}.latency_ms", latency_ms)

        if success:
            self.collector.increment(f"embeddings.{operation}.success", 1)
        else:
            self.collector.increment(f"embeddings.{operation}.errors", 1)
            # Only count errors for encoding operations
            if operation in ["encode", "encode_batch"]:
                self._errors += 1

        # Record in performance metrics
        self.performance.record_latency(operation, latency_ms)

        # Update specific counters
        if operation in ["encode", "encode_batch"]:
            self._total_encoding_operations += 1
            if success:
                self._embeddings_generated += 1

    def record_cache_hit(self):
        """Records a cache hit."""
        self._cache_hits += 1
        self.collector.increment("embeddings.cache.hits", 1)

    def record_cache_miss(self):
        """Records a cache miss."""
        self._cache_misses += 1
        self.collector.increment("embeddings.cache.misses", 1)

    def get_cache_hit_rate(self) -> float:
        """Calculates cache hit rate.

        Returns:
            Hit rate between 0.0 and 1.0
        """
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    def get_p95_latency(self) -> float:
        """Calculates global p95 latency.

        Returns:
            p95 latency in milliseconds
        """
        return self.performance.get_p95()

    def _calculate_error_rate(self) -> float:
        """Calculates error rate, handling edge cases correctly.

        Returns:
            Error rate between 0.0 and 1.0
        """
        if self._total_encoding_operations == 0:
            return 0.0
        return self._errors / self._total_encoding_operations

    def get_summary(self) -> EmbeddingsMetricsSummary:
        """Returns complete metrics summary.

        Returns:
            EmbeddingsMetricsSummary with all module metrics
        """
        performance_stats = self.performance.get_stats()
        quality_report = self.search_quality.get_quality_report()

        return {
            "performance": performance_stats,
            "quality": quality_report,
            "cache": {
                "hit_rate": self.get_cache_hit_rate(),
                "hits": self._cache_hits,
                "misses": self._cache_misses,
            },
            "operations": {
                "total_embeddings": self._embeddings_generated,
                "total_errors": self._errors,
                "error_rate": self._calculate_error_rate(),
            },
            "health": {
                "meets_sla": performance_stats.get("_global", {}).get("meets_sla", False),
                "quality_threshold": quality_report.get("mrr", 0) > 0.7,  # MRR > 0.7
            },
        }

    def log_summary(self):
        """Logs metrics summary for monitoring."""
        summary = self.get_summary()

        logger.info(
            f"Embeddings metrics - "
            f"Cache hit rate: {summary['cache']['hit_rate']:.2%}, "
            f"P95 latency: {summary['performance'].get('_global', {}).get('p95', 0):.1f}ms, "
            f"MRR: {summary['quality']['mrr']:.3f}"
        )
