"""
Embeddings module for ACOLYTE.

Generates vector representations of code using UniXcoder
for advanced semantic search.
"""

from typing import Optional

from acolyte.embeddings.types import (
    EmbeddingVector,
    EmbeddingsMetricsSummaryDict,
    RerankerMetricsSummary,
    MetricsProvider,
)
from acolyte.embeddings.unixcoder import UniXcoderEmbeddings
from acolyte.embeddings.context import RichCodeContext, RichCodeContextDict
from acolyte.embeddings.reranker import CrossEncoderReranker
from acolyte.embeddings.cache import ContextAwareCache, CacheEntry, CacheStats
from acolyte.embeddings.persistent_cache import SmartPersistentCache
from acolyte.embeddings.metrics import (
    EmbeddingsMetrics,
    PerformanceMetrics,
    SearchQualityMetrics,
    PerformanceStatsDict,
    SearchQualityReport,
    EmbeddingsMetricsSummary,
)

def get_embeddings() -> UniXcoderEmbeddings: ...
def get_reranker() -> CrossEncoderReranker: ...
def get_embeddings_metrics() -> Optional[MetricsProvider]: ...

__all__ = [
    # Standard types
    "EmbeddingVector",
    # Protocol to prevent circular imports
    "MetricsProvider",
    # Main classes
    "UniXcoderEmbeddings",
    "CrossEncoderReranker",
    "RichCodeContext",
    # Cache
    "ContextAwareCache",
    "CacheEntry",
    "CacheStats",
    "SmartPersistentCache",
    # Metrics
    "EmbeddingsMetrics",
    "PerformanceMetrics",
    "SearchQualityMetrics",
    # TypedDicts for improved type safety
    "RichCodeContextDict",
    "EmbeddingsMetricsSummaryDict",
    "RerankerMetricsSummary",
    "PerformanceStatsDict",
    "SearchQualityReport",
    "EmbeddingsMetricsSummary",
    # Singleton functions
    "get_embeddings",
    "get_reranker",
    "get_embeddings_metrics",
]
