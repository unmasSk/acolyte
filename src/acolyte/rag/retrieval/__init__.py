"""
Retrieval module for ACOLYTE's RAG system.

Implements hybrid search (70% semantic + 30% lexical) with supporting
components for filtering, caching, re-ranking, and metrics.
"""

from acolyte.rag.retrieval.hybrid_search import (
    HybridSearch,
    ScoredChunk,
    SearchFilters as HybridSearchFilters,  # Alias to avoid confusion
)
from acolyte.rag.retrieval.filters import SearchFilters
from acolyte.rag.retrieval.cache import SearchCache
from acolyte.rag.retrieval.rerank import SimpleReranker
from acolyte.rag.retrieval.metrics import RAGMetrics, get_rag_metrics
from acolyte.rag.retrieval.fuzzy_matcher import FuzzyMatcher, get_fuzzy_matcher

__all__ = [
    # Main search
    "HybridSearch",
    "ScoredChunk",
    # Filters
    "SearchFilters",
    "HybridSearchFilters",
    # Cache
    "SearchCache",
    # Re-ranking
    "SimpleReranker",
    # Metrics
    "RAGMetrics",
    "get_rag_metrics",
    # Fuzzy matching
    "FuzzyMatcher",
    "get_fuzzy_matcher",
]
