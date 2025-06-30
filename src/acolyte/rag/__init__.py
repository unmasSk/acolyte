"""
RAG (Retrieval-Augmented Generation) module for ACOLYTE.

This module provides intelligent code retrieval and context building
for the AI assistant, including hybrid search and contextual compression.
"""

# Compression exports
from acolyte.rag.compression import (
    ContextualCompressor,
    CompressionMetrics,
    CompressionResult,
    QueryAnalyzer,
    QueryContext,
)

# Retrieval exports
from acolyte.rag.retrieval import (
    HybridSearch,
    ScoredChunk,
    SearchFilters,
    FuzzyMatcher,
    get_fuzzy_matcher,
)

__all__ = [
    # Compression
    "ContextualCompressor",
    "CompressionMetrics",
    "CompressionResult",
    "QueryAnalyzer",
    "QueryContext",
    # Retrieval
    "HybridSearch",
    "ScoredChunk",
    "SearchFilters",
    "FuzzyMatcher",
    "get_fuzzy_matcher",
]
