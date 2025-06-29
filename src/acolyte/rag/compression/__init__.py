"""
Compression module for intelligent chunk optimization.

This module provides heuristic compression strategies that optimize token usage
without using LLMs, ensuring <50ms latency.
"""

from acolyte.rag.compression.chunk_compressor import (
    CompressionMetrics,
    CompressionResult,
    ContextualCompressor,
)
from acolyte.rag.compression.contextual import (
    QueryAnalyzer,
    QueryContext,
)
from acolyte.rag.compression.strategies import (
    CodeCompressionStrategy,
    CompressionStrategy,
    ConfigCompressionStrategy,
    DataCompressionStrategy,
    MarkdownCompressionStrategy,
    OtherCompressionStrategy,
    get_compression_strategy,
)

__all__ = [
    # Main compressor
    "ContextualCompressor",
    "CompressionMetrics",
    "CompressionResult",
    # Query analysis
    "QueryAnalyzer",
    "QueryContext",
    # Strategies
    "CompressionStrategy",
    "CodeCompressionStrategy",
    "MarkdownCompressionStrategy",
    "ConfigCompressionStrategy",
    "DataCompressionStrategy",
    "OtherCompressionStrategy",
    "get_compression_strategy",
]
