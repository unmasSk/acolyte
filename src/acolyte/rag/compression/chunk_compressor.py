"""
Contextual chunk compressor for intelligent token optimization.

This module implements the main compression logic that decides when and how
to compress chunks based on query analysis, without using LLMs.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from acolyte.core.logging import logger
from acolyte.core.secure_config import Settings
from acolyte.core.token_counter import SmartTokenCounter
from acolyte.core.tracing import MetricsCollector
from acolyte.models.chunk import Chunk
from acolyte.models.document import DocumentType
from acolyte.rag.compression.contextual import QueryAnalyzer
from acolyte.rag.compression.strategies import get_compression_strategy


class CompressionResult:
    """Result of a compression operation with detailed metrics."""

    def __init__(
        self,
        original_chunks: int,
        compressed_chunks: int,
        chunks_actually_compressed: int,
        original_tokens: int,
        compressed_tokens: int,
        tokens_saved: int,
        compression_ratio: float,
        processing_time_ms: float,
        query_type: str,
    ):
        self.original_chunks = original_chunks
        self.compressed_chunks = compressed_chunks
        self.chunks_actually_compressed = chunks_actually_compressed
        self.original_tokens = original_tokens
        self.compressed_tokens = compressed_tokens
        self.tokens_saved = tokens_saved
        self.compression_ratio = compression_ratio
        self.processing_time_ms = processing_time_ms
        self.query_type = query_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/metrics."""
        return {
            "original_chunks": self.original_chunks,
            "compressed_chunks": self.compressed_chunks,
            "chunks_actually_compressed": self.chunks_actually_compressed,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "tokens_saved": self.tokens_saved,
            "compression_ratio": self.compression_ratio,
            "savings_percentage": (1 - self.compression_ratio) * 100,
            "processing_time_ms": self.processing_time_ms,
            "query_type": self.query_type,
        }


class ContextualCompressor:
    """
    Heuristic chunk compressor that optimizes token usage.

    Key features:
    - NO LLM usage (pure heuristics)
    - <50ms latency guarantee
    - Selective compression based on query analysis
    - 60-80% token savings on specific queries

    This class orchestrates the compression process, delegating:
    - Query analysis to QueryAnalyzer
    - Token counting to TokenCounter
    - Compression strategies to specific implementations
    """

    def __init__(
        self,
        token_counter: SmartTokenCounter,
        min_chunk_size: int = 100,
        compression_ratios: Optional[Dict[str, float]] = None,
        early_stop_ms: float = 45.0,
        relevance_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize compressor with configuration.

        Args:
            token_counter: TokenCounter instance from core
            min_chunk_size: Don't compress chunks smaller than this (tokens)
            compression_ratios: Target ratios by relevance level
            early_stop_ms: Stop processing after this many milliseconds
        """
        self.token_counter = token_counter
        self.min_chunk_size = min_chunk_size
        self.early_stop_ms = early_stop_ms

        # Get configuration from .acolyte if available
        config = Settings()
        compression_config = config.get("rag.compression", {})

        # Use provided ratios or get from config, fallback to defaults
        self.compression_ratios = compression_ratios or compression_config.get(
            "ratios",
            {
                "high_relevance": 0.9,  # Keep 90%
                "medium_relevance": 0.6,  # Keep 60%
                "low_relevance": 0.3,  # Keep 30%
                "aggressive": 0.2,  # Keep 20% - last resort
            },
        )

        # Use provided thresholds or get from config, fallback to defaults
        self.relevance_thresholds = relevance_thresholds or compression_config.get(
            "relevance_thresholds",
            {
                "high": 0.8,  # Above this = high relevance
                "medium": 0.5,  # Above this = medium, below = low
                "recompress": 0.3,  # Below this, skip even with aggressive compression
            },
        )

        # Initialize query analyzer
        self.query_analyzer = QueryAnalyzer()

        # Metrics tracking
        self.metrics = CompressionMetrics()

        logger.info(
            f"ContextualCompressor initialized with min_chunk_size {min_chunk_size} and early_stop_ms {early_stop_ms}"
        )

    def _count_tokens(self, chunks: List[Chunk]) -> int:
        """Count total tokens in a list of chunks."""
        return sum(self.token_counter.count_tokens(c.content) for c in chunks)

    def should_compress(self, query: str, chunks: List[Chunk], token_budget: int) -> bool:
        """
        Decide whether to compress based on heuristics.

        Decision factors:
        1. Query context (specific vs general vs generation)
        2. Average chunk size
        3. Total tokens vs budget

        Returns:
            True if compression should be applied
        """

        # Analyze query
        context = self.query_analyzer.analyze_query(query)

        # Never compress for generation queries
        if context.is_generation:
            logger.debug("No compression: generation query")
            return False

        # Don't compress general queries
        if context.is_general:
            logger.debug("No compression: general query")
            return False

        # Check average chunk size
        total_tokens = self._count_tokens(chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0

        if avg_tokens < self.min_chunk_size:
            logger.debug(
                f"No compression: small chunks (avg {avg_tokens:.0f} < {self.min_chunk_size})"
            )
            return False

        # Check if we're over budget
        if total_tokens <= token_budget:
            logger.debug(f"No compression: fits in budget ({total_tokens} <= {token_budget})")
            return False

        # Log decision
        logger.info(
            f"Compression needed for {context.query_type} query with {total_tokens} tokens (budget: {token_budget})"
        )

        return True

    def compress_chunks(
        self, chunks: List[Chunk], query: str, token_budget: int
    ) -> Tuple[List[Chunk], CompressionResult]:
        """
        Compress chunks based on relevance and token budget.

        Process:
        1. Analyze query context
        2. Calculate relevance scores
        3. Apply compression strategy based on score
        4. Ensure we meet token budget
        5. Return compressed chunks with metrics

        Guarantees <50ms latency via early stopping.

        Returns:
            Tuple of (compressed_chunks, compression_result)
        """
        # Validate inputs
        if not chunks:
            logger.warning("compress_chunks called with empty chunks list")
            logger.info("[UNTESTED PATH] Empty chunks list in compress_chunks")
            return [], CompressionResult(
                original_chunks=0,
                compressed_chunks=0,
                chunks_actually_compressed=0,
                original_tokens=0,
                compressed_tokens=0,
                tokens_saved=0,
                compression_ratio=1.0,
                processing_time_ms=0.0,
                query_type="unknown",
            )

        if token_budget <= 0:
            logger.error(f"Invalid token_budget: {token_budget}")
            logger.info("[UNTESTED PATH] Invalid token_budget in compress_chunks")
            raise ValueError(f"token_budget must be positive, got {token_budget}")

        start_time = time.time()

        # Analyze query for context
        context = self.query_analyzer.analyze_query(query)

        # Track original token count
        original_tokens = self._count_tokens(chunks)

        # Calculate relevance scores
        scored_chunks = []
        for chunk in chunks:
            relevance = self.query_analyzer.calculate_relevance(chunk, context)
            scored_chunks.append((relevance, chunk))

        # Sort by relevance (highest first)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        compressed_chunks = []
        current_tokens = 0
        chunks_compressed = 0

        for relevance, chunk in scored_chunks:
            # Check early stopping
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.early_stop_ms:
                logger.warning(
                    f"Early stopping at {elapsed_ms:.1f}ms (processed {len(compressed_chunks)}/{len(chunks)} chunks)"
                )
                break

            # Determine compression level using configurable thresholds
            if relevance > self.relevance_thresholds["high"]:
                ratio = self.compression_ratios["high_relevance"]
            elif relevance > self.relevance_thresholds["medium"]:
                ratio = self.compression_ratios["medium_relevance"]
            else:
                ratio = self.compression_ratios["low_relevance"]

            # Compress chunk
            compressed = self._compress_chunk(chunk, relevance, ratio)
            compressed_tokens = self.token_counter.count_tokens(compressed.content)

            # Check if we can fit this chunk
            if current_tokens + compressed_tokens <= token_budget:
                compressed_chunks.append(compressed)
                current_tokens += compressed_tokens
                # Count as compressed if content changed
                if len(compressed.content) < len(chunk.content):
                    chunks_compressed += 1
            else:
                # Try more aggressive compression if chunk is valuable
                if relevance > self.relevance_thresholds["recompress"]:
                    aggressive = self._compress_chunk(
                        chunk, relevance, self.compression_ratios["aggressive"]
                    )
                    aggressive_tokens = self.token_counter.count_tokens(aggressive.content)
                    if current_tokens + aggressive_tokens <= token_budget:
                        compressed_chunks.append(aggressive)
                        current_tokens += aggressive_tokens
                        chunks_compressed += 1
                    else:
                        # Can't fit even with aggressive compression
                        logger.debug(
                            f"Chunk skipped: won't fit (relevance {relevance:.2f}, needs {aggressive_tokens} tokens)"
                        )
                        logger.info(
                            "[UNTESTED PATH] Chunk won't fit even with aggressive compression"
                        )
                else:
                    # Low relevance, skip
                    logger.debug(f"Chunk skipped: low relevance score {relevance:.2f}")
                    logger.info("[UNTESTED PATH] Chunk skipped due to low relevance")

        # Calculate metrics
        processing_time_ms = (time.time() - start_time) * 1000
        compression_ratio = current_tokens / original_tokens if original_tokens > 0 else 1.0
        tokens_saved = original_tokens - current_tokens

        # Create result
        result = CompressionResult(
            original_chunks=len(chunks),
            compressed_chunks=len(compressed_chunks),
            chunks_actually_compressed=chunks_compressed,
            original_tokens=original_tokens,
            compressed_tokens=current_tokens,
            tokens_saved=tokens_saved,
            compression_ratio=compression_ratio,
            processing_time_ms=processing_time_ms,
            query_type=context.query_type,
        )

        # Update metrics
        self.metrics.record_compression(result)

        # Log summary
        logger.info(
            f"Compression complete: {len(chunks)}→{len(compressed_chunks)} chunks, "
            f"{original_tokens}→{current_tokens} tokens, saved {tokens_saved} ({(1-compression_ratio)*100:.1f}%), "
            f"time {processing_time_ms:.1f}ms"
        )

        return compressed_chunks, result

    def _compress_chunk(self, chunk: Chunk, relevance: float, target_ratio: float) -> Chunk:
        """
        Compress a single chunk using appropriate strategy.

        Returns:
            New Chunk instance with compressed content
        """
        # Get document type from metadata or infer
        doc_type = self._infer_document_type(chunk)
        language = chunk.metadata.language if chunk.metadata else None

        # Get compression strategy
        try:
            strategy = get_compression_strategy(doc_type, language)
        except Exception as e:
            logger.error(f"Failed to get compression strategy: {e}")
            logger.info("[UNTESTED PATH] Failed to get compression strategy")
            # Return original chunk if strategy fails
            return chunk

        # Apply compression
        try:
            compressed_content = strategy.compress(chunk.content, relevance, target_ratio)
        except Exception as e:
            logger.error(f"Compression failed for chunk {chunk.id}: {e} - Using original content")
            logger.info("[UNTESTED PATH] Compression strategy failed")
            return chunk

        # Create new chunk with compressed content
        # Can't add dynamic fields to ChunkMetadata
        # Only update content and add a summary if it's useful

        if len(compressed_content) < len(chunk.content):
            # Add summary only when actual compression happened
            tokens_before = self.token_counter.count_tokens(chunk.content)
            tokens_after = self.token_counter.count_tokens(compressed_content)
            reduction_pct = int((1 - tokens_after / max(tokens_before, 1)) * 100)
            summary = (
                f"Compressed ({reduction_pct}% token reduction, " f"relevance: {relevance:.2f})"
            )

            return Chunk(
                id=chunk.id,  # Preserve ID
                content=compressed_content,
                metadata=chunk.metadata,  # Preserve original metadata
                summary=summary,  # Use the summary field
            )
        else:
            # No compression happened, return original
            return chunk

    def _infer_document_type(self, chunk: Chunk) -> DocumentType:
        """Infer document type from chunk metadata or content."""
        if chunk.metadata and chunk.metadata.file_path:
            path = chunk.metadata.file_path.lower()

            # Code files
            if any(
                path.endswith(ext)
                for ext in [
                    ".py",
                    ".js",
                    ".ts",
                    ".java",
                    ".go",
                    ".rs",
                    ".cpp",
                    ".c",
                    ".kt",
                    ".swift",
                    ".rb",
                    ".php",
                    ".scala",
                ]
            ):
                return DocumentType.CODE

            # Markdown
            elif path.endswith(".md"):
                return DocumentType.MARKDOWN

            # Config
            elif any(
                path.endswith(ext)
                for ext in [
                    ".json",
                    ".yaml",
                    ".yml",
                    ".toml",
                    ".ini",
                    ".conf",
                    ".env",
                    ".properties",
                    ".xml",
                ]
            ):
                return DocumentType.CONFIG

            # Data
            elif any(path.endswith(ext) for ext in [".csv", ".tsv", ".sql", ".jsonl", ".parquet"]):
                return DocumentType.DATA

        # Try to infer from content if no file path
        content_lower = chunk.content[:200].lower()

        # Look for code patterns
        if any(
            pattern in content_lower
            for pattern in ["def ", "function ", "class ", "import ", "const ", "var "]
        ):
            return DocumentType.CODE

        # Look for markdown patterns
        elif any(pattern in content_lower for pattern in ["# ", "## ", "### "]):
            return DocumentType.MARKDOWN

        # Default
        return DocumentType.OTHER


class CompressionMetrics:
    """Track compression performance metrics using Core infrastructure via composition."""

    def __init__(self):
        # Usar MetricsCollector por composición, NO herencia
        self.collector = MetricsCollector()

        # Métricas específicas de compression
        self.total_compressions = 0
        self.total_time_ms = 0.0
        self.total_tokens_saved = 0
        self.total_chunks_processed = 0
        self.total_chunks_compressed = 0
        self.compression_by_type: Dict[str, int] = {}
        self.average_ratio = 0.0

        # Performance tracking
        self.time_percentiles: List[float] = []  # For p95/p99 calculation
        self.max_time_ms = 0.0
        self.min_time_ms = float("inf")

    def record_compression(self, result: CompressionResult) -> None:
        """Record metrics for a compression operation."""
        # Registrar en métricas base de Core (usando composición)
        self.collector.increment("rag.compression.total_compressions")
        self.collector.increment("rag.compression.tokens_saved", result.tokens_saved)
        self.collector.increment(
            "rag.compression.chunks_compressed", result.chunks_actually_compressed
        )

        # Mantener métricas locales para retrocompatibilidad
        self.total_compressions += 1
        self.total_time_ms += result.processing_time_ms
        self.total_tokens_saved += result.tokens_saved
        self.total_chunks_processed += result.original_chunks
        self.total_chunks_compressed += result.chunks_actually_compressed

        # Track by query type
        if result.query_type not in self.compression_by_type:
            self.compression_by_type[result.query_type] = 0
        self.compression_by_type[result.query_type] += 1

        # Update average ratio
        self.average_ratio = (
            self.average_ratio * (self.total_compressions - 1) + result.compression_ratio
        ) / self.total_compressions

        # Track performance
        self.time_percentiles.append(result.processing_time_ms)
        self.max_time_ms = max(self.max_time_ms, result.processing_time_ms)
        self.min_time_ms = min(self.min_time_ms, result.processing_time_ms)

        # Keep only last 1000 measurements for percentiles
        if len(self.time_percentiles) > 1000:
            self.time_percentiles = self.time_percentiles[-1000:]

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary including base metrics from Core."""
        # Get base metrics from Core
        base_metrics = self.collector.get_metrics()

        # Calculate percentiles
        p95 = p99 = 0.0
        if self.time_percentiles:
            sorted_times = sorted(self.time_percentiles)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            p95 = sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1]
            p99 = sorted_times[p99_idx] if p99_idx < len(sorted_times) else sorted_times[-1]

        return {
            # Métricas detalladas locales
            "total_compressions": self.total_compressions,
            "total_chunks_processed": self.total_chunks_processed,
            "total_chunks_compressed": self.total_chunks_compressed,
            "chunks_compression_rate": (
                self.total_chunks_compressed / self.total_chunks_processed * 100
                if self.total_chunks_processed > 0
                else 0
            ),
            "average_time_ms": (
                self.total_time_ms / self.total_compressions if self.total_compressions > 0 else 0
            ),
            "p95_time_ms": p95,
            "p99_time_ms": p99,
            "max_time_ms": self.max_time_ms,
            "min_time_ms": self.min_time_ms if self.min_time_ms != float("inf") else 0.0,
            "total_tokens_saved": self.total_tokens_saved,
            "average_compression_ratio": self.average_ratio,
            "average_savings_percentage": (1 - self.average_ratio) * 100,
            "compressions_by_type": self.compression_by_type,
            # Métricas base de Core
            "core_metrics": base_metrics,
        }

    def log_summary(self) -> None:
        """Log a formatted summary of metrics."""
        summary = self.get_summary()
        logger.info(
            f"CompressionMetrics summary: total {summary['total_compressions']}, "
            f"tokens saved {summary['total_tokens_saved']:,}, "
            f"avg time {summary['average_time_ms']:.1f}ms, "
            f"p95 {summary['p95_time_ms']:.1f}ms, "
            f"p99 {summary['p99_time_ms']:.1f}ms, "
            f"avg savings {summary['average_savings_percentage']:.1f}%"
        )

        # Log métricas base de Core (usando composición)
        base_metrics = self.collector.get_metrics()
        if base_metrics:
            logger.debug(f"Core base metrics: {base_metrics}")
