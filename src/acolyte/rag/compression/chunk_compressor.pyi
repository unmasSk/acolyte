from typing import Any, Dict, List, Optional, Tuple

from acolyte.core.token_counter import SmartTokenCounter
from acolyte.models.chunk import Chunk

class CompressionResult:
    original_chunks: int
    compressed_chunks: int
    chunks_actually_compressed: int
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    compression_ratio: float
    processing_time_ms: float
    query_type: str

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
    ) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...

class ContextualCompressor:
    token_counter: SmartTokenCounter
    min_chunk_size: int
    early_stop_ms: float
    compression_ratios: Dict[str, float]
    relevance_thresholds: Dict[str, float]
    query_analyzer: Any
    metrics: CompressionMetrics

    def __init__(
        self,
        token_counter: SmartTokenCounter,
        min_chunk_size: int = ...,
        compression_ratios: Optional[Dict[str, float]] = ...,
        early_stop_ms: float = ...,
        relevance_thresholds: Optional[Dict[str, float]] = ...,
    ) -> None: ...
    def should_compress(self, query: str, chunks: List[Chunk], token_budget: int) -> bool: ...
    def compress_chunks(
        self, chunks: List[Chunk], query: str, token_budget: int
    ) -> Tuple[List[Chunk], CompressionResult]: ...

class CompressionMetrics:
    collector: Any
    total_compressions: int
    total_time_ms: float
    total_tokens_saved: int
    total_chunks_processed: int
    total_chunks_compressed: int
    compression_by_type: Dict[str, int]
    average_ratio: float
    time_percentiles: List[float]
    max_time_ms: float
    min_time_ms: float

    def __init__(self) -> None: ...
    def record_compression(self, result: CompressionResult) -> None: ...
    def get_summary(self) -> Dict[str, Any]: ...
    def log_summary(self) -> None: ...
