from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from acolyte.models.chunk import Chunk
from acolyte.rag.retrieval.cache import SearchCache
from acolyte.core.token_counter import SmartTokenCounter
from acolyte.rag.compression import ContextualCompressor

@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    source: str = ""

@dataclass
class SearchFilters:
    file_path: Optional[str] = None
    file_types: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    chunk_types: Optional[List[str]] = None

class HybridSearch:
    weaviate_client: Any
    lexical_index: Any
    semantic_weight: float  # Automatically normalized if weights don't sum to 1.0
    lexical_weight: float  # Automatically normalized if weights don't sum to 1.0
    enable_compression: bool
    token_counter: Optional[SmartTokenCounter]
    compressor: Optional[ContextualCompressor]
    cache: SearchCache

    def __init__(
        self,
        weaviate_client: Any,
        lexical_index: Any = None,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        enable_compression: bool = True,
    ) -> None: ...
    async def search(
        self, query: str, max_chunks: int = 10, filters: Optional[SearchFilters] = None
    ) -> List[ScoredChunk]: ...
    async def search_with_compression(
        self,
        query: str,
        max_chunks: int = 10,
        token_budget: Optional[int] = None,
        compression_ratio: Optional[float] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[Chunk]: ...
    async def search_with_graph_expansion(
        self,
        query: str,
        max_results: int = 10,
        expansion_depth: int = 2,
        filters: Optional[SearchFilters] = None,
    ) -> List[Chunk]: ...
    def invalidate_cache(self, pattern: Optional[str] = None) -> None: ...
    def invalidate_cache_for_file(self, file_path: str) -> None: ...
    def get_cache_stats(self) -> Dict[str, Any]: ...
    async def _semantic_search(
        self, query: str, limit: int, filters: Optional[SearchFilters] = None
    ) -> List[ScoredChunk]: ...
    async def _lexical_search(
        self, query: str, limit: int, filters: Optional[SearchFilters] = None
    ) -> List[ScoredChunk]: ...
    def _combine_results(
        self, semantic_results: List[ScoredChunk], lexical_results: List[ScoredChunk]
    ) -> List[ScoredChunk]: ...
    def _normalize_scores(self, results: List[ScoredChunk]) -> List[ScoredChunk]: ...
    async def _load_chunks_from_file(self, file_path: str) -> List[Chunk]: ...
    async def _rerank_by_relevance(
        self, query: str, chunks: List[Chunk], max_results: int
    ) -> List[Chunk]: ...
