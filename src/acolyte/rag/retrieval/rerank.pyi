"""
Simple re-ranking strategies for search results type stubs.
"""

from typing import List, Optional, Dict, Any

from acolyte.models.chunk import ChunkType
from acolyte.rag.retrieval.hybrid_search import ScoredChunk

class SimpleReranker:
    CHUNK_TYPE_PRIORITY: Dict[ChunkType, int]

    def rerank_by_recency(
        self, results: List[ScoredChunk], decay_factor: float = 0.95
    ) -> List[ScoredChunk]: ...
    def boost_modified_files(
        self,
        results: List[ScoredChunk],
        boost_factor: float = 1.2,
        modified_files: Optional[List[str]] = None,
    ) -> List[ScoredChunk]: ...
    def prioritize_by_chunk_type(
        self, results: List[ScoredChunk], priority_types: Optional[List[ChunkType]] = None
    ) -> List[ScoredChunk]: ...
    def rerank(
        self, results: List[ScoredChunk], strategy: str = "mixed", **kwargs: Any
    ) -> List[ScoredChunk]: ...
    def diversity_rerank(
        self, results: List[ScoredChunk], soft_max_per_file: int = 3
    ) -> List[ScoredChunk]: ...
