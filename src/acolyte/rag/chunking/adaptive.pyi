from typing import List, Any
from dataclasses import dataclass
from enum import Enum
from acolyte.models.chunk import Chunk
from acolyte.core.secure_config import Settings

class ChunkingStrategy(Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    BALANCED = "balanced"
    DOCUMENTATION = "documentation"

@dataclass
class ContentAnalysis:
    total_nodes: int
    function_nodes: int
    class_nodes: int
    import_nodes: int
    comment_ratio: float
    avg_function_size: float
    max_function_size: int
    has_complex_nesting: bool
    is_test_file: bool
    recommended_strategy: ChunkingStrategy

class AdaptiveChunker:
    config: Settings
    _current_strategy: ChunkingStrategy
    _current_language: str
    _current_chunker: Any
    _base_chunk_size: int
    _base_overlap: float

    def __init__(self) -> None: ...
    async def chunk(self, content: str, file_path: str) -> List[Chunk]: ...
    def _analyze_ast(self, tree: Any, file_path: str) -> ContentAnalysis: ...
    def _adjust_parameters(self, analysis: ContentAnalysis) -> None: ...
