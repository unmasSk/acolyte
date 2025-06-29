from typing import Any, Dict, List, Set

from acolyte.core.tracing import MetricsCollector
from acolyte.models.chunk import Chunk
from acolyte.rag.graph import NeuralGraph

class GraphBuilder:
    graph: NeuralGraph
    _processed_files: Set[str]
    metrics: MetricsCollector

    def __init__(self) -> None: ...
    async def update_from_chunks(self, chunks: List[Chunk], metadata: Dict[str, Any]) -> None: ...
    async def extract_relationships_from_ast(
        self, chunk: Chunk, language: str = ...
    ) -> Dict[str, List[str]]: ...
