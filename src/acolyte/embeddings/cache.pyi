from dataclasses import dataclass
from typing import Optional, Union, TypedDict
from acolyte.embeddings.types import EmbeddingVector
from acolyte.embeddings.context import RichCodeContext
from acolyte.models.chunk import Chunk

class CacheStats(TypedDict):
    size: int
    max_size: int
    ttl_seconds: float
    oldest_entry_age: float
    newest_entry_age: float
    capacity_used: float
    cleanup_interval: int
    operations_until_cleanup: int

@dataclass
class CacheEntry:
    embedding: EmbeddingVector
    created_at: float

class ContextAwareCache:
    max_size: int
    ttl_seconds: float
    def __init__(self, max_size: int = ..., ttl_seconds: float = ...) -> None: ...
    def _generate_key(self, text: Union[str, Chunk], context: Optional[RichCodeContext]) -> str: ...
    def get(
        self, text: Union[str, Chunk], context: Optional[RichCodeContext]
    ) -> Optional[EmbeddingVector]: ...
    def set(
        self,
        text: Union[str, Chunk],
        context: Optional[RichCodeContext],
        embedding: EmbeddingVector,
    ) -> None: ...
    def clear(self) -> None: ...
    def cleanup_expired(self) -> int: ...
    @property
    def size(self) -> int: ...
    def get_stats(self) -> CacheStats: ...
