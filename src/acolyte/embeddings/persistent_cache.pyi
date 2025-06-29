from typing import Dict, Optional, Union, Any
from pathlib import Path
from acolyte.embeddings.cache import ContextAwareCache
from acolyte.embeddings.types import EmbeddingVector
from acolyte.embeddings.context import RichCodeContext
from acolyte.models.chunk import Chunk

class SmartPersistentCache(ContextAwareCache):
    CACHE_VERSION: int = 1
    _dirty: bool
    _save_interval: int
    def __init__(
        self,
        max_size: int = ...,
        ttl_seconds: int = ...,
        save_interval: int = ...,
        cache_dir: Optional[Path] = ...,
    ) -> None: ...
    def save_to_disk(self) -> None: ...
    def set(
        self,
        text: Union[str, Chunk],
        context: Optional[RichCodeContext],
        embedding: EmbeddingVector,
    ) -> None: ...
    def clear(self) -> None: ...
    def cleanup_expired(self) -> int: ...
    def close(self) -> None: ...
    def __enter__(self) -> SmartPersistentCache: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...
    def get_persistent_stats(self) -> Dict[str, Any]: ...
