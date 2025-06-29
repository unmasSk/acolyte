from typing import List, Optional, Tuple, Union, Any
import torch
from acolyte.core.secure_config import Settings
from acolyte.core.database import DatabaseManager
from acolyte.embeddings.types import EmbeddingVector, EmbeddingsMetricsSummaryDict, MetricsProvider
from acolyte.embeddings.persistent_cache import SmartPersistentCache
from acolyte.embeddings.context import RichCodeContext
from acolyte.models.chunk import Chunk

class UniXcoderEmbeddings:
    config: Settings
    cache: SmartPersistentCache
    metrics: Optional[MetricsProvider]
    model_name: str
    max_length: int
    batch_size: int
    _model: Optional[Any]
    _tokenizer: Optional[Any]
    _device: Optional[torch.device]
    _reranker: Optional[Any]
    _db: Optional[DatabaseManager]

    def __init__(self, metrics: Optional[MetricsProvider] = ...) -> None: ...
    def _validate_config(self) -> None: ...
    async def _get_db(self) -> DatabaseManager: ...
    def _load_device_state(self) -> None: ...
    async def _save_device_state(self, device_str: str) -> None: ...
    @property
    def device(self) -> torch.device: ...
    def _load_model(self) -> None: ...
    def _prepare_input(
        self, text: Union[str, Chunk], context: Optional[RichCodeContext]
    ) -> str: ...
    def _prepare_plain_text(self, text: str, context: Optional[RichCodeContext]) -> str: ...
    def encode(
        self, text: Union[str, Chunk], context: Optional[RichCodeContext] = ...
    ) -> EmbeddingVector: ...
    def encode_batch(
        self,
        texts: List[Union[str, Chunk]],
        contexts: Optional[List[Optional[RichCodeContext]]] = ...,
        batch_size: Optional[int] = ...,
        max_tokens_per_batch: Optional[int] = ...,
    ) -> List[EmbeddingVector]: ...
    def _encode_batch_legacy(
        self,
        texts: List[Union[str, Chunk]],
        contexts: List[Optional[RichCodeContext]],
        batch_size: int,
    ) -> List[EmbeddingVector]: ...
    def _estimate_tokens(self, text: Union[str, Chunk]) -> int: ...
    def _encode_batch_token_limited(
        self,
        texts: List[Union[str, Chunk]],
        contexts: List[Optional[RichCodeContext]],
        max_tokens_per_batch: int,
    ) -> List[EmbeddingVector]: ...
    def _process_batch_tokens(
        self,
        batch_texts: List[Union[str, Chunk]],
        batch_contexts: List[Optional[RichCodeContext]],
        batch_indices: List[int],
        all_texts: List[Union[str, Chunk]],
        all_contexts: List[Optional[RichCodeContext]],
        results: List[Optional[EmbeddingVector]],
    ) -> None: ...
    def _get_reranker(self) -> Any: ...
    def encode_with_rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: int = ...,
        initial_retrieval_factor: int = ...,
    ) -> List[Tuple[str, float]]: ...
    def get_metrics_summary(self) -> EmbeddingsMetricsSummaryDict: ...
