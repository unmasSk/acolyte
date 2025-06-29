from dataclasses import dataclass
import numpy as np
from typing import List, Union, Dict, Any, TypedDict, Protocol, runtime_checkable

@dataclass
class EmbeddingVector:
    _data: np.ndarray
    def __init__(self, data: Union[np.ndarray, List[float]]) -> None: ...
    @property
    def numpy(self) -> np.ndarray: ...
    @property
    def list(self) -> List[float]: ...
    def to_weaviate(self) -> List[float]: ...
    @property
    def dimension(self) -> int: ...
    def validate(self) -> bool: ...
    def cosine_similarity(self, other: EmbeddingVector) -> float: ...

class EmbeddingsMetricsSummaryDict(TypedDict, total=False):
    cache_size: int
    cache_ttl_seconds: int
    model_loaded: bool
    device: str
    metrics_enabled: bool
    cache_persistence: Dict[str, Any]
    p95_latency_ms: float
    cache_hit_rate: float

class RerankerMetricsSummary(TypedDict, total=False):
    model_loaded: bool
    cache_size: int
    device: str
    metrics_enabled: bool
    cache_hit_rate: float
    total_rerank_operations: int

@runtime_checkable
class MetricsProvider(Protocol):
    def record_operation(self, operation: str, latency_ms: float, success: bool = ...) -> None: ...
    def record_cache_hit(self) -> None: ...
    def record_cache_miss(self) -> None: ...
    def get_cache_hit_rate(self) -> float: ...
    def get_p95_latency(self) -> float: ...
