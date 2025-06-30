"""
Standard types for the embeddings module.

Defines EmbeddingVector as the unique embedding format throughout ACOLYTE,
and TypedDicts to avoid circular dependencies.
"""

from dataclasses import dataclass
from typing import List, Union, Dict, Any, TypedDict, Protocol, runtime_checkable
import numpy as np
from acolyte.core.logging import logger


@dataclass
class EmbeddingVector:
    """Standard representation of embeddings in ACOLYTE.

    Internally uses NumPy for efficiency.
    Exports to list for Weaviate/serialization.

    IMPORTANT: This is the ONLY embedding format throughout ACOLYTE.

    Attributes:
        _data: Normalized NumPy array of 768 dimensions (float32)
    """

    _data: np.ndarray

    def __init__(self, data: Union[np.ndarray, List[float]]):
        """Initializes the embedding with validation and normalization.

        Args:
            data: Vector as NumPy array or list of floats

        Raises:
            ValueError: If the vector does not have 768 dimensions
        """
        # Always NumPy float32 internally
        if isinstance(data, list):
            self._data = np.array(data, dtype=np.float32)
        else:
            self._data = data.astype(np.float32)

        # Always normalize (architectural decision)
        self._normalize()

        # Robust dimension validation
        if len(self._data.shape) != 1 or self._data.shape[0] != 768:
            actual_shape = self._data.shape if len(self._data.shape) > 0 else "scalar"
            raise ValueError(f"Embedding must have 768 dimensions, has {actual_shape}")

    def _normalize(self):
        """L2 normalization for cosine similarity.

        If the vector is zero (norm = 0), it becomes a unit vector
        in the first dimension to avoid division by zero and maintain
        the normalization property.
        """
        norm = np.linalg.norm(self._data)
        if norm > 0:
            self._data = self._data / norm
        else:
            # Zero vector has no direction, assign default direction
            # Unit vector in first dimension [1, 0, 0, ...]
            logger.warning("Normalizing zero vector, using default unit vector")
            self._data = np.zeros(768, dtype=np.float32)
            self._data[0] = 1.0

    @property
    def numpy(self) -> np.ndarray:
        """For efficient mathematical operations."""
        return self._data

    @property
    def list(self) -> List[float]:
        """For generic serialization."""
        return self._data.tolist()

    def to_weaviate(self) -> List[float]:
        """Weaviate needs float64 to avoid known bugs.

        Returns:
            List of floats in float64 for Weaviate
        """
        return self._data.astype(np.float64).tolist()

    @property
    def dimension(self) -> int:
        """Always 768 - system constant."""
        return 768

    def validate(self) -> bool:
        """Full format validation.

        Returns:
            True if the embedding is valid (768 dims, float32, normalized)
        """
        return bool(
            self._data.shape == (768,)
            and self._data.dtype == np.float32
            and bool(np.isclose(np.linalg.norm(self._data), 1.0, atol=1e-5))
        )

    def cosine_similarity(self, other: "EmbeddingVector") -> float:
        """Efficiently computes cosine similarity.

        Since both are normalized, it's just the dot product.

        Args:
            other: Another EmbeddingVector to compare

        Returns:
            Cosine similarity between 0.0 and 1.0
        """
        return float(np.dot(self._data, other._data))


# TypedDicts to avoid circular dependencies
# These are imported from here instead of from the modules that define them


class EmbeddingsMetricsSummaryDict(TypedDict, total=False):
    """Metrics summary of the embeddings module.

    total=False because some fields are optional depending
    on whether metrics are enabled.
    """

    cache_size: int
    cache_ttl_seconds: int
    model_loaded: bool
    device: str
    metrics_enabled: bool
    cache_persistence: Dict[str, Any]  # Optional, only if the cache supports it
    p95_latency_ms: float  # Only if metrics_enabled
    cache_hit_rate: float  # Only if metrics_enabled


class RerankerMetricsSummary(TypedDict, total=False):
    """Metrics summary of the re-ranker.

    total=False because some fields are optional.
    """

    model_loaded: bool
    cache_size: int
    device: str
    metrics_enabled: bool
    cache_hit_rate: float  # Only if metrics_enabled
    total_rerank_operations: int  # Only if metrics_enabled


# Protocol to prevent circular dependencies
@runtime_checkable
class MetricsProvider(Protocol):
    """Protocol for metrics providers.

    Defines the minimal interface that any metrics system must implement
    to be used in the embeddings module. This prevents circular dependencies
    by defining the contract regardless of implementation.

    Usage:
        - types.py defines the Protocol (no metrics imports)
        - metrics.py implements the Protocol (no consumer imports)
        - unixcoder/reranker consume via Protocol (type-safe)
    """

    def record_operation(self, operation: str, latency_ms: float, success: bool = True) -> None:
        """Records an operation with its latency."""
        ...

    def record_cache_hit(self) -> None:
        """Records a cache hit."""
        ...

    def record_cache_miss(self) -> None:
        """Records a cache miss."""
        ...

    def get_cache_hit_rate(self) -> float:
        """Returns the cache hit rate."""
        ...

    def get_p95_latency(self) -> float:
        """Returns the p95 latency in milliseconds."""
        ...
