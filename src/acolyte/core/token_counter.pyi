"""
Token limit management for ACOLYTE.

Complete implementation of token counting and management system.
Uses simple estimation instead of async calls to Ollama to avoid complexity.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

class TruncateStrategy(Enum):
    """Text truncation strategies."""

    END = "end"
    START = "start"
    MIDDLE = "middle"
    SMART = "smart"

@dataclass
class TokenCount:
    """Token count result."""

    total: int
    content: int
    special: int
    estimated: bool = ...

@dataclass
class ContextSplit:
    """Context split result."""

    included: List[Any]
    truncated: List[Any]
    excluded: List[Any]
    token_usage: Dict[str, int]

class TokenEncoder(ABC):
    """Interface for different token encoders."""

    @abstractmethod
    def encode(self, text: str) -> List[int]: ...
    @abstractmethod
    def decode(self, tokens: List[int]) -> str: ...

class OllamaEncoder(TokenEncoder):
    """
    Encoder using simple estimation (100% local, no async complexity).
    """

    base_url: str

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None: ...
    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...

class SmartTokenCounter:
    """
    Smart token counter with optimizations.

    Features:
    1. Cache of frequent counts
    2. Fast estimation for long texts
    3. Multi-model support
    4. Special token management
    """

    encoders: Dict[str, TokenEncoder]
    default_encoder: str
    _count_cache: Dict[Any, TokenCount]
    _default_encoder_instance: Optional[OllamaEncoder]

    def __init__(self) -> None: ...
    def _get_encoder(self, model: Optional[str] = None) -> TokenEncoder: ...
    def count(
        self, text: str, model: Optional[str] = None, include_special: bool = True
    ) -> TokenCount: ...
    def count_tokens(self, text: str, model: Optional[str] = None) -> int: ...
    def truncate_to_limit(
        self, text: str, limit: int, model: Optional[str] = None, strategy: TruncateStrategy = ...
    ) -> str: ...
    def split_for_context(
        self, messages: List[Dict[str, Any]], max_context: int, reserve_output: int = 1000
    ) -> ContextSplit: ...

class TokenBudgetManager:
    """
    Manages token budget for operations.

    Useful for:
    1. RAG with multiple sources
    2. Long conversations
    3. Batch processing
    4. Dynamic distribution by query type
    """

    total_budget: int
    allocations: Dict[str, int]
    used: Dict[str, int]
    priorities: Dict[str, float]

    def __init__(self, total_budget: int) -> None: ...
    def allocate(self, category: str, tokens: int) -> bool: ...
    def use(self, category: str, tokens: int) -> bool: ...
    def get_remaining(self, category: str) -> int: ...
    def allocate_for_dream_cycle(
        self, model_size: str, cycle_number: int = 1
    ) -> Dict[str, int]: ...
    def get_distribution(self) -> Dict[str, Any]: ...
    def optimize_allocations(self, priorities: Optional[Dict[str, float]] = None) -> None: ...
    def allocate_for_query_type(self, query_type: str) -> Dict[str, int]: ...

_default_counter: Optional[SmartTokenCounter]

def get_token_counter() -> SmartTokenCounter: ...
