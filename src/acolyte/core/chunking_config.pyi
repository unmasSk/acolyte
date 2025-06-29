"""
Chunking strategies configuration for ACOLYTE.
"""

from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass

class ChunkingStrategy(Enum):
    SEMANTIC: str
    HIERARCHICAL: str
    FIXED_SIZE: str
    ADAPTIVE: str

@dataclass
class StrategyConfig:
    chunk_size: int
    overlap: int
    min_chunk_size: int
    max_chunk_size: int
    preserve_structure: bool
    language_specific_rules: Dict[str, Any]

@dataclass
class ValidationResult:
    is_valid: bool
    score: float
    problems: List[str]

class ChunkingConfig:
    DEFAULT_CONFIGS: Dict[str, Dict[str, int]]

    @staticmethod
    def get_strategy_config(strategy: ChunkingStrategy, language: str) -> StrategyConfig: ...
    @staticmethod
    def validate_chunk(
        chunk: str, min_size: int, max_size: int, language: str
    ) -> ValidationResult: ...
