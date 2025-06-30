import re
from dataclasses import dataclass
from typing import Dict, List, Set

from acolyte.models.chunk import Chunk

@dataclass
class QueryContext:
    query: str
    query_type: str
    query_tokens: Set[str]
    entities: Dict[str, List[str]]
    compression_needed: bool
    suggested_ratio: float

    @property
    def is_specific(self) -> bool: ...
    @property
    def is_general(self) -> bool: ...
    @property
    def is_generation(self) -> bool: ...

class QueryAnalyzer:
    max_specific_length: int
    patterns: Dict[str, List[re.Pattern[str]]]
    general_keywords: Set[str]
    specific_indicators: Set[str]
    generation_keywords: Set[str]

    def __init__(self, max_specific_length: int = ..., cache_size: int = ...) -> None: ...
    def analyze_query(self, query: str) -> QueryContext: ...
    def calculate_relevance(self, chunk: Chunk, context: QueryContext) -> float: ...
    def clear_cache(self) -> None: ...
