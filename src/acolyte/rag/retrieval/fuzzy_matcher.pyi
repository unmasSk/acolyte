from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
import re

@dataclass
class FuzzyVariation:
    term: str
    variation_type: str
    confidence: float = 1.0

class FuzzyMatcher:
    enabled: bool
    max_variations: int
    min_term_length: int
    snake_pattern: re.Pattern[str]
    camel_pattern: re.Pattern[str]
    pascal_pattern: re.Pattern[str]
    kebab_pattern: re.Pattern[str]

    def __init__(self) -> None: ...
    def expand_query(self, query: str) -> List[str]: ...
    def find_similar_terms(
        self, term: str, candidates: List[str], threshold: float = 0.7
    ) -> List[Tuple[str, float]]: ...
    def _extract_identifiers(self, query: str) -> Set[str]: ...
    def _generate_variations(self, identifier: str) -> List[FuzzyVariation]: ...
    def _detect_style(self, identifier: str) -> str: ...
    def _parse_identifier(self, identifier: str, style: str) -> List[str]: ...
    def _generate_query_variations(
        self, query: str, identifier_variations: Dict[str, List[FuzzyVariation]]
    ) -> List[str]: ...

def get_fuzzy_matcher() -> FuzzyMatcher: ...

_fuzzy_matcher: Optional[FuzzyMatcher]
