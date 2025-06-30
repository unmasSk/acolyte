from typing import List
from acolyte.models.semantic_types import SessionReference
from acolyte.core.tracing import MetricsCollector

class ReferenceResolver:
    SPANISH_PATTERNS: List[str]
    ENGLISH_PATTERNS: List[str]
    reference_patterns: List[str]
    metrics: MetricsCollector

    def __init__(self) -> None: ...
    def resolve_temporal_references(self, message: str) -> List[SessionReference]: ...
    def _detect_specific_references(self, message: str) -> List[SessionReference]: ...
