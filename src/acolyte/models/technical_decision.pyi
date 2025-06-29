from enum import Enum
from typing import List, Optional
from pydantic import Field
from acolyte.models.base import AcolyteBaseModel, TimestampMixin, IdentifiableMixin

class DecisionType(str, Enum):
    ARCHITECTURE = "architecture"
    LIBRARY = "library"
    PATTERN = "pattern"
    SECURITY = "security"

class TechnicalDecision(AcolyteBaseModel, TimestampMixin, IdentifiableMixin):
    task_id: Optional[str] = Field(None)
    session_id: str = Field(...)
    decision_type: DecisionType = Field(...)
    title: str = Field(..., max_length=200)
    description: str = Field(...)
    rationale: str = Field(...)
    alternatives_considered: List[str] = Field(default_factory=list)
    impact_level: int = Field(..., ge=1, le=5)
    code_references: List[str] = Field(default_factory=list)

    def get_summary(self) -> str: ...
    def to_search_text(self) -> str: ...
