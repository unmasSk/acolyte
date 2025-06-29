from dataclasses import dataclass
from typing import Literal, Optional, List

@dataclass
class TokenDistribution:
    type: Literal["generation", "simple", "normal"]
    response_ratio: float
    context_ratio: float

@dataclass
class TaskDetection:
    is_new_task: bool
    task_title: Optional[str]
    confidence: float
    continues_task: Optional[str] = None

@dataclass
class SummaryResult:
    summary: str
    entities: List[str]
    intent_type: str
    tokens_saved: int

@dataclass
class SessionReference:
    pattern_matched: str
    context_hint: Optional[str]
    search_type: Literal["temporal"]

@dataclass
class DetectedDecision:
    decision_type: str
    title: str
    description: str
    rationale: str
    alternatives_considered: List[str]
    impact_level: int
