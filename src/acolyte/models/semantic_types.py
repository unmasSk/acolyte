"""
Types for the Semantic module.

Defines structures needed for language processing.
"""

from dataclasses import dataclass
from typing import Literal, Optional, List


@dataclass
class TokenDistribution:
    """Query analysis result for token distribution."""

    type: Literal["generation", "simple", "normal"]
    response_ratio: float
    context_ratio: float


@dataclass
class TaskDetection:
    """Task detection result."""

    is_new_task: bool
    task_title: Optional[str]
    confidence: float
    continues_task: Optional[str] = None


@dataclass
class SummaryResult:
    """Summary generation result."""

    summary: str
    entities: List[str]
    intent_type: str
    tokens_saved: int


@dataclass
class SessionReference:
    """Reference to previous session detected."""

    pattern_matched: str
    context_hint: Optional[str]
    search_type: Literal["temporal"]


@dataclass
class DetectedDecision:
    """
    Technical decision detected by text analysis (DTO).

    DATA TRANSFORMATION PATTERN:
    1. DecisionDetector analyzes text → DetectedDecision (without session context)
    2. ChatService completes context → TechnicalDecision (for persistence)

    This type does NOT include session_id/task_id because DecisionDetector
    should not know session context (separation of concerns).
    ChatService adds the IDs before persisting.

    NOT duplication of TechnicalDecision - they are different types:
    - DetectedDecision: DTO without context (pure analysis)
    - TechnicalDecision: Complete entity with IDs for DB
    """

    decision_type: str  # DecisionType value
    title: str
    description: str
    rationale: str
    alternatives_considered: List[str]
    impact_level: int  # 1-5
