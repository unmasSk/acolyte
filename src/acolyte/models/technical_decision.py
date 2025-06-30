"""
Model for important technical decisions.
Records architectural, library, pattern and security decisions.
"""

from enum import Enum
from typing import List, Optional
from pydantic import Field
from acolyte.models.base import AcolyteBaseModel, TimestampMixin, IdentifiableMixin


class DecisionType(str, Enum):
    """Types of technical decisions."""

    ARCHITECTURE = "architecture"  # Architecture decisions
    LIBRARY = "library"  # Library/framework choices
    PATTERN = "pattern"  # Design patterns
    SECURITY = "security"  # Security decisions


class TechnicalDecision(AcolyteBaseModel, TimestampMixin, IdentifiableMixin):
    """
    Important technical decision recorded during development.

    Automatically detected by patterns or explicit marking (@decision).
    Allows tracking the "why" of decisions made.
    """

    # Context
    task_id: Optional[str] = Field(None, description="Task where decision was made")
    session_id: str = Field(..., description="Conversation session")

    # Decision
    decision_type: DecisionType = Field(..., description="Decision type")
    title: str = Field(..., max_length=200, description="Concise title")
    description: str = Field(..., description="Detailed description")
    rationale: str = Field(..., description="Why this decision was made")

    # Analysis
    alternatives_considered: List[str] = Field(
        default_factory=list, description="Alternatives evaluated"
    )
    impact_level: int = Field(..., ge=1, le=5, description="Impact level (1=low, 5=critical)")

    # References
    code_references: List[str] = Field(
        default_factory=list, description="Affected files or functions"
    )

    def get_summary(self) -> str:
        """Generates summary for quick search.

        STANDARD INTERFACE for decision summary used by:
        - Related decision search
        - Conversation context
        - Technical decision reports
        """
        alts = (
            f" (vs {', '.join(self.alternatives_considered)})"
            if self.alternatives_considered
            else ""
        )
        return f"{self.title}{alts} - {self.decision_type} - Impact: {self.impact_level}/5"

    def to_search_text(self) -> str:
        """Generates text for semantic decision search.

        STANDARD INTERFACE compatible with embeddings.
        """
        parts = [
            f"Decision: {self.title}",
            f"Type: {self.decision_type}",
            f"Impact: {self.impact_level}/5",
        ]

        if self.description:
            parts.append(f"Description: {self.description}")

        if self.rationale:
            parts.append(f"Rationale: {self.rationale}")

        if self.alternatives_considered:
            parts.append(f"Alternatives: {', '.join(self.alternatives_considered[:3])}")

        if self.code_references:
            parts.append(f"Files: {', '.join(self.code_references[:3])}")

        return " ".join(parts)
