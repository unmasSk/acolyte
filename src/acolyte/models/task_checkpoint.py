"""
Models for task checkpoints.
Groups multiple sessions related to a project/task.
"""

from enum import Enum
from typing import List, Optional
from pydantic import Field
from acolyte.models.base import AcolyteBaseModel, TimestampMixin, IdentifiableMixin


class TaskType(str, Enum):
    """Task types that ACOLYTE groups."""

    IMPLEMENTATION = "implementation"  # Create new functionality
    DEBUGGING = "debugging"  # Resolve bugs
    REFACTORING = "refactoring"  # Improve existing code
    DOCUMENTATION = "documentation"  # Write docs
    RESEARCH = "research"  # Investigate solutions
    REVIEW = "review"  # Review code


class TaskStatus(str, Enum):
    """Task states."""

    PLANNING = "planning"  # Defining what to do
    IN_PROGRESS = "in_progress"  # Working on it
    COMPLETED = "completed"  # Task finished


class TaskCheckpoint(AcolyteBaseModel, TimestampMixin, IdentifiableMixin):
    """
    Task checkpoint that groups sessions.

    One task = days/weeks of work on a specific topic.
    Allows recovering ALL context with: "What were we doing with auth?"
    """

    # Identification
    title: str = Field(..., max_length=200, description="Descriptive title")
    description: str = Field(..., description="Detailed description")
    task_type: TaskType = Field(..., description="Task type")

    # State
    status: TaskStatus = Field(TaskStatus.PLANNING, description="Current state")

    # Starting context
    initial_context: str = Field(..., description="Initial task context")
    initial_session_id: Optional[str] = Field(
        None, description="The first session that originated this task"
    )

    # Associated sessions
    session_ids: List[str] = Field(default_factory=list, description="IDs of all related sessions")

    # Key context
    key_decisions: List[str] = Field(default_factory=list, description="Important decisions made")

    # For search
    keywords: List[str] = Field(default_factory=list, description="Keywords to find this task")

    def add_session(self, session_id: str) -> None:
        """Associates a new session to this task."""
        if session_id not in self.session_ids:
            self.session_ids.append(session_id)
            self.touch()

    def add_decision(self, decision: str) -> None:
        """Records an important decision."""
        self.key_decisions.append(decision)
        self.touch()

    def complete(self) -> None:
        """Marks the task as completed."""
        self.status = TaskStatus.COMPLETED
        self.touch()

    def get_summary(self) -> str:
        """Generates task summary for quick search.

        STANDARD INTERFACE for task summary used by:
        - Related task search
        - Conversation context summary
        - Dashboard and reporting
        """
        return (
            f"{self.title} ({self.task_type}) - "
            f"{len(self.session_ids)} sessions - "
            f"Status: {self.status}"
        )

    def to_search_text(self) -> str:
        """Generates text for semantic task search.

        STANDARD INTERFACE compatible with embeddings.
        """
        parts = [
            f"Task: {self.title}",
            f"Type: {self.task_type}",
            f"Status: {self.status}",
        ]

        if self.description:
            parts.append(f"Description: {self.description}")

        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")

        if self.key_decisions:
            parts.append(f"Decisions: {', '.join(self.key_decisions[:3])}")

        return " ".join(parts)
