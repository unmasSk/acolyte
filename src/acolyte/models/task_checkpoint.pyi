from enum import Enum
from typing import List, Optional
from pydantic import Field
from acolyte.models.base import AcolyteBaseModel, TimestampMixin, IdentifiableMixin

class TaskType(str, Enum):
    IMPLEMENTATION = "implementation"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    REVIEW = "review"

class TaskStatus(str, Enum):
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class TaskCheckpoint(AcolyteBaseModel, TimestampMixin, IdentifiableMixin):
    title: str = Field(..., max_length=200)
    description: str = Field(...)
    task_type: TaskType = Field(...)
    status: TaskStatus = Field(default=TaskStatus.PLANNING)
    initial_context: str = Field(...)
    initial_session_id: Optional[str] = Field(None)
    session_ids: List[str] = Field(default_factory=list)
    key_decisions: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

    def add_session(self, session_id: str) -> None: ...
    def add_decision(self, decision: str) -> None: ...
    def complete(self) -> None: ...
    def get_summary(self) -> str: ...
    def to_search_text(self) -> str: ...
