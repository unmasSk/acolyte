from enum import Enum
from typing import List, Optional
from datetime import datetime
from typing_extensions import Self
from pydantic import Field, model_validator
from acolyte.models.base import AcolyteBaseModel, TimestampMixin, SessionIdMixin
from acolyte.models.chat import Message

class ConversationStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"

class Conversation(AcolyteBaseModel, TimestampMixin, SessionIdMixin):
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE)
    messages: List[Message] = Field(default_factory=list)
    summary: Optional[str] = Field(None)
    keywords: List[str] = Field(default_factory=list)
    related_sessions: List[str] = Field(default_factory=list)
    task_checkpoint_id: Optional[str] = Field(None)
    total_tokens: int = Field(default=0)
    message_count: int = Field(default=0)

    def add_message(self, message: Message) -> None: ...
    def get_context_window(self, max_messages: int = 10) -> List[Message]: ...
    def complete(self) -> None: ...

class ConversationSearchRequest(AcolyteBaseModel):
    query: str = Field(...)
    limit: int = Field(default=5, ge=1, le=100)
    include_completed: bool = Field(default=True)
    task_id: Optional[str] = Field(None)
    date_from: Optional[datetime] = Field(None)
    date_to: Optional[datetime] = Field(None)

    @model_validator(mode='after')
    def validate_date_range(self) -> Self: ...

class ConversationSearchResult(AcolyteBaseModel):
    conversation_id: str
    session_id: str
    relevance_score: float = Field(...)
    summary: str
    keywords: List[str]
    message_count: int
    created_at: datetime
    relevant_messages: List[Message] = Field(default_factory=list)
