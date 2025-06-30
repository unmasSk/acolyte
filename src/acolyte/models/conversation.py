"""
Conversation persistence models.
Manages sessions with associative memory.
"""

from enum import Enum
from typing import List, Optional
from datetime import datetime
from typing_extensions import Self
from pydantic import Field, model_validator
from acolyte.models.base import AcolyteBaseModel, TimestampMixin, SessionIdMixin
from acolyte.models.chat import Message


class ConversationStatus(str, Enum):
    """Conversation states."""

    ACTIVE = "active"  # In progress
    COMPLETED = "completed"  # Archived


class Conversation(AcolyteBaseModel, TimestampMixin, SessionIdMixin):
    """
    Individual conversation (session).
    One session = 1-2 hours of work.

    IMPORTANT: Uses SessionIdMixin to implement Identifiable protocol.
    The session_id is automatically inherited from the mixin.
    """

    # State
    status: ConversationStatus = Field(ConversationStatus.ACTIVE, description="Conversation state")

    # Messages
    messages: List[Message] = Field(default_factory=list, description="Message history")

    # Context for associative search
    summary: Optional[str] = Field(None, description="Conversation summary")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords for search")

    # References to other related sessions
    related_sessions: List[str] = Field(default_factory=list, description="Related session IDs")

    # Associated task (if exists)
    task_checkpoint_id: Optional[str] = Field(None, description="Parent task ID")

    # Metrics
    total_tokens: int = Field(0, description="Total tokens used")
    message_count: int = Field(0, description="Number of messages")

    def add_message(self, message: Message) -> None:
        """Adds a message to the conversation."""
        self.messages.append(message)
        self.message_count += 1

        # Update tokens if metadata available
        if message.metadata and "tokens" in message.metadata:
            self.total_tokens += message.metadata["tokens"]

        self.touch()

    def get_context_window(self, max_messages: int = 10) -> List[Message]:
        """
        Gets most recent messages for context.
        Useful for LLM context windows.
        """
        return self.messages[-max_messages:] if self.messages else []

    def complete(self) -> None:
        """Marks conversation as completed."""
        self.status = ConversationStatus.COMPLETED
        self.touch()


class ConversationSearchRequest(AcolyteBaseModel):
    """
    Request to search related conversations.
    Used by associative memory system.
    """

    query: str = Field(..., description="Semantic search")
    limit: int = Field(5, ge=1, le=100, description="Number of results")
    include_completed: bool = Field(True, description="Include archived conversations")

    # Optional filters
    task_id: Optional[str] = Field(None, description="Filter by task")
    date_from: Optional[datetime] = Field(None, description="Minimum date")
    date_to: Optional[datetime] = Field(None, description="Maximum date")

    @model_validator(mode='after')
    def validate_date_range(self) -> Self:
        """Ensure date_from is not later than date_to."""
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from cannot be later than date_to")
        return self


class ConversationSearchResult(AcolyteBaseModel):
    """Conversation search result."""

    conversation_id: str
    session_id: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    summary: str
    keywords: List[str]
    message_count: int
    created_at: datetime

    # Relevant context
    relevant_messages: List[Message] = Field(
        default_factory=list, description="Most relevant messages"
    )
