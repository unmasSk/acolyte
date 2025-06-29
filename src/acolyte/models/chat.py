"""
OpenAI-compatible chat models.
Defines structures for requests and responses of the main endpoint.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import Field, field_validator
from acolyte.models.base import AcolyteBaseModel


class Role(str, Enum):
    """Valid roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(AcolyteBaseModel):
    """
    Individual message in a conversation.
    OpenAI format compatible.
    """

    role: Role = Field(..., description="Sender's role")
    content: str = Field(..., min_length=1, description="Message content")

    # Optional metadata for internal tracking
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Internal metadata (tokens, timing, etc)"
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensures content is not empty."""
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v


class ChatRequest(AcolyteBaseModel):
    """
    Request for /v1/chat/completions endpoint.
    100% OpenAI compatible.
    """

    # Standard OpenAI fields
    model: str = Field(..., description="Requested model (ignored, always acolyte:latest)")
    messages: List[Message] = Field(..., min_length=1, description="Message history")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Model creativity")
    max_tokens: Optional[int] = Field(None, gt=0, description="Response token limit")
    stream: bool = Field(False, description="Response streaming")

    # ACOLYTE-specific fields (optional)
    debug: bool = Field(False, description="Include debug information")
    explain_rag: bool = Field(False, description="Explain RAG process")

    @field_validator("messages")
    @classmethod
    def validate_message_flow(cls, messages: List[Message]) -> List[Message]:
        """Validates message flow is coherent."""
        if not messages:
            raise ValueError("Must have at least one message")

        # First message must be user or system
        if messages[0].role not in [Role.USER, Role.SYSTEM]:
            raise ValueError("Conversation must start with user or system")

        return messages


class Choice(AcolyteBaseModel):
    """Individual choice in response (OpenAI format)."""

    message: Message
    finish_reason: str = "stop"
    index: int = 0


class Usage(AcolyteBaseModel):
    """Token usage information (OpenAI format)."""

    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)


class ChatResponse(AcolyteBaseModel):
    """
    Response from /v1/chat/completions endpoint.
    100% OpenAI compatible.
    """

    # Standard OpenAI fields
    id: str = Field(..., description="Unique response ID")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix creation timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Choice] = Field(..., description="Generated responses")
    usage: Usage = Field(..., description="Tokens used")

    # ACOLYTE debug fields (only if debug=true)
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Process debug information")
    rag_explanation: Optional[Dict[str, Any]] = Field(None, description="RAG process explanation")
