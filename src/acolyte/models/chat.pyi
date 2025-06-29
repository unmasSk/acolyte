from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import Field, field_validator
from acolyte.models.base import AcolyteBaseModel

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(AcolyteBaseModel):
    role: Role = Field(...)
    content: str = Field(...)
    metadata: Optional[Dict[str, Any]] = Field(None)

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str: ...

class ChatRequest(AcolyteBaseModel):
    model: str = Field(...)
    messages: List[Message] = Field(...)
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(None)
    stream: bool = Field(default=False)
    debug: bool = Field(default=False)
    explain_rag: bool = Field(default=False)

    @field_validator("messages")
    @classmethod
    def validate_message_flow(cls, messages: List[Message]) -> List[Message]: ...

class Choice(AcolyteBaseModel):
    message: Message
    finish_reason: str = Field(default="stop")
    index: int = Field(default=0)

class Usage(AcolyteBaseModel):
    prompt_tokens: int = Field(...)
    completion_tokens: int = Field(...)
    total_tokens: int = Field(...)

class ChatResponse(AcolyteBaseModel):
    id: str = Field(...)
    object: str = Field(default="chat.completion")
    created: int = Field(...)
    model: str = Field(...)
    choices: List[Choice] = Field(...)
    usage: Usage = Field(...)
    debug_info: Optional[Dict[str, Any]] = Field(None)
    rag_explanation: Optional[Dict[str, Any]] = Field(None)
