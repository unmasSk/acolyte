from enum import Enum
from typing import Optional, List
from typing_extensions import Self
from pydantic import Field, field_validator, model_validator
from acolyte.models.base import AcolyteBaseModel, TimestampMixin, IdentifiableMixin

class DocumentType(str, Enum):
    CODE = "code"
    MARKDOWN = "markdown"
    CONFIG = "config"
    DATA = "data"
    OTHER = "other"

class Document(AcolyteBaseModel, TimestampMixin, IdentifiableMixin):
    file_path: str = Field(...)
    document_type: DocumentType = Field(...)
    content: str = Field(...)
    size_bytes: int = Field(...)
    language: Optional[str] = Field(default=None)
    encoding: str = Field(default="utf-8")
    indexed: bool = Field(default=False)
    chunks_count: int = Field(default=0)

    @field_validator("file_path")
    @classmethod
    def validate_path_safety(cls, v: str) -> str: ...
    def mark_indexed(self, chunks_count: int) -> None: ...

class IndexingBatch(AcolyteBaseModel):
    documents: List[Document] = Field(..., min_length=1, max_length=100)
    force_reindex: bool = Field(default=False)
    total_size_bytes: int = Field(default=0)

    @model_validator(mode="after")
    def validate_batch_size(self) -> Self: ...
    def _get_batch_size_limit(self) -> int: ...

class IndexingProgress(AcolyteBaseModel):
    task_id: str = Field(...)
    status: str = Field(...)
    progress_percent: float = Field(...)
    files_processed: int = Field(default=0)
    files_total: int = Field(default=0)
    chunks_created: int = Field(default=0)
    current_file: Optional[str] = Field(default=None)
    elapsed_seconds: float = Field(default=0.0)
    estimated_remaining: float = Field(default=0.0)
