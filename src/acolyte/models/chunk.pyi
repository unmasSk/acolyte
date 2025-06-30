from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import Field
from acolyte.models.base import AcolyteBaseModel, TimestampMixin, IdentifiableMixin

class ChunkType(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    PROPERTY = "property"
    CLASS = "class"
    INTERFACE = "interface"
    MODULE = "module"
    NAMESPACE = "namespace"
    COMMENT = "comment"
    DOCSTRING = "docstring"
    README = "readme"
    IMPORTS = "imports"
    CONSTANTS = "constants"
    TYPES = "types"
    TESTS = "tests"
    SUMMARY = "summary"
    SUPER_SUMMARY = "super_summary"
    UNKNOWN = "unknown"

class ChunkMetadata(AcolyteBaseModel):
    file_path: str = Field(...)
    language: str = Field(...)
    start_line: int = Field(...)
    end_line: int = Field(...)
    chunk_type: ChunkType = Field(...)
    name: Optional[str] = Field(...)
    last_modified: Optional[datetime] = Field(...)
    language_specific: Optional[Dict[str, Any]] = Field(...)

    @property
    def line_count(self) -> int: ...

class Chunk(AcolyteBaseModel, TimestampMixin, IdentifiableMixin):
    content: str = Field(...)
    metadata: ChunkMetadata = Field(...)
    summary: Optional[str] = Field(...)

    def to_search_text(self, rich_context: Optional[Any] = None) -> str: ...
