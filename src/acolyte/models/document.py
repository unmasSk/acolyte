"""
Document and indexing models.
Represents complete files and their indexing process.
"""

from enum import Enum
from typing import Optional, List
from typing_extensions import Self
from pydantic import Field, field_validator, model_validator
from pathlib import Path
from acolyte.models.base import AcolyteBaseModel, TimestampMixin, IdentifiableMixin
from acolyte.core.secure_config import Settings
import yaml


class DocumentType(str, Enum):
    """Supported document types."""

    CODE = "code"  # Code files
    MARKDOWN = "markdown"  # Documentation
    CONFIG = "config"  # Configuration
    DATA = "data"  # JSON, CSV, etc
    OTHER = "other"  # Other types


class Document(AcolyteBaseModel, TimestampMixin, IdentifiableMixin):
    """
    Complete document for indexing.
    Represents a file in the project.
    """

    # Identification
    file_path: str = Field(..., description="Relative file path")
    document_type: DocumentType = Field(..., description="Document type")

    # Content
    content: str = Field(..., description="File content")
    size_bytes: int = Field(..., ge=0, description="Size in bytes")

    # Metadata
    language: Optional[str] = Field(None, description="Detected language")
    encoding: str = Field("utf-8", description="File encoding")

    # Indexing state
    indexed: bool = Field(False, description="Whether indexed")
    chunks_count: int = Field(0, description="Number of generated chunks")

    @field_validator("file_path")
    @classmethod
    def validate_path_safety(cls, v: str) -> str:
        """Simple validation for local mono-user system.

        Only prevents common accidental errors:
        - Empty paths
        - Absolute paths
        - Path traversal (..)
        """
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")

        path = Path(v.strip())

        # Only essential validations for local system
        if path.is_absolute():
            raise ValueError("Path must be relative to project")

        if ".." in path.parts:
            raise ValueError("Path cannot exit project directory")

        return str(path)

    def mark_indexed(self, chunks_count: int) -> None:
        """Marks document as indexed."""
        self.indexed = True
        self.chunks_count = chunks_count
        self.touch()


class IndexingBatch(AcolyteBaseModel):
    """
    Batch of files to index.
    Groups multiple files for efficient processing.
    """

    documents: List[Document] = Field(
        ..., min_length=1, max_length=100, description="Documents to index"
    )

    # Options
    force_reindex: bool = Field(False, description="Reindex even if already indexed")

    # Statistics
    total_size_bytes: int = Field(0, description="Total batch size")

    @model_validator(mode="after")
    def validate_batch_size(self) -> Self:
        """Validates total batch size according to .acolyte configuration."""
        max_size_mb = self._get_batch_size_limit()
        max_size_bytes = max_size_mb * 1024 * 1024

        total = sum(doc.size_bytes for doc in self.documents)
        self.total_size_bytes = total  # Update statistics

        if total > max_size_bytes:
            raise ValueError(f"Batch exceeds {max_size_mb}MB (configurable limit in .acolyte)")

        return self

    def _get_batch_size_limit(self) -> int:
        """Gets batch limit from .acolyte with fallback to Settings."""
        # Try to read from .acolyte first (source of truth)
        acolyte_path = Path(".acolyte")
        if acolyte_path.exists():
            try:
                with open(acolyte_path, "r", encoding="utf-8") as f:
                    acolyte_config = yaml.safe_load(f)
                    if acolyte_config and "indexing" in acolyte_config:
                        return acolyte_config["indexing"].get("batch_max_size_mb", 50)
            except (yaml.YAMLError, KeyError, TypeError):
                pass  # Fallback to Settings if error

        # Fallback to Settings (can read config.yaml)
        config = Settings()
        return config.get("batch_max_size_mb", 50)  # Default 50MB


class IndexingProgress(AcolyteBaseModel):
    """
    Indexing progress for WebSocket.
    Sent during initial project indexing.
    """

    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Current status")
    progress_percent: float = Field(..., ge=0.0, le=100.0)

    # Statistics
    files_processed: int = Field(0, description="Processed files")
    files_total: int = Field(0, description="Total files")
    chunks_created: int = Field(0, description="Created chunks")

    # Current file
    current_file: Optional[str] = Field(None, description="File being processed")

    # Time
    elapsed_seconds: float = Field(0.0, description="Elapsed time")
    estimated_remaining: float = Field(0.0, description="Estimated remaining time")
