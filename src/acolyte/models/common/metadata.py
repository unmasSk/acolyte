"""
Metadata shared between different models.
Only essential fields, expandable as needed.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import Field
from acolyte.models.base import AcolyteBaseModel


class FileMetadata(AcolyteBaseModel):
    """Basic file metadata."""

    path: str = Field(..., description="File path")
    size_bytes: int = Field(..., ge=0, description="Size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type")
    encoding: str = Field("utf-8", description="File encoding")

    # File system timestamps
    modified_time: Optional[datetime] = Field(default=None, description="Last modification")


class GitMetadata(AcolyteBaseModel):
    """
    Complete Git metadata for advanced analysis.
    Evolved from basic to comprehensive to support Dream system and RAG.
    """

    # ✅ CURRENT BASICS (maintain compatibility)
    commit_hash: Optional[str] = Field(default=None, max_length=40, description="Commit hash")
    author: Optional[str] = Field(default=None, description="Author of last change")
    commit_message: Optional[str] = Field(
        default=None, max_length=500, description="Commit message"
    )
    commit_time: Optional[datetime] = Field(default=None, description="Commit date")

    # ✅ IMPLEMENTED BY EnrichmentService (per rag/enrichment/README.md)
    commits_last_30_days: Optional[int] = Field(
        default=None, description="Recent change frequency for Dream system"
    )
    stability_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Stability score (0=volatile, 1=stable) calculated by EnrichmentService",
    )
    contributors: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Contributor analysis with blame calculated by EnrichmentService",
    )
    total_commits: Optional[int] = Field(
        default=None, description="Total commits that touched the file"
    )
    file_age_days: Optional[int] = Field(default=None, ge=0, description="Days since file creation")
    is_actively_developed: Optional[bool] = Field(
        default=None, description="Has changes in the last 30 days"
    )

    # ✅ IMPLEMENTED IN EnrichmentService
    merge_conflicts_count: Optional[int] = Field(
        default=None,
        description="Merge conflicts resolved in this file (implemented in service.py)",
    )
    directories_restructured: Optional[int] = Field(
        default=None,
        description="Times file was moved between directories (implemented in service.py)",
    )
    code_volatility_index: Optional[float] = Field(
        default=None,
        description="% of code that changed in the last 30 days (implemented in service.py)",
    )

    # SAFE HELPER METHODS to avoid NULL errors
    def get_commits_last_30_days(self) -> int:
        """Returns recent commits or neutral default value."""
        return self.commits_last_30_days if self.commits_last_30_days is not None else 3

    def get_stability_score(self) -> float:
        """Returns stability score or neutral default value."""
        return self.stability_score if self.stability_score is not None else 0.5

    def get_file_age_days(self) -> int:
        """Returns file age or default value."""
        return self.file_age_days if self.file_age_days is not None else 30

    def get_is_actively_developed(self) -> bool:
        """Returns if actively developed or conservative default."""
        return self.is_actively_developed if self.is_actively_developed is not None else False

    def get_total_commits(self) -> int:
        """Returns total commits or safe minimum."""
        return self.total_commits if self.total_commits is not None else 1

    def get_merge_conflicts_count(self) -> int:
        """Returns merge conflicts or conservative default."""
        return self.merge_conflicts_count if self.merge_conflicts_count is not None else 0

    def get_directories_restructured(self) -> int:
        """Returns restructures or conservative default."""
        return self.directories_restructured if self.directories_restructured is not None else 0

    def get_code_volatility_index(self) -> float:
        """Returns volatility index or conservative default."""
        return self.code_volatility_index if self.code_volatility_index is not None else 0.1


class LanguageInfo(AcolyteBaseModel):
    """
    Detected language information.
    For chunking and specific analysis.
    """

    language: str = Field(..., description="Primary language")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")
    file_extension: str = Field(..., description="File extension")

    # Automatically detected frameworks (optional)
    frameworks: Optional[List[str]] = Field(
        default=None,
        description="Automatically detected frameworks/libraries (e.g. ['react', 'fastapi', 'pytest'])",
    )
