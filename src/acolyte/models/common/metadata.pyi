from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import Field
from acolyte.models.base import AcolyteBaseModel

class FileMetadata(AcolyteBaseModel):
    path: str = Field(...)
    size_bytes: int = Field(..., ge=0)
    mime_type: Optional[str] = Field(None)
    encoding: str = Field(default="utf-8")
    modified_time: Optional[datetime] = Field(default=None)

class GitMetadata(AcolyteBaseModel):
    commit_hash: Optional[str] = Field(default=None, max_length=40)
    author: Optional[str] = Field(default=None)
    commit_message: Optional[str] = Field(default=None, max_length=500)
    commit_time: Optional[datetime] = Field(default=None)
    commits_last_30_days: Optional[int] = Field(default=None)
    stability_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    contributors: Optional[Dict[str, Dict[str, Any]]] = Field(default=None)
    total_commits: Optional[int] = Field(default=None)
    file_age_days: Optional[int] = Field(default=None)
    is_actively_developed: Optional[bool] = Field(default=None)
    merge_conflicts_count: Optional[int] = Field(default=None)
    directories_restructured: Optional[int] = Field(default=None)
    code_volatility_index: Optional[float] = Field(default=None)

    def get_commits_last_30_days(self) -> int: ...
    def get_stability_score(self) -> float: ...
    def get_file_age_days(self) -> int: ...
    def get_is_actively_developed(self) -> bool: ...
    def get_total_commits(self) -> int: ...
    def get_merge_conflicts_count(self) -> int: ...
    def get_directories_restructured(self) -> int: ...
    def get_code_volatility_index(self) -> float: ...

class LanguageInfo(AcolyteBaseModel):
    language: str = Field(...)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    file_extension: str = Field(...)
    frameworks: Optional[List[str]] = Field(default=None)
