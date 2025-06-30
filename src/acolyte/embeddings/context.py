"""
Types of context for embeddings.

Defines the RichCodeContext structure that enriches embeddings
with additional information like imports, dependencies, and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, TypedDict


class RichCodeContextDict(TypedDict):
    """TypedDict for serializing RichCodeContext."""

    language: str
    file_path: str
    imports: List[str]
    dependencies: List[str]
    called_by: List[str]
    calls: List[str]
    last_modified: Optional[str]
    test_coverage: Optional[float]
    complexity: Optional[int]
    semantic_tags: List[str]


@dataclass
class RichCodeContext:
    """Rich context to improve embedding quality.

    Additional information that significantly improves the relevance
    of the generated embeddings.
    """

    # Basic information
    language: str  # "python", "javascript", etc.
    file_path: str  # "src/auth/login.py"

    # Advanced context for better relevance
    imports: List[str] = field(default_factory=list)  # ["django.auth", "jwt", "bcrypt"]
    dependencies: List[str] = field(default_factory=list)  # ["User", "Token", "validate_password"]
    called_by: List[str] = field(default_factory=list)  # ["api_login", "cli_authenticate"]
    calls: List[str] = field(default_factory=list)  # ["hash_password", "create_session"]
    last_modified: Optional[datetime] = None  # When it was modified
    test_coverage: Optional[float] = None  # 0.85 (85% covered)
    complexity: Optional[int] = None  # Cyclomatic complexity
    semantic_tags: List[str] = field(
        default_factory=list
    )  # ["authentication", "security", "user-management"]

    def __post_init__(self):
        """Validates types and required fields.

        Raises:
            TypeError: If the required fields are not of the correct type
            ValueError: If the values are not valid
        """
        # Validate required fields
        if not isinstance(self.language, str):
            raise TypeError(f"language must be str, got {type(self.language).__name__}")
        if not isinstance(self.file_path, str):
            raise TypeError(f"file_path must be str, got {type(self.file_path).__name__}")

        # Validate that they are not empty
        if not self.language.strip():
            raise ValueError("language cannot be empty")
        if not self.file_path.strip():
            raise ValueError("file_path cannot be empty")

        # Validate numeric optional fields
        if self.test_coverage is not None:
            if not isinstance(self.test_coverage, (int, float)):
                raise TypeError(
                    f"test_coverage must be float, got {type(self.test_coverage).__name__}"
                )
            if not 0.0 <= self.test_coverage <= 1.0:
                raise ValueError(
                    f"test_coverage must be between 0.0 and 1.0, got {self.test_coverage}"
                )

        if self.complexity is not None:
            if not isinstance(self.complexity, int):
                raise TypeError(f"complexity must be int, got {type(self.complexity).__name__}")
            if self.complexity < 0:
                raise ValueError(f"complexity must be >= 0, got {self.complexity}")

        # Validate datetime if it exists
        if self.last_modified is not None:
            if not isinstance(self.last_modified, datetime):
                raise TypeError(
                    f"last_modified must be datetime, got {type(self.last_modified).__name__}"
                )

    def to_dict(self) -> RichCodeContextDict:
        """Converts the context to a dictionary for serialization.

        Returns:
            RichCodeContextDict with all context fields
        """
        return {
            "language": self.language,
            "file_path": self.file_path,
            "imports": self.imports,
            "dependencies": self.dependencies,
            "called_by": self.called_by,
            "calls": self.calls,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "test_coverage": self.test_coverage,
            "complexity": self.complexity,
            "semantic_tags": self.semantic_tags,
        }
