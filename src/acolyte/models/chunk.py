"""
Code chunking models.
Defines how code is fragmented and stored for RAG.
"""

from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import Field
from acolyte.models.base import AcolyteBaseModel, TimestampMixin, IdentifiableMixin


class ChunkType(str, Enum):
    """
    Code chunk types.
    18 types for maximum precision in RAG searches.
    """

    # Functional
    FUNCTION = "function"  # Individual functions and methods
    METHOD = "method"  # Class methods specifically
    CONSTRUCTOR = "constructor"  # Constructors/initializers
    PROPERTY = "property"  # Properties/getters/setters

    # Structural
    CLASS = "class"  # Classes
    INTERFACE = "interface"  # Interfaces/protocols
    MODULE = "module"  # Complete module/file
    NAMESPACE = "namespace"  # Namespaces/packages

    # Documentary
    COMMENT = "comment"  # Important comments
    DOCSTRING = "docstring"  # Docstrings/JSDoc
    README = "readme"  # README files

    # Semantic
    IMPORTS = "imports"  # Imports/requires
    CONSTANTS = "constants"  # Constants/configuration
    TYPES = "types"  # Type definitions
    TESTS = "tests"  # Tests/specs

    # Hierarchical (for summaries)
    SUMMARY = "summary"  # Summary of larger chunk
    SUPER_SUMMARY = "super_summary"  # Summary of multiple summaries

    UNKNOWN = "unknown"  # Fallback


class ChunkMetadata(AcolyteBaseModel):
    """
    Essential metadata for a chunk.
    Only necessary fields to start.
    """

    # Location
    file_path: str = Field(..., description="File path")
    language: str = Field(..., description="Programming language")
    start_line: int = Field(..., ge=1, description="Start line")
    end_line: int = Field(..., ge=1, description="End line")

    # Type and context
    chunk_type: ChunkType = Field(..., description="Chunk type")
    name: Optional[str] = Field(None, description="Element name (function, class, etc)")

    # Basic Git (optional for now)
    last_modified: Optional[datetime] = Field(None, description="Last modification timestamp")

    # Language-specific metadata
    language_specific: Optional[Dict[str, Any]] = Field(
        None, description="Language-specific metadata (complexity, patterns, etc)"
    )

    @property
    def line_count(self) -> int:
        """Number of lines in the chunk."""
        return self.end_line - self.start_line + 1


class Chunk(AcolyteBaseModel, TimestampMixin, IdentifiableMixin):
    """
    Indexed code fragment.
    Basic unit of the RAG system.
    """

    # Content
    content: str = Field(..., min_length=1, description="Chunk code")

    # Metadata
    metadata: ChunkMetadata = Field(..., description="Chunk information")

    # For search
    # NOTE: Embeddings go directly to Weaviate using EmbeddingVector,
    # not stored in the model to comply with ACOLYTE architecture
    summary: Optional[str] = Field(None, max_length=500, description="Chunk summary for search")

    def to_search_text(self, rich_context=None) -> str:
        """
        Generates optimized text for embeddings and search.
        STANDARD INTERFACE used by the entire system to prepare chunks.

        Args:
            rich_context: Optional RichCodeContext for enrichment

        Returns:
            Formatted text for embeddings with rich context
        """
        parts = []

        # Prioritize RichCodeContext if available
        if rich_context:
            parts.append(f"<{rich_context.language}>")

            if hasattr(rich_context, "imports") and rich_context.imports:
                parts.append(f"imports: {', '.join(rich_context.imports[:5])}")

            if hasattr(rich_context, "semantic_tags") and rich_context.semantic_tags:
                parts.append(f"tags: {', '.join(rich_context.semantic_tags[:3])}")

            if hasattr(rich_context, "dependencies") and rich_context.dependencies:
                parts.append(f"uses: {', '.join(rich_context.dependencies[:3])}")

        # Fallback to basic metadata if no rich context
        elif self.metadata.name:
            parts.append(f"Name: {self.metadata.name}")

        if self.summary:
            parts.append(f"Summary: {self.summary}")

        # Content always goes at the end
        parts.append(self.content)

        # Use spaces for consistency with embeddings (not \n\n)
        return " ".join(parts)
