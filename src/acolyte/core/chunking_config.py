"""
Chunking strategies configuration for ACOLYTE.
"""

from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass

from acolyte.core.logging import logger


class ChunkingStrategy(Enum):
    """
    Code fragmentation strategies.

    SEMANTIC: Division based on meaning and structure
    HIERARCHICAL: Hierarchical division respecting levels
    FIXED_SIZE: Fixed size division
    ADAPTIVE: Division that adapts to content
    """

    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    FIXED_SIZE = "fixed_size"
    ADAPTIVE = "adaptive"


@dataclass
class StrategyConfig:
    """Configuration specific to a strategy."""

    chunk_size: int
    overlap: int
    min_chunk_size: int
    max_chunk_size: int
    preserve_structure: bool
    language_specific_rules: Dict[str, Any]


@dataclass
class ValidationResult:
    """Chunk validation result."""

    is_valid: bool
    score: float
    problems: List[str]


class ChunkingConfig:
    """
    Centralized configuration for chunking.

    Must handle:
    1. Strict size limits
    2. Context preservation
    3. Intelligent overlap
    4. Chunk validation
    """

    # Default configurations according to PROMPT.md
    DEFAULT_CONFIGS = {
        "python": {"size": 150, "overlap": 30},  # 20% overlap
        "java": {"size": 100, "overlap": 20},
        "javascript": {"size": 120, "overlap": 24},
        "typescript": {"size": 120, "overlap": 24},
        "go": {"size": 100, "overlap": 20},
        "rust": {"size": 100, "overlap": 20},
        "markdown": {"size": 50, "overlap": 10},
        "default": {"size": 100, "overlap": 20},
    }

    @staticmethod
    def get_strategy_config(strategy: ChunkingStrategy, language: str) -> StrategyConfig:
        """
        Gets specific configuration for strategy and language.

        Language configurations:
        - Python: Respect indentation and blocks
        - JavaScript: Consider closures and callbacks
        - Java: Keep complete classes if possible
        - SQL: Separate by statements

        Returns:
            StrategyConfig with optimized parameters
        """
        base_config = ChunkingConfig.DEFAULT_CONFIGS.get(
            language.lower(), ChunkingConfig.DEFAULT_CONFIGS["default"]
        )

        language_rules = {}

        if language.lower() == "python":
            language_rules = {
                "respect_indentation": True,
                "preserve_docstrings": True,
                "ast_aware": True,
                "break_points": ["def", "class", "async def"],
            }
        elif language.lower() in ["javascript", "typescript"]:
            language_rules = {
                "preserve_closures": True,
                "respect_blocks": True,
                "ast_aware": True,
                "break_points": ["function", "class", "const", "let"],
            }
        elif language.lower() == "java":
            language_rules = {
                "preserve_classes": True,
                "respect_methods": True,
                "ast_aware": True,
                "break_points": ["class", "interface", "public", "private"],
            }
        elif language.lower() == "sql":
            language_rules = {
                "split_by_statement": True,
                "preserve_transactions": True,
                "break_points": ["CREATE", "SELECT", "INSERT", "UPDATE"],
            }

        return StrategyConfig(
            chunk_size=base_config["size"],
            overlap=base_config["overlap"],
            min_chunk_size=30,
            max_chunk_size=base_config["size"] * 2,
            preserve_structure=strategy != ChunkingStrategy.FIXED_SIZE,
            language_specific_rules=language_rules,
        )

    @staticmethod
    def validate_chunk(chunk: str, min_size: int, max_size: int, language: str) -> ValidationResult:
        """
        Validates that a chunk meets quality criteria.

        Criteria:
        1. Size within limits
        2. Basically valid syntax
        3. Doesn't cut in the middle of strings/comments
        4. Preserves minimum context

        Returns:
            ValidationResult with score and problems
        """
        problems: List[str] = []
        score = 1.0

        # Validate size
        chunk_lines = chunk.strip().split("\n")
        if len(chunk_lines) < min_size:
            problems.append(f"Chunk too small: {len(chunk_lines)} lines")
            score *= 0.5
        elif len(chunk_lines) > max_size:
            problems.append(f"Chunk too large: {len(chunk_lines)} lines")
            score *= 0.7

        # Validate basic syntax
        if language.lower() == "python":
            # Check consistent indentation
            if chunk.strip() and not chunk[0].isspace() and "\n " in chunk:
                # First line not indented but there are indented lines
                score *= 0.9

            # Check for cut strings
            if chunk.count('"""') % 2 != 0:
                problems.append("Possible cut docstring")
                score *= 0.6

        # Check that it doesn't cut in the middle of multiline comments
        if language.lower() in ["java", "javascript", "typescript", "c", "cpp"]:
            if chunk.count("/*") != chunk.count("*/"):
                problems.append("Cut multiline comment")
                score *= 0.6

        # Check minimum context
        if len(chunk.strip()) < 10:  # Less than 10 characters
            problems.append("Chunk without significant content")
            score *= 0.3

        is_valid = score > 0.5 and len(problems) == 0

        # Log validation failures for debugging
        if not is_valid and problems:
            logger.warning(
                "Chunk validation failed",
                language=language,
                score=score,
                problems=problems,
                chunk_preview=chunk[:50] + "..." if len(chunk) > 50 else chunk,
            )

        return ValidationResult(is_valid=is_valid, score=score, problems=problems)
