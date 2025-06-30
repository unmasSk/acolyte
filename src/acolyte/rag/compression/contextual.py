"""
Contextual query analyzer for intelligent compression decisions.

This module analyzes queries to determine if and how to compress chunks,
without using LLMs. Part of Decision #14: Contextual compression.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Set

from acolyte.core.logging import logger

from acolyte.core.secure_config import Settings
from acolyte.models.chunk import Chunk, ChunkMetadata  # CORRECT: chunk without 's'


@dataclass
class QueryContext:
    """Context extracted from query analysis."""

    query: str
    query_type: str  # 'specific', 'general', 'generation'
    query_tokens: Set[str]
    entities: Dict[str, List[str]]  # type -> [entities]
    compression_needed: bool
    suggested_ratio: float

    @property
    def is_specific(self) -> bool:
        return self.query_type == "specific"

    @property
    def is_general(self) -> bool:
        return self.query_type == "general"

    @property
    def is_generation(self) -> bool:
        return self.query_type == "generation"


class QueryAnalyzer:
    """
    Analyzes queries to extract context for compression decisions.

    Key responsibilities:
    - Detect query type (specific/general/generation)
    - Extract entities (files, functions, classes, etc.)
    - Determine if compression is beneficial
    - Calculate relevance scores
    """

    def __init__(self, max_specific_length: int = 5, cache_size: int = 1000):
        """
        Initialize analyzer with configuration.

        Args:
            max_specific_length: Max words for a query to be "specific"
            cache_size: LRU cache size for query analysis
        """
        self.max_specific_length = max_specific_length

        # Pre-compiled patterns for performance
        self.patterns = self._compile_patterns()

        # Load keyword sets from configuration
        config = Settings()
        compression_config = config.get("rag.compression.contextual", {})

        # General keywords (queries that should NOT be compressed)
        self.general_keywords = set(
            compression_config.get(
                "broad_query_keywords",
                ["arquitectura", "completo", "general", "overview", "estructura"],
            )
        )

        # Specific keywords (queries that SHOULD be compressed)
        self.specific_indicators = set(
            compression_config.get(
                "specific_query_keywords",
                ["error", "bug", "función", "método", "variable", "línea"],
            )
        )

        # Generation keywords - these can be hardcoded since
        # they are part of the system's fundamental behavior
        self.generation_keywords = {
            # Spanish - infinitive and imperative forms
            "crear",
            "crea",
            "generar",
            "genera",
            "escribir",
            "escribe",
            "implementar",
            "implementa",
            "desarrollar",
            "desarrolla",
            "desarróllame",
            "construir",
            "construye",
            "producir",
            "produce",
            "hacer",
            "haz",
            "hazme",
            "créame",
            # English
            "create",
            "generate",
            "write",
            "implement",
            "develop",
            "build",
            "produce",
            "make",
        }

        # Enable LRU cache
        self._analyze_query_cached = lru_cache(maxsize=cache_size)(self._analyze_query_impl)

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for entity extraction."""
        return {
            "files": [
                # Full path references (must come first)
                re.compile(r"\b([\w/]+/\w+\.\w+)\b"),  # src/modules/auth.py
                # Direct filename with extension
                re.compile(r"\b(\w+\.\w+)\b"),
                # File references in different languages
                re.compile(r"\b(?:archivo|file|en|in)\s+(\w+\.\w+)", re.IGNORECASE),
            ],
            "functions": [
                # Function calls with parentheses
                re.compile(r"\b(\w+)\s*\(\s*\)"),
                # Function/method keywords - capture full name including underscores
                re.compile(r"\b(?:función|function|método|method|def)\s+(\w+)", re.IGNORECASE),
                # Async functions
                re.compile(r"\b(?:async\s+)?(\w+)\s*\(", re.IGNORECASE),
                # Function name without parentheses (for queries like "calculate_total function")
                re.compile(r"\b(\w+)\s+(?:function|método|función)\b", re.IGNORECASE),
            ],
            "classes": [
                # PascalCase names
                re.compile(r"\b([A-Z][a-zA-Z0-9]+)(?:\s|$)"),
                # Class keywords
                re.compile(r"\b(?:clase|class)\s+(\w+)", re.IGNORECASE),
            ],
            "lines": [
                # Line ranges MUST come before single lines to match first
                re.compile(r"\b(?:líneas|lines)\s+(\d+\s*[-–]\s*\d+)", re.IGNORECASE),
                # Single line numbers
                re.compile(r"\b(?:línea|line)\s+(\d+)", re.IGNORECASE),
                re.compile(r"\b(?:l|L)\.?\s*(\d+)"),  # L.23, l23
                re.compile(r":(\d+)(?:\s|$)"),  # file.py:23
            ],
            "errors": [
                # Error messages in quotes
                re.compile(r'error:?\s*["\']([^"\']+)["\']', re.IGNORECASE),
                # Exception types
                re.compile(r"\b(\w+Error|\w+Exception)\b"),
                # Common error keywords
                re.compile(r"\b(KeyError|ValueError|TypeError|NameError|AttributeError)\b"),
            ],
            "variables": [
                # snake_case variables
                re.compile(r"\b([a-z][a-z0-9_]+)\b"),
                # Variable assignment
                re.compile(r"\b(\w+)\s*=\s*"),
                # Variable keywords
                re.compile(r"\b(?:variable|var)\s+(\w+)", re.IGNORECASE),
            ],
        }

    def analyze_query(self, query: str) -> QueryContext:
        """
        Analyze query and extract context.

        Uses LRU cache for repeated queries.
        """
        return self._analyze_query_cached(query)

    def _analyze_query_impl(self, query: str) -> QueryContext:
        """Actual implementation of query analysis."""
        # Tokenize query
        query_lower = query.lower()
        query_tokens = set(re.findall(r"\w+", query_lower))

        # Extract entities
        entities = self._extract_entities(query)

        # Determine query type
        query_type = self._determine_query_type(query, query_lower, query_tokens, entities)

        # Determine if compression is needed
        compression_needed = query_type == "specific"

        # Suggest compression ratio based on type
        if query_type == "generation":
            suggested_ratio = 1.0  # No compression
        elif query_type == "general":
            suggested_ratio = 0.9  # Minimal compression
        elif len(entities) > 3:  # Very specific
            suggested_ratio = 0.3  # Aggressive compression
        else:
            suggested_ratio = 0.6  # Moderate compression

        return QueryContext(
            query=query,
            query_type=query_type,
            query_tokens=query_tokens,
            entities=entities,
            compression_needed=compression_needed,
            suggested_ratio=suggested_ratio,
        )

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract all entities from query."""
        entities = {}

        for entity_type, patterns in self.patterns.items():
            found = []
            for pattern in patterns:
                matches = pattern.findall(query)
                found.extend([m if isinstance(m, str) else m[0] for m in matches])

            # Deduplicate while preserving order
            seen = set()
            unique = []
            for item in found:
                if item.lower() not in seen:
                    seen.add(item.lower())
                    unique.append(item)

            if unique:
                entities[entity_type] = unique

        return entities

    def _determine_query_type(
        self, query: str, query_lower: str, query_tokens: Set[str], entities: Dict[str, List[str]]
    ) -> str:
        """Determine if query is specific, general, or generation."""
        # Check for generation first (highest priority)
        if self._is_generation_query(query_lower, query_tokens):
            return "generation"

        # Check for general queries
        if self._is_general_query(query_lower, query_tokens):
            return "general"

        # Check for specific indicators
        if self._is_specific_query(query, query_tokens, entities):
            return "specific"

        # Default based on length
        word_count = len(query.split())
        if word_count <= self.max_specific_length:
            return "specific"
        else:
            return "general"

    def _is_generation_query(self, query_lower: str, tokens: Set[str]) -> bool:
        """Check if query is asking to generate code."""
        # Check each word individually for generation keywords
        words = query_lower.split()
        for word in words:
            if word in self.generation_keywords:
                return True

        return False

    def _is_general_query(self, query_lower: str, tokens: Set[str]) -> bool:
        """Check if query is general/broad."""
        # Direct keyword match - look for any general keyword in query
        for keyword in self.general_keywords:
            if keyword in query_lower:
                return True

        # Additional general indicators
        general_patterns = {
            "explain the",
            "what is the",
            "describe the",
            "explicar",
            "cuál es",
            "describir",
            "give me a",
            "dame un",
            "explain how",
            "how does",
            "how the",
            "cómo funciona",
            "cómo el",
        }
        if any(pattern in query_lower for pattern in general_patterns):
            return True

        # Count total words, not unique tokens
        word_count = len(query_lower.split())

        # Very long queries are likely general explanations
        if word_count > 50:
            return True

        # Long queries without specific references
        if len(tokens) > 15 and not any(tokens & set(self.specific_indicators)):
            return True

        return False

    def _is_specific_query(
        self, query: str, tokens: Set[str], entities: Dict[str, List[str]]
    ) -> bool:
        """Check if query is specific."""
        # Has specific entities
        if any(entities.values()):
            logger.info("[UNTESTED PATH] Query classified as specific due to entities")
            return True

        # Has specific indicators
        if tokens & set(self.specific_indicators):
            return True

        # Very short query
        if len(query.split()) <= 3:
            return True

        return False

    def clear_cache(self) -> None:
        """Clear the LRU cache when configuration changes."""
        self._analyze_query_cached.cache_clear()
        logger.debug("Query analysis cache cleared")
        logger.info("[UNTESTED PATH] Query analysis cache cleared")

    def calculate_relevance(self, chunk: Chunk, context: QueryContext) -> float:
        """
        Calculate relevance score for a chunk.

        Scoring components:
        - File match: 0.4
        - Function/class match: 0.3
        - Line proximity: 0.0-0.4
        - Token overlap: 0.0-0.3
        - Entity matches: 0.1 per match

        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        chunk_lower = chunk.content.lower()

        # 1. File match (0.4 points)
        if chunk.metadata and chunk.metadata.file_path:
            file_name = chunk.metadata.file_path.split("/")[-1]
            for file_entity in context.entities.get("files", []):
                if file_entity.lower() in file_name.lower():
                    score += 0.4
                    break

        # 2. Function/class match (0.3 points)
        function_matched = False
        for func in context.entities.get("functions", []):
            # Check in content
            if func.lower() in chunk_lower:
                score += 0.3
                function_matched = True
                logger.info("[UNTESTED PATH] Function matched in chunk content")
                break
            # Also check in metadata name
            if (
                chunk.metadata
                and chunk.metadata.name
                and func.lower() == chunk.metadata.name.lower()
            ):
                score += 0.3
                function_matched = True
                logger.info("[UNTESTED PATH] Function matched in metadata name")
                break

        # Only check classes if no function match (to avoid double scoring)
        if not function_matched:
            for cls in context.entities.get("classes", []):
                if cls in chunk.content:  # Case sensitive for classes
                    score += 0.3
                    break

        # 3. Line proximity (0.0-0.4 points)
        if context.entities.get("lines") and chunk.metadata:
            score += self._calculate_line_proximity_score(context.entities["lines"], chunk.metadata)

        # 4. Token overlap (0.0-0.2 points) - Reduced to avoid inflating scores
        if context.query_tokens:
            chunk_tokens = set(re.findall(r"\w+", chunk_lower))
            # Remove common stop words
            stop_words = {
                "el",
                "la",
                "de",
                "en",
                "y",
                "a",
                "que",
                "es",
                "the",
                "of",
                "in",
                "and",
                "to",
                "is",
                "for",
            }
            relevant_query_tokens = context.query_tokens - stop_words
            relevant_chunk_tokens = chunk_tokens - stop_words

            if relevant_query_tokens and relevant_chunk_tokens:
                overlap = len(relevant_query_tokens & relevant_chunk_tokens)
                overlap_ratio = overlap / len(relevant_query_tokens)
                # Cap at 0.15 to avoid non-matching chunks getting high scores
                score += min(overlap_ratio * 0.15, 0.15)
                logger.info("[UNTESTED PATH] Token overlap scoring applied")

        # 5. Entity matches (0.1 per match, max 0.3)
        entity_matches = 0
        for entity_type, entities in context.entities.items():
            if entity_type in ["files", "functions", "classes"]:
                continue  # Already counted

            for entity in entities:
                if entity.lower() in chunk_lower:
                    entity_matches += 1
                    logger.info("[UNTESTED PATH] Entity match found in chunk")

        score += min(entity_matches * 0.1, 0.3)
        if entity_matches > 0:
            logger.info("[UNTESTED PATH] Entity matches added to score")

        return min(score, 1.0)

    def _calculate_line_proximity_score(
        self, line_entities: List[str], metadata: ChunkMetadata
    ) -> float:
        """Calculate score based on line number proximity."""
        if not metadata.start_line:
            logger.info("[UNTESTED PATH] No start_line in metadata for line proximity")
            return 0.0

        chunk_start = metadata.start_line
        chunk_end = metadata.end_line or chunk_start

        best_score = 0.0
        for line_str in line_entities:
            try:
                # Handle ranges like "20-30"
                if "-" in line_str:
                    start, end = map(int, line_str.split("-"))
                    target_lines = range(start, end + 1)
                else:
                    target_lines = [int(line_str)]

                for target in target_lines:
                    if chunk_start <= target <= chunk_end:
                        # Line is within chunk
                        best_score = max(best_score, 0.4)
                        logger.info("[UNTESTED PATH] Line within chunk boundaries")
                    else:
                        # Calculate proximity
                        distance = min(abs(target - chunk_start), abs(target - chunk_end))
                        if distance <= 5:
                            best_score = max(best_score, 0.3)
                            logger.info("[UNTESTED PATH] Line within 5 lines of chunk")
                        elif distance <= 10:
                            best_score = max(best_score, 0.2)
                            logger.info("[UNTESTED PATH] Line within 10 lines of chunk")
                        elif distance <= 20:
                            best_score = max(best_score, 0.1)
                            logger.info("[UNTESTED PATH] Line within 20 lines of chunk")

            except ValueError:
                logger.info("[UNTESTED PATH] ValueError parsing line number")
                continue

        return best_score
