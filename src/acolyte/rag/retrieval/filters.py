"""
Search filters for hybrid search results.

Provides basic filtering capabilities for search results by metadata.
Simple, efficient filters for mono-user local usage.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import re

from acolyte.core.logging import logger
from acolyte.core.utils.file_types import FileTypeDetector
from acolyte.models.chunk import Chunk, ChunkType
from acolyte.core.utils.datetime_utils import parse_iso_datetime


class SearchFilters:
    """
    Basic filters for search results.

    Filters can be applied pre-search (in Weaviate) or post-search (in Python).
    This implementation focuses on simplicity over performance optimization.
    """

    def __init__(self):
        """Initialize search filters."""
        self.applied_filters = []

    def filter_by_type(self, chunks: List[Chunk], file_types: List[str]) -> List[Chunk]:
        """
        Filter chunks by file type extension.

        Args:
            chunks: List of chunks to filter
            file_types: List of extensions (e.g., ['.py', '.js'])

        Returns:
            Filtered list of chunks
        """
        if not file_types:
            return chunks

        # Normalize extensions
        normalized_types = [ext if ext.startswith('.') else f'.{ext}' for ext in file_types]

        filtered = [
            chunk
            for chunk in chunks
            if any(chunk.metadata.file_path.endswith(ext) for ext in normalized_types)
        ]

        logger.debug("Filter by type", types=file_types, before=len(chunks), after=len(filtered))

        return filtered

    def filter_by_date(
        self, chunks: List[Chunk], start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> List[Chunk]:
        """
        Filter chunks by modification date.

        Args:
            chunks: List of chunks to filter
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            Filtered list of chunks
        """
        if not start and not end:
            return chunks

        filtered = []
        for chunk in chunks:
            # Check if chunk metadata has last_modified timestamp
            if chunk.metadata.last_modified:
                # Use centralized system to parse if it's a string
                if isinstance(chunk.metadata.last_modified, str):
                    try:
                        chunk_date = parse_iso_datetime(chunk.metadata.last_modified)
                    except Exception:
                        # Skip this chunk if date parsing fails
                        logger.info("[UNTESTED PATH] Date parsing failed")
                        continue
                    if chunk_date is None:
                        # Skip this chunk if parsing returns None
                        logger.info("[UNTESTED PATH] parse_iso_datetime returned None")
                        continue
                elif isinstance(chunk.metadata.last_modified, datetime):
                    chunk_date = chunk.metadata.last_modified
                else:
                    # Skip this chunk if we can't parse the date
                    logger.info("[UNTESTED PATH] Unknown date type for last_modified")
                    continue

                if start and chunk_date < start:
                    continue
                if end and chunk_date > end:
                    continue

                filtered.append(chunk)
            else:
                # If no date info, include by default
                filtered.append(chunk)

        logger.debug(
            "Filter by date", start=start, end=end, before=len(chunks), after=len(filtered)
        )

        return filtered

    def filter_by_language(self, chunks: List[Chunk], languages: List[str]) -> List[Chunk]:
        """
        Filter chunks by programming language.

        Args:
            chunks: List of chunks to filter
            languages: List of languages (e.g., ['python', 'javascript'])

        Returns:
            Filtered list of chunks
        """
        if not languages:
            return chunks

        # Normalize language names
        normalized_langs = [lang.lower() for lang in languages]

        filtered = []
        for chunk in chunks:
            # Get language from FileTypeDetector
            path = Path(chunk.metadata.file_path)
            lang = FileTypeDetector.get_language(path).lower()

            if lang in normalized_langs:
                filtered.append(chunk)

        logger.debug(
            "Filter by language", languages=languages, before=len(chunks), after=len(filtered)
        )

        return filtered

    def filter_by_path(self, chunks: List[Chunk], path_pattern: str) -> List[Chunk]:
        """
        Filter chunks by file path pattern (regex).

        Args:
            chunks: List of chunks to filter
            path_pattern: Regex pattern for paths

        Returns:
            Filtered list of chunks
        """
        if not path_pattern:
            return chunks

        try:
            pattern = re.compile(path_pattern, re.IGNORECASE)
            filtered = [chunk for chunk in chunks if pattern.search(chunk.metadata.file_path)]

            logger.debug(
                "Filter by path", pattern=path_pattern, before=len(chunks), after=len(filtered)
            )

            return filtered

        except re.error as e:
            logger.warning("Invalid regex pattern", pattern=path_pattern, error=str(e))
            return chunks

    def filter_by_chunk_type(
        self, chunks: List[Chunk], chunk_types: List[ChunkType]
    ) -> List[Chunk]:
        """
        Filter by the 18 official ChunkTypes.

        Args:
            chunks: List of chunks to filter
            chunk_types: List of ChunkTypes from models

        Returns:
            Filtered list of chunks
        """
        if not chunk_types:
            return chunks

        # Convert to set for O(1) lookup
        type_set = set(chunk_types)

        filtered = [chunk for chunk in chunks if chunk.metadata.chunk_type in type_set]

        logger.debug(
            "Filter by chunk type",
            types=[t.value for t in chunk_types],
            before=len(chunks),
            after=len(filtered),
        )

        return filtered

    def apply(self, chunks: List[Chunk], filters: Dict[str, Any]) -> List[Chunk]:
        """
        Apply multiple filters in sequence.

        Args:
            chunks: List of chunks to filter
            filters: Dict of filter criteria

        Returns:
            Filtered list of chunks
        """
        result = chunks

        # Apply each filter if specified
        if 'file_types' in filters:
            result = self.filter_by_type(result, filters['file_types'])

        if 'languages' in filters:
            result = self.filter_by_language(result, filters['languages'])

        if 'date_from' in filters or 'date_to' in filters:
            result = self.filter_by_date(result, filters.get('date_from'), filters.get('date_to'))

        if 'path_pattern' in filters:
            result = self.filter_by_path(result, filters['path_pattern'])

        if 'chunk_types' in filters:
            result = self.filter_by_chunk_type(result, filters['chunk_types'])

        logger.info(
            "Applied filters",
            filter_count=len(filters),
            input_count=len(chunks),
            output_count=len(result),
        )

        return result

    def to_weaviate_where(self, filters: Dict[str, Any]) -> Optional[Dict]:
        """
        Convert filters to Weaviate where clause for pre-filtering.

        Only some filters can be applied at Weaviate level.
        Complex filters are applied post-search.

        Args:
            filters: Filter criteria

        Returns:
            Weaviate where clause or None
        """
        operands = []

        # File types filter
        if 'file_types' in filters and filters['file_types']:
            file_type_operands = []
            for file_type in filters['file_types']:
                ext = file_type if file_type.startswith('.') else f'.{file_type}'
                file_type_operands.append(
                    {"path": ["file_path"], "operator": "Like", "valueString": f"*{ext}"}
                )

            if len(file_type_operands) == 1:
                operands.append(file_type_operands[0])
            else:
                operands.append({"operator": "Or", "operands": file_type_operands})

        # Language filter (using FileTypeDetector)
        if 'languages' in filters and filters['languages']:
            lang_operands = []

            # Get all supported extensions
            all_extensions = FileTypeDetector.get_all_supported_extensions()

            # Find extensions that map to requested languages
            for ext in all_extensions:
                path = Path(f"test{ext}")
                detected_lang = FileTypeDetector.get_language(path).lower()

                if detected_lang in [lang.lower() for lang in filters['languages']]:
                    lang_operands.append(
                        {"path": ["file_path"], "operator": "Like", "valueString": f"*{ext}"}
                    )

            if lang_operands:
                if len(lang_operands) == 1:
                    operands.append(lang_operands[0])
                else:
                    operands.append({"operator": "Or", "operands": lang_operands})

        # Chunk type filter
        if 'chunk_types' in filters and filters['chunk_types']:
            chunk_type_operands = []
            for chunk_type in filters['chunk_types']:
                # Convert ChunkType enum to string if needed
                type_value = chunk_type.value if hasattr(chunk_type, 'value') else str(chunk_type)
                chunk_type_operands.append(
                    {
                        "path": ["chunk_type"],
                        "operator": "Equal",
                        "valueString": type_value,  # ChunkType values are lowercase
                    }
                )

            if len(chunk_type_operands) == 1:
                operands.append(chunk_type_operands[0])
            else:
                logger.info("[UNTESTED PATH] Multiple chunk types in filter")
                operands.append({"operator": "Or", "operands": chunk_type_operands})

        # Date filters
        if 'date_from' in filters and filters['date_from']:
            operands.append(
                {
                    "path": ["last_modified"],
                    "operator": "GreaterThanEqual",
                    "valueDate": (
                        filters['date_from'].isoformat()
                        if hasattr(filters['date_from'], 'isoformat')
                        else str(filters['date_from'])
                    ),
                }
            )

        if 'date_to' in filters and filters['date_to']:
            operands.append(
                {
                    "path": ["last_modified"],
                    "operator": "LessThanEqual",
                    "valueDate": (
                        filters['date_to'].isoformat()
                        if hasattr(filters['date_to'], 'isoformat')
                        else str(filters['date_to'])
                    ),
                }
            )

        # Path pattern filter (simple containment, not full regex)
        if 'path_pattern' in filters and filters['path_pattern']:
            # Weaviate doesn't support full regex, so we do simple contains
            # Remove regex anchors only at start/end, preserve middle occurrences
            pattern = filters['path_pattern']
            # Remove ^ only at start
            simple_pattern = re.sub(r'^\^', '', pattern)
            # Remove $ only at end
            simple_pattern = re.sub(r'\$$', '', simple_pattern)
            # Replace .* with * for Weaviate Like operator
            simple_pattern = simple_pattern.replace('.*', '*')
            # Replace escaped dots with regular dots
            simple_pattern = simple_pattern.replace(r'\.', '.')
            # Always add wrapper wildcards
            simple_pattern = '*' + simple_pattern + '*'

            operands.append(
                {"path": ["file_path"], "operator": "Like", "valueString": simple_pattern}
            )

        # Build final where clause
        if not operands:
            return None
        elif len(operands) == 1:
            return operands[0]
        else:
            logger.debug("Generated Weaviate where clause", conditions=len(operands))
            return {"operator": "And", "operands": operands}
