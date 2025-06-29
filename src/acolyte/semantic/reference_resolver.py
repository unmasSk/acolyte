"""
Temporal reference resolver in conversations.

Detects when the user refers to previous sessions or contexts so that ConversationService can search and load the appropriate context.
"""

import re
from typing import List

from acolyte.core.logging import logger
from acolyte.core.utils.datetime_utils import utc_now
from acolyte.core.tracing import MetricsCollector
from acolyte.models.semantic_types import SessionReference


class ReferenceResolver:
    """Detects references to previous sessions/contexts."""

    # Temporal reference patterns in Spanish
    SPANISH_PATTERNS = [
        r"lo que (?:hicimos|estábamos haciendo|trabajamos)",
        r"(?:recuerdas|acuérdate) cuando",
        r"sobre (?:el|la|los|las) (.+?) que (?:hablamos|vimos|refactorizamos)",
        r"en la (?:conversación|sesión) (?:anterior|pasada)",
        r"ayer (?:estuvimos|trabajamos) (?:en|con)",
        r"la última vez que",
        r"cuando (?:arreglamos|implementamos|creamos)",
        r"aquel (?:bug|problema|código) (?:de|que)",
        r"el (?:auth|login|sistema) que (?:cambiamos|modificamos)",
    ]

    # Temporal reference patterns in English
    ENGLISH_PATTERNS = [
        r"what we (?:did|were doing|worked on)",
        r"(?:remember|recall) when",
        r"about the (.+?) (?:we|that we) (?:discussed|saw|refactored)",
        r"in the (?:previous|last) (?:conversation|session)",
        r"yesterday we (?:were|worked) (?:on|with)",
        r"the last time (?:we|that)",
        r"when we (?:fixed|implemented|created)",
        r"that (?:bug|issue|code) (?:from|that)",
        r"the (?:auth|login|system) (?:we|that we) (?:changed|modified)",
    ]

    def __init__(self):
        # Combine patterns from both languages
        self.reference_patterns = self.SPANISH_PATTERNS + self.ENGLISH_PATTERNS
        # Metrics
        self.metrics = MetricsCollector()
        logger.info("ReferenceResolver initialized", total_patterns=len(self.reference_patterns))

    def resolve_temporal_references(self, message: str) -> List[SessionReference]:
        """
        Detects references to previous sessions.

        DOES NOT perform semantic search, only detects patterns so that
        ConversationService can perform the actual search.

        Args:
            message: User message

        Returns:
            List of detected references
        """
        start_time = utc_now()

        references = []
        message_lower = message.lower()

        for pattern in self.reference_patterns:
            match = re.search(pattern, message_lower)
            if match:
                # Extract context hint if there's a captured group
                context_hint = None
                if match.groups():
                    # Find first non-empty group
                    context_hint = next((g for g in match.groups() if g), None)

                reference = SessionReference(
                    pattern_matched=pattern,
                    context_hint=context_hint,
                    search_type="temporal",  # Always temporal, not semantic
                )

                references.append(reference)
                logger.info(
                    "Detected temporal reference", pattern=pattern[:30], context_hint=context_hint
                )
                self.metrics.increment("semantic.reference_resolver.temporal_references_detected")

        # Detect more specific references
        specific_refs = self._detect_specific_references(message)
        references.extend(specific_refs)
        if specific_refs:
            self.metrics.increment(
                "semantic.reference_resolver.specific_references_detected", len(specific_refs)
            )

        # Remove duplicates maintaining order
        seen = set()
        unique_refs = []
        for ref in references:
            # Use pattern as unique key
            if ref.pattern_matched not in seen:
                seen.add(ref.pattern_matched)
                unique_refs.append(ref)

        # Record metrics
        elapsed_ms = (utc_now() - start_time).total_seconds() * 1000
        self.metrics.record("semantic.reference_resolver.resolution_time_ms", elapsed_ms)
        self.metrics.increment("semantic.reference_resolver.references_resolved")
        if unique_refs:
            self.metrics.record(
                "semantic.reference_resolver.references_per_message", len(unique_refs)
            )

        return unique_refs

    def _detect_specific_references(self, message: str) -> List[SessionReference]:
        """Detects more specific references to code or files."""
        specific_refs = []

        # References to specific files with temporal context
        file_temporal_pattern = r"(?:el archivo|the file) ([\w\-]+\.(?:py|js|ts|java)) (?:que|that) (?:modificamos|cambiamos|we modified|we changed)"
        matches = re.finditer(file_temporal_pattern, message.lower())
        for match in matches:
            specific_refs.append(
                SessionReference(
                    pattern_matched=file_temporal_pattern,
                    context_hint=match.group(1),  # The mentioned file
                    search_type="temporal",
                )
            )

        # References to functions with temporal context
        function_temporal_pattern = (
            r"(?:la función|the function) (\w+) (?:que|that) (?:arreglamos|corregimos|we fixed)"
        )
        matches = re.finditer(function_temporal_pattern, message.lower())
        for match in matches:
            specific_refs.append(
                SessionReference(
                    pattern_matched=function_temporal_pattern,
                    context_hint=f"function:{match.group(1)}",
                    search_type="temporal",
                )
            )

        # References to specific bugs or issues
        bug_pattern = (
            r"(?:el bug|the bug|el problema|the issue) (?:de|del|of|from) (.+?) (?:que|that)"
        )
        matches = re.finditer(bug_pattern, message.lower())
        for match in matches:
            specific_refs.append(
                SessionReference(
                    pattern_matched=bug_pattern,
                    context_hint=f"bug:{match.group(1)}",
                    search_type="temporal",
                )
            )

        # References to sessions by description
        session_pattern = r"(?:cuando|when) (?:hablamos de|discutimos|we talked about|we discussed) (.+?)(?:\.|,|$)"
        matches = re.finditer(session_pattern, message.lower())
        for match in matches:
            topic = match.group(1).strip()
            if len(topic) > 3:  # Avoid very short matches
                specific_refs.append(
                    SessionReference(
                        pattern_matched=session_pattern,
                        context_hint=f"topic:{topic}",
                        search_type="temporal",
                    )
                )

        return specific_refs
