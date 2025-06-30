"""
Technical decision detector in conversations.

Automatically identifies when important architectural, library, pattern, or security decisions are made.
"""

import re
from typing import Optional, List, Dict

from acolyte.core.logging import logger
from acolyte.core.utils.datetime_utils import utc_now
from acolyte.core.secure_config import Settings
from acolyte.core.tracing import MetricsCollector
from acolyte.models.technical_decision import DecisionType
from acolyte.models.semantic_types import DetectedDecision
from acolyte.semantic.utils import detect_language


class DecisionDetector:
    """Detects important technical decisions."""

    def __init__(self):
        settings = Settings()
        self.language = settings.get("semantic.language", "es")

        # Load configuration
        decision_config = settings.get("semantic.decision_detection", {})
        self.auto_detect = decision_config.get("auto_detect", True)
        self.explicit_marker = decision_config.get("explicit_marker", "@decision")
        self.patterns = decision_config.get("patterns", {})

        # Default patterns only if auto_detect is enabled
        if not self.patterns and self.auto_detect:
            self.patterns = self._get_default_patterns()

        # Metrics
        self.metrics = MetricsCollector()
        logger.info(
            "DecisionDetector initialized", language=self.language, auto_detect=self.auto_detect
        )

    def detect_technical_decision(
        self, message: str, context: Optional[str] = None
    ) -> Optional[DetectedDecision]:
        """
        Detects technical decisions in the message.

        Args:
            message: Message to analyze
            context: Additional context if available

        Returns:
            DetectedDecision if detected, None otherwise
            Note: Does not include session_id/task_id, which must be added by the service
        """
        start_time = utc_now()

        result = None

        # 1. Explicit marker detection
        if self.explicit_marker in message:
            result = self._extract_explicit_decision(message)
            self.metrics.increment("semantic.decision_detector.explicit_decisions")

        # 2. Automatic detection if enabled and no explicit marker
        elif self.auto_detect:
            detected_lang = detect_language(message, self.language)
            result = self._detect_automatic_decision(message, detected_lang, context)
            if result:
                self.metrics.increment("semantic.decision_detector.automatic_decisions")
                self.metrics.increment(
                    f"semantic.decision_detector.decision_type.{result.decision_type.lower()}"
                )

        # Register metrics
        elapsed_ms = (utc_now() - start_time).total_seconds() * 1000
        self.metrics.record("semantic.decision_detector.detection_time_ms", elapsed_ms)
        self.metrics.increment("semantic.decision_detector.decisions_analyzed")

        if result:
            self.metrics.increment("semantic.decision_detector.decisions_detected")
            self.metrics.record("semantic.decision_detector.impact_level", result.impact_level)

        return result

    def _extract_explicit_decision(self, message: str) -> DetectedDecision:
        """Extracts explicitly marked decision."""
        # Search for the marker and extract the text after
        pattern = rf"{re.escape(self.explicit_marker)}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, message, re.IGNORECASE)

        if match:
            decision_text = match.group(1).strip()
        else:
            # If marker is inline without ':', extract rest of line
            inline_pattern = rf"{re.escape(self.explicit_marker)}\s+(.+?)(?:\n|$)"
            inline_match = re.search(inline_pattern, message, re.IGNORECASE)

            if inline_match:
                decision_text = inline_match.group(1).strip()
            else:
                # Take line after marker
                logger.warning("[UNTESTED PATH] Extracting decision from line after marker")
                lines = message.split("\n")
                for i, line in enumerate(lines):
                    if self.explicit_marker in line:
                        # If marker is at end of line
                        remaining = line.split(self.explicit_marker, 1)[1].strip()
                        if remaining:
                            decision_text = remaining
                        elif i + 1 < len(lines):
                            decision_text = lines[i + 1].strip()
                        else:
                            decision_text = "Decisión técnica marcada"
                        break
                else:
                    logger.warning("[UNTESTED PATH] No decision text found after marker")
                    decision_text = "Decisión técnica marcada"

        logger.info("Detected explicit decision", decision_text=decision_text[:50])

        # Return intermediate type without IDs
        return DetectedDecision(
            decision_type=DecisionType.ARCHITECTURE,  # Default for explicit
            title=decision_text[:100],
            description=message,
            rationale="Marcada explícitamente por el usuario",
            impact_level=3,  # Medium impact by default
            alternatives_considered=[],
        )

    def _detect_automatic_decision(
        self, message: str, lang: str, context: Optional[str] = None
    ) -> Optional[DetectedDecision]:
        """Automatically detects decisions using patterns."""
        lang_patterns = self.patterns.get(lang, [])
        if not lang_patterns:
            lang_patterns = self.patterns.get(self.language, [])

        full_context = f"{message}\n{context}" if context else message

        # Search for decision patterns
        # Search for decision patterns with original message to preserve case
        for pattern in lang_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return self._extract_decision_from_match(match, pattern, message, full_context)

        # Universal patterns (technical)
        logger.info("[UNTESTED BRANCH] Checking universal patterns for automatic decision")
        universal_patterns = [
            (r"usar[é]?\s+(\w+)(?:\s+(?:en lugar de|instead of)\s+(\w+))?", DecisionType.LIBRARY),
            (
                r"implement(?:ar[é]?)?\s+(.+?)\s+(?:usando|using|with)\s+(\w+)",
                DecisionType.ARCHITECTURE,
            ),
            (r"(?:patrón|pattern)\s+(\w+)\s+(?:para|for)\s+(.+)", DecisionType.PATTERN),
            (r"(?:arquitectura|architecture):\s*(.+)", DecisionType.ARCHITECTURE),
            (r"(?:seguridad|security):\s*(.+)", DecisionType.SECURITY),
        ]

        for pattern, decision_type in universal_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return self._create_decision_from_pattern(
                    match, decision_type, message, full_context
                )

        return None

    def _extract_decision_from_match(
        self, match: re.Match, pattern: str, message: str, context: str
    ) -> DetectedDecision:
        """Extracts decision from a pattern match."""
        groups = match.groups()

        # Determine type based on pattern
        if "arquitectura" in pattern or "architecture" in pattern:
            decision_type = DecisionType.ARCHITECTURE
        elif any(word in pattern for word in ["usar", "use", "library", "librería"]):
            decision_type = DecisionType.LIBRARY
        elif any(word in pattern for word in ["patrón", "pattern"]):
            decision_type = DecisionType.PATTERN
        else:
            decision_type = DecisionType.ARCHITECTURE  # Default

        # Extract title and alternatives
        if len(groups) >= 1:
            # Preserve case of captured text
            title = f"Usar {groups[0]}"
            alternatives = list(groups[1:]) if len(groups) > 1 else None
        else:
            title = match.group(0)
            alternatives = None

        # Extract rationale from context
        rationale = self._extract_rationale(message, match.end())

        logger.info("Detected technical decision", decision_type=decision_type.value, title=title)

        return DetectedDecision(
            decision_type=decision_type,
            title=title,
            description=message,
            rationale=rationale or "Detectado automáticamente por patrón",
            alternatives_considered=alternatives or [],
            impact_level=self._estimate_impact(decision_type, message),
        )

    def _create_decision_from_pattern(
        self, match: re.Match, decision_type: str, message: str, context: str
    ) -> DetectedDecision:
        """Creates decision from a universal pattern."""
        groups = match.groups()

        # Convert decision_type to DecisionType if it's str
        if isinstance(decision_type, str):
            try:
                decision_type_enum = DecisionType(decision_type)
            except ValueError:
                decision_type_enum = DecisionType.ARCHITECTURE
        else:
            decision_type_enum = decision_type

        # Build title based on type
        if decision_type_enum == DecisionType.LIBRARY:
            if len(groups) >= 2 and groups[1] is not None:
                title = f"Usar {groups[0]} en lugar de {groups[1]}"
                alternatives = [groups[1]]
            elif len(groups) >= 1:
                title = f"Usar {groups[0]}"
                alternatives = None
            else:
                title = match.group(0)
                alternatives = None
        elif decision_type_enum == DecisionType.ARCHITECTURE and len(groups) >= 2:
            title = f"Implementar {groups[0]} con {groups[1]}"
            alternatives = None
        else:
            title = groups[0] if len(groups) >= 1 else match.group(0)
            alternatives = None

        rationale = self._extract_rationale(message, match.end())

        return DetectedDecision(
            decision_type=decision_type_enum,
            title=title[:100],
            description=message,
            rationale=rationale or "Detectado por patrón universal",
            alternatives_considered=alternatives or [],
            impact_level=self._estimate_impact(decision_type_enum, message),
        )

    def _extract_rationale(self, message: str, start_pos: int) -> Optional[str]:
        """Extracts the rationale after the decision."""
        logger.info("[UNTESTED BRANCH] Extracting rationale from decision")
        remaining_text = message[start_pos:].strip()

        # If no remaining text, return None
        if not remaining_text:
            return None

        # If remaining text starts with dot, it's a complete new sentence
        # that's probably the complete rationale
        if remaining_text.startswith(".") and len(remaining_text) > 1:
            # Remove initial dot and spaces
            after_dot = remaining_text[1:].strip()
            if after_dot:
                # If there's another dot, take until there; if not, take all
                if "." in after_dot:
                    next_sentence = after_dot.split(".")[0].strip()
                else:
                    next_sentence = after_dot.strip()
                if next_sentence:
                    return next_sentence[:200]

        # Search for words that indicate rationale ONLY if we don't start with dot
        # (because if it starts with dot, the complete sentence is the rationale)
        if not remaining_text.startswith("."):
            rationale_indicators = [
                "porque",
                "because",
                "para",
                "for",
                "to",
                "ya que",
                "since",
                "debido a",
                "due to",
            ]

            for indicator in rationale_indicators:
                pattern = rf"\b{indicator}\b\s+(.+?)(?:\.|$)"
                match = re.search(pattern, remaining_text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()[:200]

        # If we don't find indicators and there's a dot in the middle,
        # search for text after first dot
        if "." in remaining_text and not remaining_text.startswith("."):
            parts = remaining_text.split(".", 1)
            if len(parts) > 1 and parts[1].strip():
                # Take text after first dot
                next_part = parts[1].strip()
                # If there's another dot, take until there
                if "." in next_part:
                    return next_part.split(".")[0].strip()[:200]
                else:
                    return next_part[:200]

        # If there's no dot, take what remains
        if remaining_text and not remaining_text.endswith("."):
            sentence = remaining_text.lstrip(".,;: ")
            if sentence:
                return sentence[:200]

        return None

    def _get_default_patterns(self) -> Dict[str, List[str]]:
        """Default patterns if not present in config."""
        return {
            "es": [
                r"vamos a usar\s+(\w+)",
                r"decidí implementar",
                r"usaremos\s+(\w+)\s+para",
                r"mejor\s+(.+?)\s+que\s+(.+?)\s+porque",
                r"usar\s+(\w+)(?:\s+para)?",  # Pattern más flexible para "usar X"
            ],
            "en": [
                r"we[''']ll\s+use\s+(\w+)",  # Matches "we'll" with different apostrophes
                r"I decided to implement",
                r"we[''']ll\s+use\s+(\w+)\s+for",
                r"(.+?)\s+is better than\s+(.+?)\s+because",
                r"use\s+(\w+)(?:\s+for)?",  # Pattern más flexible para "use X"
            ],
        }

    def _estimate_impact(self, decision_type: DecisionType, message: str) -> int:
        """Estimates the impact level based on type and content."""
        # Security is always high impact
        if decision_type == DecisionType.SECURITY:
            return 5

        # Architecture is typically high
        if decision_type == DecisionType.ARCHITECTURE:
            return 4

        # Words that indicate high impact
        high_impact_words = [
            "crítico",
            "critical",
            "importante",
            "important",
            "principal",
            "main",
            "core",
            "fundamental",
        ]

        message_lower = message.lower()
        if any(word in message_lower for word in high_impact_words):
            return 4

        # Library and Pattern are typically medium
        return 3
