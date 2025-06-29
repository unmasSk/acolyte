"""
Extractive summary generator for conversations.

Uses regex techniques and entity extraction without ML libraries.
Goal: 70-80% reduction (500-1000 tokens -> 100-200 tokens).
"""

import re
from typing import List, Optional

from acolyte.core.logging import logger
from acolyte.core.utils.datetime_utils import utc_now
from acolyte.core.token_counter import get_token_counter
from acolyte.core.tracing import MetricsCollector
from acolyte.models.semantic_types import SummaryResult
from acolyte.models.chunk import Chunk


class Summarizer:
    """Generates ultra-fast extractive summaries."""

    # Patterns for entity extraction
    ENTITY_PATTERNS = {
        "file": r"\b([\w\-]+\.(?:py|js|ts|jsx|tsx|java|go|rs|md|yml|yaml|json))\b",
        "function": r"\b(?:def|function|func|fn|const|let|var)\s+(\w+)|\b(\w+)\s*\(",
        "class": r"\bclass\s+(\w+)|\b([A-Z]\w+)(?:Service|Controller|Model|Handler)",
        "module": r"(?:from|import)\s+([\w\.]+)",
        "line": r"(?:line|línea)\s*(\d+)",
        "error": r"(?:error|exception|Error|Exception):\s*(.+?)(?:\.|$)",
        "variable": r"\b(?:const|let|var|val)\s+(\w+)|(\w+)\s*=",
    }

    # Types of simple intentions
    INTENT_KEYWORDS = {
        "debugging": ["error", "bug", "fix", "problem", "crash"],
        "implementation": ["create", "implement", "add", "new"],
        "refactoring": ["refactor", "improve", "optimize", "clean"],
        "documentation": ["document", "explain", "comment"],
        "research": ["investigate", "analyze", "search", "understand"],
    }

    # Keywords to detect suggested actions
    ACTION_KEYWORDS = {
        "suggest": [r"sugiero", r"suggest", r"recomiendo", r"recommend"],
        "should": [r"deberías", r"should", r"convendría", r"would be"],
        "can": [r"puedes", r"can", r"podrías", r"could"],
        "need": [r"necesitas", r"need", r"hace falta", r"required"],
    }

    def __init__(self):
        self.token_counter = get_token_counter()
        # Use MetricsCollector by composition as indicated in Core
        self.metrics = MetricsCollector()
        logger.info("Summarizer initialized")

    async def generate_summary(
        self, user_msg: str, assistant_msg: str, context_chunks: Optional[List[Chunk]] = None
    ) -> SummaryResult:
        """
        Generates extractive summary using patterns and regex.

        Args:
            user_msg: User message
            assistant_msg: Assistant response
            context_chunks: Analyzed code chunks (to extract entities)

        Returns:
            SummaryResult with summary and metadata
        """
        start_time = utc_now()

        # Extract entities from all sources
        entities = self._extract_entities(user_msg, assistant_msg, context_chunks)

        # Detect user intention
        intent = self._detect_intent(user_msg)

        # Extract suggested action (NOT applied)
        action = self._extract_suggested_action(assistant_msg)

        # Format summary
        summary = self._format_summary(user_msg, assistant_msg, intent, action, entities)

        # Calculate reduction
        original_tokens = self.token_counter.count_tokens(user_msg + "\n" + assistant_msg)
        summary_tokens = self.token_counter.count_tokens(summary)
        tokens_saved = original_tokens - summary_tokens

        # Log timing
        elapsed_ms = (utc_now() - start_time).total_seconds() * 1000
        logger.info("Summary generated", elapsed_ms=elapsed_ms, tokens_saved=tokens_saved)

        # Register metrics
        self.metrics.record("semantic.summarizer.generation_time_ms", elapsed_ms)
        self.metrics.record(
            "semantic.summarizer.compression_ratio",
            summary_tokens / original_tokens if original_tokens > 0 else 0,
        )
        self.metrics.record("semantic.summarizer.tokens_saved", tokens_saved)
        self.metrics.increment("semantic.summarizer.summaries_generated")

        return SummaryResult(
            summary=summary, entities=entities, intent_type=intent, tokens_saved=tokens_saved
        )

    def _extract_entities(
        self, user_msg: str, assistant_msg: str, chunks: Optional[List[Chunk]] = None
    ) -> List[str]:
        """Extracts relevant entities from all sources."""
        entities = set()

        # Combine texts for analysis
        combined_text = f"{user_msg}\n{assistant_msg}"

        # Extract from messages
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            for match in re.finditer(pattern, combined_text, re.IGNORECASE):
                # Get captured group (may be in different positions)
                entity = next((g for g in match.groups() if g), None)
                if entity:
                    entities.add(f"{entity_type}:{entity}")

        # Extract from chunks if available
        if chunks:
            for chunk in chunks[:10]:  # Limit to first 10 for performance
                if chunk.metadata.file_path:
                    entities.add(f"file:{chunk.metadata.file_path}")
                if chunk.metadata.chunk_type:
                    # Extract function/class names from content
                    content_preview = chunk.content[:200]  # Only start
                    for match in re.finditer(self.ENTITY_PATTERNS["function"], content_preview):
                        entity = next((g for g in match.groups() if g), None)
                        if entity:
                            entities.add(f"function:{entity}")

        return sorted(list(entities))[:20]  # Maximum 20 entities

    def _detect_intent(self, user_msg: str) -> str:
        """Detects basic intent type."""
        user_lower = user_msg.lower()

        for intent_type, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in user_lower for kw in keywords):
                return intent_type

        return "general"

    def _extract_suggested_action(self, assistant_msg: str) -> str:
        """Extracts main action in a simple way."""
        # Search for the first sentence with an action verb
        action_verbs = [
            "sugiero",
            "recomiendo",
            "deberías",
            "puedes",
            "suggest",
            "recommend",
            "should",
            "can",
            "crear",
            "modificar",
            "cambiar",
            "agregar",
            "create",
            "modify",
            "change",
            "add",
        ]

        sentences = assistant_msg.split(".")
        for sentence in sentences[:3]:  # Only first 3 sentences
            sentence_lower = sentence.lower()
            if any(verb in sentence_lower for verb in action_verbs):
                return sentence.strip()[:60]  # Max 60 chars

        # If no sentence found, take the first relevant sentence
        first_sentence = sentences[0].strip() if sentences else ""
        if len(first_sentence) > 10:
            return first_sentence[:60]

        return "provided suggestions"

    def _format_summary(
        self, user_msg: str, assistant_msg: str, intent: str, action: str, entities: List[str]
    ) -> str:
        """Formats a concise but informative summary."""
        # Extract main intent from user's message
        user_intent = self._extract_main_intent(user_msg)

        # Main action of the assistant (max 80 chars)
        if len(action) > 80:
            action = action[:77] + "..."

        # Key entities (up to 4 main ones)
        key_entities = self._extract_key_entities(entities)

        # Detect if there's involved code
        has_code = any("function:" in e or "class:" in e for e in entities)

        # Build summary in parts
        parts = []

        # Part 1: Intent and context
        if intent != "general":
            parts.append(f"[{intent.upper()}]")
        parts.append(f"User: {user_intent}")

        # Part 2: Involved entities
        if key_entities:
            entities_str = ", ".join(key_entities[:4])
            parts.append(f"Context: {entities_str}")

        # Part 3: Action performed
        if action != "provided suggestions":
            parts.append(f"ACOLYTE: {action}")

        # Part 4: Additional indicators
        indicators = []
        if has_code:
            indicators.append("code")
        if any("error" in e for e in entities):
            indicators.append("error")
        if len(entities) > 10:
            indicators.append(f"{len(entities)} entities")

        if indicators:
            parts.append(f"[{', '.join(indicators)}]")

        return " | ".join(parts)

    def _extract_main_intent(self, user_msg: str) -> str:
        """Extracts the main intent from the user's message."""
        # Search for the first significant sentence
        sentences = re.split(r"[.!?]", user_msg)

        for sentence in sentences:
            sentence = sentence.strip()
            # Ignore very short sentences or greetings
            if len(sentence) > 10 and not any(
                greeting in sentence.lower() for greeting in ["hola", "hello", "buenos", "good"]
            ):
                return sentence[:60] + "..." if len(sentence) > 60 else sentence

        # Fallback: first 60 chars
        return user_msg[:60] + "..." if len(user_msg) > 60 else user_msg

    def _extract_key_entities(self, entities: List[str]) -> List[str]:
        """Extracts the most relevant entities."""
        # Prioritize by type
        priority_order = ["file", "class", "function", "error", "module"]
        key_entities = []

        for priority_type in priority_order:
            for entity in entities:
                if entity.startswith(f"{priority_type}:"):
                    _, name = entity.split(":", 1)
                    if name not in key_entities:
                        key_entities.append(name)

        return key_entities
