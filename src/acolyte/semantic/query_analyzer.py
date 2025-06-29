"""
Query analyzer for dynamic token distribution.

Determines the type of query to optimize token distribution
between context and response.
"""

import re
from typing import List, Dict

from acolyte.core.logging import logger
from acolyte.core.utils.datetime_utils import utc_now
from acolyte.core.secure_config import Settings
from acolyte.core.tracing import MetricsCollector
from acolyte.models.semantic_types import TokenDistribution
from acolyte.semantic.utils import detect_language


class QueryAnalyzer:
    """Analyzes user intent to optimize tokens."""

    def __init__(self):
        settings = Settings()
        self.language = settings.get("semantic.language", "es")

        # Load keywords from config
        query_config = settings.get("semantic.query_analysis", {})
        self.generation_keywords = query_config.get("generation_keywords", {})
        self.simple_patterns = query_config.get("simple_question_patterns", {})

        # Defaults if no config
        if not self.generation_keywords:
            self.generation_keywords = self._get_default_generation_keywords()
        if not self.simple_patterns:
            self.simple_patterns = self._get_default_simple_patterns()

        # Metrics
        self.metrics = MetricsCollector()
        logger.info("QueryAnalyzer initialized", language=self.language)

    def analyze_query_intent(self, query: str) -> TokenDistribution:
        """
        Determines optimal token distribution based on the query.

        Args:
            query: User message

        Returns:
            TokenDistribution with type and ratios
        """
        start_time = utc_now()

        query_lower = query.lower()
        detected_lang = detect_language(query, self.language)

        # 1. File generation detection
        if self._is_generation_query(query_lower, detected_lang):
            logger.info("Detected generation query", token_allocation="75% response")
            result = TokenDistribution(
                type="generation",
                response_ratio=0.75,  # 75% for response
                context_ratio=0.25,  # 25% for context
            )
            self.metrics.increment("semantic.query_analyzer.query_type.generation")

        # 2. Simple question detection
        elif self._is_simple_question(query, detected_lang):
            logger.info("Detected simple question", token_allocation="20% response")
            result = TokenDistribution(
                type="simple",
                response_ratio=0.20,  # 20% for response
                context_ratio=0.80,  # 80% for context
            )
            self.metrics.increment("semantic.query_analyzer.query_type.simple")

        # 3. Default: normal analysis/exploration
        else:
            logger.info("Detected normal query", token_allocation="10% response")
            result = TokenDistribution(
                type="normal",
                response_ratio=0.10,  # 10% for response
                context_ratio=0.90,  # 90% for context
            )
            self.metrics.increment("semantic.query_analyzer.query_type.normal")

        # Record metrics
        elapsed_ms = (utc_now() - start_time).total_seconds() * 1000
        self.metrics.record("semantic.query_analyzer.analysis_time_ms", elapsed_ms)
        self.metrics.increment("semantic.query_analyzer.queries_analyzed")

        return result

    def _is_generation_query(self, query: str, lang: str) -> bool:
        """Detects if it is a code/file generation query."""
        # Get language keywords
        keywords = self.generation_keywords.get(lang, [])
        if not keywords and lang != self.language:
            # Fallback to configured language
            keywords = self.generation_keywords.get(self.language, [])

        # Search for generation keywords
        for keyword in keywords:
            if keyword in query:
                return True

        # Additional patterns indicating generation
        generation_patterns = [
            r"archivo\s+completo",
            r"complete\s+file",
            r"código\s+completo",
            r"full\s+code",
            r"template",
            r"boilerplate",
            r"scaffold",
        ]

        for pattern in generation_patterns:
            if re.search(pattern, query):
                return True

        return False

    def _is_simple_question(self, query: str, lang: str) -> bool:
        """Detects if it is a simple/direct question."""
        # Criteria 1: Short question with ?
        words = query.split()
        if len(words) < 10 and "?" in query:
            return True

        # Criteria 2: Simple question patterns
        patterns = self.simple_patterns.get(lang, [])
        if not patterns and lang != self.language:
            patterns = self.simple_patterns.get(self.language, [])

        query_lower = query.lower()
        for pattern in patterns:
            if re.match(pattern, query_lower):
                return True

        # Criteria 3: Definition questions in any language
        universal_patterns = [
            r"^what\s+is\s+",
            r"^how\s+does\s+",
            r"^why\s+",
            r"^when\s+",
            r"^where\s+",
            r"^who\s+",
            r"^qué\s+es\s+",
            r"^cómo\s+funciona\s+",
            r"^por\s+qué\s+",
            r"^cuándo\s+",
            r"^dónde\s+",
            r"^quién\s+",
        ]

        for pattern in universal_patterns:
            if re.match(pattern, query_lower):
                return True

        return False

    def _get_default_generation_keywords(self) -> Dict[str, List[str]]:
        """Default keywords for generation detection."""
        return {
            "es": [
                "crea",
                "genera",
                "escribe",
                "implementa",
                "archivo completo",
                "hazme",
                "desarróllame",
                "código completo",
                "script completo",
            ],
            "en": [
                "create",
                "generate",
                "write",
                "implement",
                "complete file",
                "make me",
                "develop",
                "full code",
                "complete script",
            ],
        }

    def _get_default_simple_patterns(self) -> Dict[str, List[str]]:
        """Default patterns for simple questions."""
        return {
            "es": [
                r"^qué es\s+",
                r"^cómo funciona\s+",
                r"^para qué sirve\s+",
                r"^cuál es\s+",
                r"^explica\s+",
            ],
            "en": [
                r"^what is\s+",
                r"^how does\s+",
                r"^what's the purpose\s+",
                r"^which is\s+",
                r"^explain\s+",
            ],
        }
