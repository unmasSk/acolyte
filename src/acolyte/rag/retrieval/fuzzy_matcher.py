"""
Fuzzy matching for query expansion in code search.

This module normalizes naming conventions to find code regardless of style:
- snake_case ≈ camelCase ≈ kebab-case ≈ PascalCase
- getUserData ≈ get_user_data ≈ GetUserData

Part of ACOLYTE's mission to understand your code as you write it,
not as a style guide demands it.
"""

import re
from typing import List, Set, Tuple
from dataclasses import dataclass

from acolyte.core.logging import logger
from acolyte.core.secure_config import Settings


@dataclass
class FuzzyVariation:
    """A variation of a term with its transformation type."""

    term: str
    variation_type: str  # 'original', 'snake', 'camel', 'pascal', 'kebab'
    confidence: float = 1.0  # How confident we are this is a valid variation


class FuzzyMatcher:
    """
    Generates variations of search terms to handle naming conventions.

    This is NOT semantic similarity - it's purely syntactic transformation
    to handle different coding styles in the same project.
    """

    def __init__(self):
        """Initialize the fuzzy matcher with configuration."""
        config = Settings()
        fuzzy_config = config.get("rag.retrieval.fuzzy_matching", {})

        self.enabled = fuzzy_config.get("enabled", True)
        self.max_variations = fuzzy_config.get("max_variations", 5)
        self.min_term_length = fuzzy_config.get("min_term_length", 3)

        # Patterns for detecting naming styles
        self.snake_pattern = re.compile(r"^[a-z_][a-z0-9_]*$")
        self.camel_pattern = re.compile(r"^[a-z][a-zA-Z0-9]*$")
        self.pascal_pattern = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
        self.kebab_pattern = re.compile(r"^[a-z]+(-[a-z]+)*$")

        logger.info(
            "FuzzyMatcher initialized", enabled=self.enabled, max_variations=self.max_variations
        )

    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query with naming variations.

        Args:
            query: Original search query

        Returns:
            List of query variations (including original)
        """
        if not self.enabled:
            return [query]

        # Extract identifiers from the query
        identifiers = self._extract_identifiers(query)

        # Generate variations for each identifier
        all_variations = {}
        for identifier in identifiers:
            if len(identifier) >= self.min_term_length:
                variations = self._generate_variations(identifier)
                all_variations[identifier] = variations

        # If no variations found, return original
        if not all_variations:
            return [query]

        # Generate query variations by substituting terms
        query_variations = self._generate_query_variations(query, all_variations)

        # Limit total variations
        if len(query_variations) > self.max_variations:
            query_variations = query_variations[: self.max_variations]

        logger.debug("Fuzzy expansion", original=query, variations=len(query_variations) - 1)

        return query_variations

    def _extract_identifiers(self, query: str) -> Set[str]:
        """Extract potential code identifiers from query."""
        # Simple tokenization - can be improved
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_-]*\b", query)

        # Filter out common words (very basic stopword list)
        stopwords = {
            "the",
            "in",
            "at",
            "of",
            "for",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "been",
            "have",
            "has",
            "def",
            "class",
            "function",
            "method",
            "var",
            "const",
            "import",
            "from",
            "return",
            "if",
            "else",
            "while",
            "find",
        }

        identifiers = {
            token
            for token in tokens
            if token.lower() not in stopwords and len(token) >= self.min_term_length
        }

        return identifiers

    def _generate_variations(self, identifier: str) -> List[FuzzyVariation]:
        """Generate naming variations for an identifier."""
        variations = [FuzzyVariation(identifier, "original", 1.0)]

        # Detect current style
        current_style = self._detect_style(identifier)

        # Parse identifier into words
        words = self._parse_identifier(identifier, current_style)

        if not words:
            return variations

        # Generate different styles
        if current_style != "snake":
            snake = "_".join(words)
            variations.append(FuzzyVariation(snake, "snake", 0.9))

        if current_style != "camel" and len(words) > 1:
            camel = words[0] + "".join(w.capitalize() for w in words[1:])
            variations.append(FuzzyVariation(camel, "camel", 0.9))

        if current_style != "pascal":
            pascal = "".join(w.capitalize() for w in words)
            variations.append(FuzzyVariation(pascal, "pascal", 0.9))

        if current_style != "kebab":
            kebab = "-".join(words)
            variations.append(FuzzyVariation(kebab, "kebab", 0.8))

        # Add common variations (singular/plural for last word)
        last_word = words[-1] if words else ""
        if last_word and not last_word.endswith("s"):
            # Simple pluralization
            words_plural = words[:-1] + [last_word + "s"]

            # Add plural for all styles
            variations.append(FuzzyVariation("_".join(words_plural), "plural_snake", 0.7))

            if len(words_plural) > 1:
                plural_camel = words_plural[0] + "".join(w.capitalize() for w in words_plural[1:])
                variations.append(FuzzyVariation(plural_camel, "plural_camel", 0.7))

            plural_pascal = "".join(w.capitalize() for w in words_plural)
            variations.append(FuzzyVariation(plural_pascal, "plural_pascal", 0.7))
        else:
            logger.info("[UNTESTED PATH] Word already ends with 's' or empty")

        return variations

    def _detect_style(self, identifier: str) -> str:
        """Detect the naming style of an identifier."""
        if self.snake_pattern.match(identifier):
            return "snake"
        elif self.camel_pattern.match(identifier):
            return "camel"
        elif self.pascal_pattern.match(identifier):
            return "pascal"
        elif self.kebab_pattern.match(identifier):
            return "kebab"
        else:
            return "unknown"

    def _parse_identifier(self, identifier: str, style: str) -> List[str]:
        """Parse identifier into component words."""
        words = []

        if style == "snake":
            words = identifier.split("_")
        elif style == "kebab":
            words = identifier.split("-")
        elif style in ("camel", "pascal"):
            # Split on capital letters, handling sequences like HTTP, XML
            current_word = []
            for i, char in enumerate(identifier):
                if char.isupper():
                    if i > 0:
                        # Check if next char is lowercase (new word boundary)
                        if i + 1 < len(identifier) and identifier[i + 1].islower():
                            if current_word:
                                words.append("".join(current_word).lower())
                            current_word = [char]
                        # Check if previous char was lowercase (new word boundary)
                        elif i > 0 and identifier[i - 1].islower():
                            if current_word:
                                words.append("".join(current_word).lower())
                            current_word = [char]
                        else:
                            # Part of uppercase sequence
                            current_word.append(char)
                    else:
                        current_word.append(char)
                else:
                    current_word.append(char)
            if current_word:
                words.append("".join(current_word).lower())
        else:
            # Try to detect word boundaries heuristically
            # For now, just treat as single word
            words = [identifier.lower()]

        return [w for w in words if w]  # Filter empty strings

    def _generate_query_variations(self, query: str, identifier_variations: dict) -> List[str]:
        """Generate query variations by substituting identifiers."""
        variations = [query]  # Always include original

        # If we have multiple identifiers, also generate some combined variations
        if len(identifier_variations) > 1:
            # Get all identifiers and their variations
            identifiers = list(identifier_variations.keys())

            # Generate combinations of variations for first two identifiers
            if len(identifiers) >= 2:
                id1, id2 = identifiers[0], identifiers[1]
                for var1 in identifier_variations[id1]:
                    for var2 in identifier_variations[id2]:
                        if var1.term != id1 or var2.term != id2:  # At least one changed
                            new_query = query
                            new_query = re.sub(r"\b" + re.escape(id1) + r"\b", var1.term, new_query)
                            new_query = re.sub(r"\b" + re.escape(id2) + r"\b", var2.term, new_query)
                            if new_query not in variations:
                                variations.append(new_query)
                                # Limit to avoid explosion
                                if len(variations) >= self.max_variations:
                                    return variations

        # Also do single substitutions
        for identifier, id_variations in identifier_variations.items():
            for variation in id_variations:
                if variation.term != identifier:  # Skip original
                    # Case-sensitive replacement
                    new_query = re.sub(r"\b" + re.escape(identifier) + r"\b", variation.term, query)
                    if new_query != query and new_query not in variations:
                        variations.append(new_query)
                        if len(variations) >= self.max_variations:
                            return variations

        return variations

    def find_similar_terms(
        self, term: str, candidates: List[str], threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find similar terms from a list of candidates.

        Args:
            term: Term to match
            candidates: List of candidate terms
            threshold: Minimum confidence threshold

        Returns:
            List of (candidate, confidence) tuples
        """
        if not self.enabled:
            return []

        # Generate variations for the search term
        term_variations = self._generate_variations(term)
        term_variants = {v.term.lower() for v in term_variations}

        matches = []
        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Exact match (case-insensitive)
            if candidate_lower in term_variants:
                matches.append((candidate, 1.0))
                logger.info("[UNTESTED PATH] Exact match found in find_similar_terms")
                continue

            # Check if candidate is a variation
            candidate_variations = self._generate_variations(candidate)
            candidate_variants = {v.term.lower() for v in candidate_variations}

            # Check overlap
            if term_variants & candidate_variants:
                # They share a common variation
                confidence = 0.8
                matches.append((candidate, confidence))

        # Sort by confidence and filter by threshold
        matches = [(c, conf) for c, conf in matches if conf >= threshold]
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches


# Global instance for convenience
_fuzzy_matcher = None


def get_fuzzy_matcher() -> FuzzyMatcher:
    """Get global fuzzy matcher instance."""
    global _fuzzy_matcher
    if _fuzzy_matcher is None:
        _fuzzy_matcher = FuzzyMatcher()
    return _fuzzy_matcher
