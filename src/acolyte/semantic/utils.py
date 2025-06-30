"""
Shared utilities for the Semantic module.
"""

import re


def detect_language(text: str, default: str = "es") -> str:
    """
    Detects the language of the text (es/en).

    Args:
        text: Text to analyze
        default: Default language if it cannot be determined

    Returns:
        'es' for Spanish, 'en' for English
    """
    # Remove "de" from Spanish indicators as it's too common and appears in mixed contexts
    spanish_indicators = [
        "vamos",
        "necesito",
        "quiero",
        "hacer",
        "crear",
        "implementar",
        "el",
        "la",
        "los",
        "las",
        # "de",  # Removed - too common in mixed contexts
        "que",
        "para",
        "con",
        "por",
        "sobre",
        "cuando",
        "donde",
        "está",
        "cómo",
        "sistema",  # Common in Spanish tech contexts
        # "archivo",  # Removed - used in English contexts too
        "hola",  # Spanish greeting
        "ayuda",  # Spanish help
    ]
    english_indicators = [
        "let's",
        "need",
        "needs",  # Added plural/conjugated forms
        "want",
        "make",
        "create",
        "implement",
        "the",
        "a",
        "an",
        "to",
        "of",
        "that",
        "for",
        "with",
        "about",
        "when",
        "where",
        "what",
        "what's",  # Common contractions
        "don't",
        "can't",
        "won't",
        "i",
        "you",
        "we",
        "they",
        "is",
        "are",
        "on",
        "in",
        "hi",
        "hello",  # Common greetings
        "help",
        "some",
        "use",
        "file",
        "error",
        "how",
        "can",
        "line",
        "know",
        "wrong",
        "code",  # Common in programming
        "refactoring",  # Technical terms
    ]

    # Normalize apostrophes in the text for better matching
    text_lower = text.lower()
    text_normalized = text_lower.replace("'", "'").replace("'", "'")

    # Search for complete words (with word boundaries)
    spanish_count = 0
    english_count = 0

    for word in spanish_indicators:
        # Search word with boundaries (start, end or surrounded by spaces/punctuation)
        if re.search(r"\b" + re.escape(word) + r"\b", text_normalized):
            spanish_count += 1

    for word in english_indicators:
        # Search with word boundaries in normalized text
        if re.search(r"\b" + re.escape(word) + r"\b", text_normalized):
            english_count += 1

    # Check for strong patterns that override simple counting
    strong_english_patterns = [
        "let's create",
        "let's use",
        "let's implement",
        "i need",
        "what do you",
        "the .* needs",
    ]
    strong_spanish_patterns = ["vamos a", "necesito crear", "qué es", "cómo puedo", "hola,"]

    for pattern in strong_spanish_patterns:
        if pattern in text_normalized:
            return "es"

    for pattern in strong_english_patterns:
        # Special handling for regex patterns
        if ".*" in pattern:
            if re.search(pattern, text_normalized):
                return "en"
        elif pattern in text_normalized:
            return "en"

    # For mixed text with small difference, use default
    # This handles cases like "Hello, necesito help" which is intentionally ambiguous
    diff = abs(english_count - spanish_count)

    # If it's clearly mixed (has words from both) and difference is small
    if english_count > 0 and spanish_count > 0 and diff <= 1:
        return default

    # Clear winner
    if english_count > spanish_count:
        return "en"
    elif spanish_count > english_count:
        return "es"
    else:
        return default
