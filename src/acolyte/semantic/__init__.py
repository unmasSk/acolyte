"""
Semantic Module - Natural language processing without ML.

Provides text analysis, intent detection, and summary generation
using extractive techniques and regex.
"""

# Import all main classes
from acolyte.semantic.summarizer import Summarizer
from acolyte.semantic.task_detector import TaskDetector
from acolyte.semantic.prompt_builder import PromptBuilder
from acolyte.semantic.query_analyzer import QueryAnalyzer
from acolyte.semantic.decision_detector import DecisionDetector
from acolyte.semantic.reference_resolver import ReferenceResolver

# Re-export for easy access
__all__ = [
    "Summarizer",
    "TaskDetector",
    "PromptBuilder",
    "QueryAnalyzer",
    "DecisionDetector",
    "ReferenceResolver",
]

# Module version
__version__ = "0.1.0"
