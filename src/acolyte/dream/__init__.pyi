"""
Dream System - Deep Search for your code.

Like Deep Search in modern AIs but specialized for your project.
Analyzes patterns, detects bugs, and optimizes memory during "sleep" cycles.
"""

from typing import Optional, Any
from .orchestrator import DreamOrchestrator

__version__: str
__all__: list[str]

def create_dream_orchestrator(weaviate_client: Optional[Any] = None) -> DreamOrchestrator: ...
