"""
Dream Analyzer - Deep code analysis engine.

Performs comprehensive analysis of the codebase using extended context windows.
Implements sliding window strategy for 32k models and full context for 128k.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from acolyte.core.secure_config import Settings

class AnalysisCapability(Enum):
    """Analysis capability levels based on available components."""

    FULL: str
    LIMITED: str
    MINIMAL: str

class DreamWindowManager:
    """
    Manages context windows for Dream analysis.

    Implements sliding window for 32k models and full context for 128k+.
    """

    context_size: int
    usable_context: int
    strategy: str
    new_code_size: int
    preserved_context_size: int
    critical_context: List[Any]

    def __init__(self, context_size: int, config: Optional[Settings] = None) -> None: ...
    def prepare_next_cycle(self, previous_findings: Dict[str, Any]) -> str: ...
    def _extract_critical_findings(self, findings: Dict[str, Any]) -> Dict[str, Any]: ...
    def _compress_to_tokens(self, data: Dict[str, Any], max_tokens: int) -> str: ...

class DreamAnalyzer:
    """
    Core analysis engine for Dream system.

    Performs deep analysis using extended context windows to find:
    - Hidden bugs and vulnerabilities
    - Performance bottlenecks
    - Architectural issues
    - Code patterns and anti-patterns
    """

    config: Settings
    ollama: Any
    neural_graph: Any
    search: Optional[Any]
    embeddings: Optional[Any]
    context_size: int
    window_manager: DreamWindowManager
    prompts: Dict[str, str]
    capability: AnalysisCapability

    def __init__(self, weaviate_client: Optional[Any] = None) -> None: ...
    def _determine_capability_level(self) -> AnalysisCapability: ...
    def get_capability_info(self) -> Dict[str, Any]: ...
    def _get_limitations(self) -> List[str]: ...
    def _load_analysis_prompts(self) -> None: ...
    async def explore_codebase(
        self, focus_areas: Optional[List[str]] = None, context_size: int = 32768
    ) -> Dict[str, Any]: ...
    async def analyze_deeply(
        self, initial_findings: Dict[str, Any], priorities: Dict[str, float]
    ) -> Dict[str, Any]: ...
    async def _select_files_for_analysis(
        self, focus_areas: Optional[List[str]]
    ) -> List[Dict[str, Any]]: ...
    async def _get_analysis_candidates(self) -> Tuple[List[str], bool]: ...
    async def _get_recent_files_from_search(self) -> List[str]: ...
    async def _load_code_context(self, files: List[Dict[str, Any]]) -> str: ...
    async def _run_analysis(
        self, analysis_type: str, code: str, context: str
    ) -> Dict[str, Any]: ...
    async def _run_prioritized_analyses(
        self, code: str, context: str, priorities: Dict[str, float]
    ) -> Dict[str, Any]: ...
    async def _analyze_all_files(self, files: List[Dict[str, Any]]) -> Dict[str, Any]: ...
    def _deduplicate_findings(self, results: Dict[str, Any]) -> Dict[str, Any]: ...
    def _prioritize_findings(self, results: Dict[str, Any]) -> Dict[str, Any]: ...
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]: ...
    async def _get_default_project_files(self) -> List[str]: ...
    def _infer_language_from_extension(self, extension: str) -> str: ...
