"""
Dream Orchestrator - Main coordinator for deep analysis system.

Manages the complete Dream cycle from fatigue detection to insight generation.
Always requires explicit user permission before starting analysis.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from acolyte.core.database import DatabaseManager
from acolyte.core.secure_config import Settings
from .state_manager import DreamStateManager
from .fatigue_monitor import FatigueMonitor
from .analyzer import DreamAnalyzer
from .insight_writer import InsightWriter

class DreamTrigger(str, Enum):
    USER_REQUEST = "USER_REQUEST"
    FATIGUE_SUGGESTION = "FATIGUE_SUGGESTION"

class DreamOrchestrator:
    config: Settings
    db: DatabaseManager
    weaviate_client: Optional[Any]
    state_manager: DreamStateManager
    fatigue_monitor: FatigueMonitor
    analyzer: DreamAnalyzer
    insight_writer: InsightWriter
    fatigue_threshold: float
    emergency_threshold: float
    cycle_duration: float

    def __init__(self, weaviate_client: Optional[Any] = None) -> None: ...
    async def check_fatigue_level(self) -> Dict[str, Any]: ...
    def generate_suggestion_message(self, fatigue_level: float, is_emergency: bool) -> str: ...
    def _generate_suggestion(self, fatigue_level: float, is_emergency: bool) -> str: ...
    async def request_analysis(
        self,
        trigger: DreamTrigger,
        focus_areas: Optional[List[str]] = None,
        user_query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...
    def _generate_user_request_message(
        self, user_query: Optional[str], focus_areas: Optional[List[str]]
    ) -> str: ...
    def _get_analysis_benefits(
        self, trigger: DreamTrigger, focus_areas: Optional[List[str]]
    ) -> List[str]: ...
    async def start_analysis(
        self,
        request_id: str,
        approved: bool,
        focus_areas: Optional[List[str]] = None,
        priorities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]: ...
    async def _run_analysis_cycle(
        self,
        session_id: str,
        focus_areas: Optional[List[str]] = None,
        priorities: Optional[Dict[str, float]] = None,
    ) -> None: ...
    def _get_default_priorities(self) -> Dict[str, float]: ...
    async def _consolidate_findings(
        self, session_id: str, initial: Dict[str, Any], deep: Dict[str, Any]
    ) -> List[Dict[str, Any]]: ...
    async def _record_completion(self, session_id: str, insights_count: int) -> None: ...
    async def _record_error(self, session_id: str, error: str) -> None: ...
    def _deduplicate_findings(self, results: Dict[str, Any]) -> Dict[str, Any]: ...
    def _prioritize_findings(self, results: Dict[str, Any]) -> Dict[str, Any]: ...
    async def get_recent_insights(
        self, limit: int = 10, insight_type: Optional[str] = None
    ) -> List[Dict[str, Any]]: ...
    async def is_analysis_in_progress(self) -> bool: ...
