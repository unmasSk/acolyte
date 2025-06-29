"""
Dream Orchestrator - Main coordinator for deep analysis system.

Manages the complete Dream cycle from fatigue detection to insight generation.
Always requires explicit user permission before starting analysis.
"""

from typing import Dict, Any, List, Optional
import asyncio
import json
import zlib
from enum import Enum

from acolyte.core.logging import logger
from acolyte.core.id_generator import generate_id
from acolyte.core.database import get_db_manager, FetchType
from acolyte.core.secure_config import Settings
from acolyte.core.exceptions import ValidationError, DatabaseError
from acolyte.core.utils.datetime_utils import utc_now, utc_now_iso, add_time

from .state_manager import DreamStateManager, DreamState
from .fatigue_monitor import FatigueMonitor
from .analyzer import DreamAnalyzer
from .insight_writer import InsightWriter


class DreamTrigger(str, Enum):
    """Types of triggers that can initiate Dream analysis."""

    USER_REQUEST = "USER_REQUEST"  # User explicitly asks for deep analysis
    FATIGUE_SUGGESTION = "FATIGUE_SUGGESTION"  # ChatService suggests based on fatigue + user query


class DreamOrchestrator:
    """
    Orchestrates the complete Dream analysis cycle.

    Like Deep Search in ChatGPT/Claude but for your codebase.
    Always requires user permission before starting.
    """

    def __init__(self, weaviate_client: Optional[Any] = None) -> None:
        """
        Initialize Dream orchestrator with all components.

        Args:
            weaviate_client: Optional Weaviate client for search functionality.
                           Should be injected from the application layer.
        """
        self.config = Settings()
        self.db = get_db_manager()

        # Store weaviate client
        self.weaviate_client = weaviate_client

        # Initialize components
        self.state_manager = DreamStateManager()
        self.fatigue_monitor = FatigueMonitor(weaviate_client=weaviate_client)
        self.analyzer = DreamAnalyzer(weaviate_client=weaviate_client)
        self.insight_writer = InsightWriter()

        # Get Dream configuration
        self.fatigue_threshold = self.config.get("dream.fatigue_threshold", 7.5)
        self.emergency_threshold = self.config.get("dream.emergency_threshold", 9.5)
        self.cycle_duration = self.config.get("dream.cycle_duration_minutes", 5)

        logger.info(
            "Dream orchestrator initialized",
            module="dream",
            has_weaviate=weaviate_client is not None,
        )

    async def check_fatigue_level(self) -> Dict[str, Any]:
        """
        Check current fatigue level.

        Note: This method ONLY calculates fatigue. It does NOT suggest optimization.
        Suggestions should come from ChatService when it detects the user's query
        would benefit from deep analysis.

        Returns:
            Dict with fatigue info (no automatic suggestions)
        """
        try:
            # Get fatigue components from monitor
            fatigue_data = await self.fatigue_monitor.calculate_fatigue()

            # Calculate thresholds but don't suggest automatically
            is_high = fatigue_data["total"] > self.fatigue_threshold
            is_emergency = fatigue_data["total"] > self.emergency_threshold

            result = {
                "fatigue_level": fatigue_data["total"],
                "components": fatigue_data["components"],
                "is_high": is_high,
                "is_emergency": is_emergency,
                "threshold": self.fatigue_threshold,
                "explanation": fatigue_data.get("explanation", ""),
                "last_optimization": await self.state_manager.get_last_optimization_time(),
            }

            # NO automatic suggestions - this violates the spec
            # Suggestions should only come when ChatService detects need

            return result

        except Exception as e:
            logger.error("Failed to check fatigue level", error=str(e))
            # Return safe defaults on error
            return {
                "fatigue_level": 5.0,
                "components": {},
                "is_high": False,
                "is_emergency": False,
                "error": str(e),
            }

    def generate_suggestion_message(self, fatigue_level: float, is_emergency: bool) -> str:
        """
        Generate suggestion message for Dream analysis.

        This is called by ChatService when it detects that the user's query
        would benefit from deep analysis AND fatigue is high.

        Args:
            fatigue_level: Current fatigue score
            is_emergency: Whether it's critical

        Returns:
            Suggestion message for the user
        """
        return self._generate_suggestion(fatigue_level, is_emergency)

    def _generate_suggestion(self, fatigue_level: float, is_emergency: bool) -> str:
        """
        Generate suggestion message based on fatigue level.

        Args:
            fatigue_level: Current fatigue score
            is_emergency: Whether it's critical

        Returns:
            Suggestion message for the user
        """
        if is_emergency:
            return (
                f"I've detected critical activity level (fatigue: {fatigue_level:.1f}/10). "
                "There are many recent changes that need deep analysis. "
                "May I take 5 minutes to perform an exhaustive Deep Search-style analysis?"
            )
        else:
            logger.info(
                "[UNTESTED PATH] Generating message for deep project analysis without query or focus areas"
            )
            return (
                f"I've noticed significant code activity (fatigue: {fatigue_level:.1f}/10). "
                "Would you mind if I take 5 minutes to optimize my memory "
                "and search for patterns or potential issues?"
            )

    async def request_analysis(
        self,
        trigger: DreamTrigger,
        focus_areas: Optional[List[str]] = None,
        user_query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Request permission to start Dream analysis.

        Note: This is called by ChatService when it detects that the user's query
        would benefit from deep analysis. The system NEVER suggests automatically.

        Args:
            trigger: What triggered the request
            focus_areas: Specific areas to analyze (for user requests)
            user_query: Original user query that triggered request
            context: Additional context (including fatigue info from ChatService)

        Returns:
            Request details for user approval
        """
        logger.info("Dream analysis requested", trigger=trigger.value, focus_areas=focus_areas)
        try:
            # Check if already analyzing
            current_state = await self.state_manager.get_current_state()
            if current_state != DreamState.MONITORING:
                return {
                    "status": "already_active",
                    "message": "Deep analysis already in progress",
                    "current_state": current_state.value,
                    "estimated_completion": await self.state_manager.get_estimated_completion(),
                }

            # Generate request message based on trigger
            if trigger == DreamTrigger.USER_REQUEST:
                message = self._generate_user_request_message(user_query, focus_areas)
            else:  # FATIGUE_SUGGESTION from ChatService
                # ChatService detected high fatigue while answering user query
                fatigue_level = context.get("fatigue_level", 0.0) if context else 0.0
                is_emergency = context.get("is_emergency", False) if context else False
                message = self._generate_suggestion(fatigue_level, is_emergency)

            # Create request ID
            request_id = generate_id()

            return {
                "status": "permission_required",
                "request_id": request_id,
                "trigger": trigger.value,
                "message": message,
                "estimated_duration_minutes": self.cycle_duration,
                "focus_areas": focus_areas,
                "benefits": self._get_analysis_benefits(trigger, focus_areas),
                "context_size": self.config.get("model.context_size", 32768),
            }

        except Exception as e:
            logger.error("Failed to request analysis", error=str(e))
            raise ValidationError("Cannot request analysis: {}".format(str(e)))

    def _generate_user_request_message(
        self, user_query: Optional[str], focus_areas: Optional[List[str]]
    ) -> str:
        """Generate message for user-requested analysis."""
        if user_query:
            return (
                f"To perform a complete analysis of '{user_query}' I need to enter "
                "DeepDream mode. It's like activating 'Deep Search' but for your code. "
                f"May I have {self.cycle_duration} minutes to investigate thoroughly?"
            )
        elif focus_areas:
            areas_str = ", ".join(focus_areas[:3])
            return (
                f"To deeply analyze {areas_str} I need to use DeepDream. "
                f"May I take {self.cycle_duration} minutes for an exhaustive analysis?"
            )
        else:
            return (
                "To perform a deep project analysis I need to activate DeepDream. "
                f"May I have {self.cycle_duration} minutes for this?"
            )

    def _get_analysis_benefits(
        self, trigger: DreamTrigger, focus_areas: Optional[List[str]]
    ) -> List[str]:
        """Get expected benefits of the analysis."""
        benefits: List[str] = []

        if trigger == DreamTrigger.USER_REQUEST:
            if focus_areas:
                if "security" in str(focus_areas).lower():
                    benefits.extend(
                        ["Vulnerability detection", "Attack surface analysis", "Input validation"]
                    )
                elif "performance" in str(focus_areas).lower():
                    benefits.extend(
                        ["Bottleneck identification", "N+1 query detection", "Cache opportunities"]
                    )
                else:
                    benefits.extend(
                        [
                            "Exhaustive analysis of requested area",
                            "Hidden problem detection",
                            "Improvement suggestions",
                        ]
                    )
            else:
                benefits.extend(
                    [
                        "Complete project analysis",
                        "Pattern detection",
                        "Technical debt identification",
                    ]
                )
        else:  # FATIGUE_SUGGESTION
            benefits.extend(
                [
                    "Index reorganization for faster searches",
                    "Recent problematic code detection",
                    "Accumulated changes analysis",
                    "Memory optimization",
                ]
            )

        return benefits

    async def start_analysis(
        self,
        request_id: str,
        approved: bool,
        focus_areas: Optional[List[str]] = None,
        priorities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Start Dream analysis after user approval.

        Args:
            request_id: ID from request_analysis
            approved: Whether user approved
            focus_areas: Areas to focus on
            priorities: Priority weights for different analysis types

        Returns:
            Analysis start confirmation or rejection
        """
        if not approved:
            logger.info("Dream analysis rejected by user", request_id=request_id)
            return {
                "status": "rejected",
                "message": "Understood, I'll continue operating normally",
                "request_id": request_id,
            }

        try:
            # Create analysis session
            session_id = generate_id()

            # Transition to DROWSY state
            await self.state_manager.transition_to(DreamState.DROWSY)  # type: ignore

            # Start analysis in background
            asyncio.create_task(
                self._run_analysis_cycle(
                    session_id=session_id, focus_areas=focus_areas, priorities=priorities
                )
            )

            logger.info("Dream analysis started", session_id=session_id, focus_areas=focus_areas)

            return {
                "status": "started",
                "session_id": session_id,
                "message": "Starting deep analysis...",
                "estimated_completion": add_time(
                    utc_now(), minutes=self.cycle_duration
                ).isoformat(),
                "current_state": DreamState.DROWSY,
            }

        except Exception as e:
            logger.error("Failed to start analysis", error=str(e))
            # Reset to monitoring on error
            await self.state_manager.transition_to(DreamState.MONITORING)  # type: ignore
            raise DatabaseError("Cannot start analysis: {}".format(str(e)))

    async def _run_analysis_cycle(
        self,
        session_id: str,
        focus_areas: Optional[List[str]] = None,
        priorities: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Run complete Dream analysis cycle.

        This is the main analysis loop that goes through all states.
        """
        try:
            logger.info("Starting Dream cycle", session_id=session_id)

            # DROWSY -> DREAMING: Initial exploration
            await asyncio.sleep(2)  # Simulate preparation
            await self.state_manager.transition_to(DreamState.DREAMING)  # type: ignore

            # Run initial analysis
            initial_findings = await self.analyzer.explore_codebase(
                focus_areas=focus_areas, context_size=self.config.get("model.context_size", 32768)
            )

            # DREAMING -> REM: Deep analysis
            await self.state_manager.transition_to(DreamState.REM)  # type: ignore

            # Run deep analysis based on initial findings
            deep_insights = await self.analyzer.analyze_deeply(
                initial_findings=initial_findings,
                priorities=priorities or self._get_default_priorities(),
            )

            # REM -> DEEP_SLEEP: Consolidation
            await self.state_manager.transition_to(DreamState.DEEP_SLEEP)  # type: ignore

            # Consolidate findings and update indices
            consolidated = await self._consolidate_findings(
                session_id=session_id, initial=initial_findings, deep=deep_insights
            )

            # DEEP_SLEEP -> WAKING: Prepare results
            await self.state_manager.transition_to(DreamState.WAKING)  # type: ignore

            # Write insights
            await self.insight_writer.write_insights(
                session_id=session_id, insights=consolidated, focus_areas=focus_areas
            )

            # Update fatigue level (reduce by 70%)
            await self.fatigue_monitor.reduce_fatigue(factor=0.3)

            # WAKING -> MONITORING: Complete
            await self.state_manager.transition_to(DreamState.MONITORING)  # type: ignore

            # Record completion
            await self._record_completion(session_id, len(consolidated))

            logger.info(
                "Dream cycle completed", session_id=session_id, insights_count=len(consolidated)
            )

        except Exception as e:
            logger.error("Dream cycle failed", error=str(e), session_id=session_id)
            # Always return to monitoring on error
            await self.state_manager.transition_to(DreamState.MONITORING)  # type: ignore
            # Record error
            await self._record_error(session_id, str(e))

    def _get_default_priorities(self) -> Dict[str, float]:
        """Get default analysis priorities."""
        return {
            "bugs": 0.3,
            "security": 0.3,
            "performance": 0.2,
            "architecture": 0.1,
            "patterns": 0.1,
        }

    async def _consolidate_findings(
        self, session_id: str, initial: Dict[str, Any], deep: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Consolidate findings from different analysis phases.

        Args:
            session_id: Dream session ID
            initial: Initial exploration findings
            deep: Deep analysis insights

        Returns:
            Consolidated list of insights
        """
        logger.debug("Consolidating findings", session_id=session_id)
        # Collect all findings in a structured format
        all_findings: Dict[str, List[Any]] = {
            "bugs": [],
            "security_issues": [],
            "performance_issues": [],
            "architectural_issues": [],
            "patterns": [],
            "recommendations": [],
        }

        # Extract from initial findings (exploration phase)
        if "patterns_detected" in initial:
            all_findings["patterns"].extend(initial["patterns_detected"])

        # Extract architectural issues from overview
        if "overview" in initial and "architectural_issues" in initial["overview"]:
            all_findings["architectural_issues"].extend(initial["overview"]["architectural_issues"])

        # Extract areas of concern (these are high priority)
        if "areas_of_concern" in initial:
            # Areas of concern can be various types, categorize them
            for concern in initial["areas_of_concern"]:
                # Determine type based on content
                if "architectural" in str(concern).lower():
                    all_findings["architectural_issues"].append(concern)
                elif "security" in str(concern).lower():
                    all_findings["security_issues"].append(concern)
                elif "performance" in str(concern).lower():
                    all_findings["performance_issues"].append(concern)
                else:
                    # Default to architectural for structural concerns
                    all_findings["architectural_issues"].append(concern)

        # Extract everything from deep analysis
        for key in all_findings.keys():
            if key in deep:
                all_findings[key].extend(deep[key])

        # Apply deduplication
        all_findings = self._deduplicate_findings(all_findings)

        # Apply prioritization
        all_findings = self._prioritize_findings(all_findings)

        # Flatten to list format for InsightWriter
        insights: List[Dict[str, Any]] = []

        # Convert to flat list while preserving structure
        for category, items in all_findings.items():
            if items and category != "recommendations":
                # Create a wrapper dict for each category
                insights.append(
                    {
                        category: items,
                        "session_id": session_id,
                        "discovered_at": utc_now_iso(),
                    }
                )

        # Add recommendations separately (they have different structure)
        if all_findings.get("recommendations"):
            for rec in all_findings["recommendations"]:
                insights.append(
                    {
                        "type": "recommendation",
                        "description": rec,
                        "session_id": session_id,
                        "discovered_at": utc_now_iso(),
                    }
                )

        return insights

    async def _record_completion(self, session_id: str, insights_count: int) -> None:
        """Record successful completion in database."""
        try:
            await self.db.execute_async(
                """
                UPDATE dream_state 
                SET last_optimization = CURRENT_TIMESTAMP,
                    optimization_count = optimization_count + 1,
                    fatigue_level = fatigue_level * 0.3
                WHERE id = 1
                """,
                (),
            )

            logger.info("Dream completion recorded", session_id=session_id, insights=insights_count)

        except Exception as e:
            logger.error("Failed to record completion", error=str(e))

    async def _record_error(self, session_id: str, error: str) -> None:
        """Record error in database for debugging."""
        try:
            # Could store in a dream_errors table or metrics
            logger.error("Dream session failed", session_id=session_id, error=error)
        except Exception as e:
            logger.error("Failed to record error", error=str(e))

    def _deduplicate_findings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate findings across cycles."""
        for key in results:
            if isinstance(results[key], list) and len(results[key]) > 0:
                # Simple deduplication by converting to unique strings
                seen = set()
                unique = []

                for item in results[key]:
                    # Create unique key for item
                    if isinstance(item, dict):
                        key_str = json.dumps(item, sort_keys=True)
                        if key_str not in seen:
                            seen.add(key_str)
                            unique.append(item)
                    else:
                        if item not in seen:
                            seen.add(item)
                            unique.append(item)

                results[key] = unique

        return results

    def _prioritize_findings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Sort findings by severity/importance."""
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

        def get_severity_value(item: Any, default_severity: str = "MEDIUM") -> int:
            """Safely extract severity value from item (dict or string)."""
            if isinstance(item, dict):
                return severity_order.get(item.get("severity", default_severity), 4)
            else:
                # For non-dict items (strings), return default severity value
                return severity_order.get(default_severity, 4)

        def get_impact_value(item: Any) -> int:
            """Safely extract impact value from item."""
            if isinstance(item, dict):
                impact_str = str(item.get("impact", "")).lower()
                return 0 if "high" in impact_str else 1
            else:
                return 1  # Default to lower priority for non-dict items

        # Sort bugs by severity
        if "bugs" in results:
            results["bugs"].sort(key=lambda x: get_severity_value(x, "LOW"))

        # Sort security issues
        if "security_issues" in results:
            results["security_issues"].sort(key=lambda x: get_severity_value(x, "LOW"))

        # Sort performance issues by impact
        if "performance_issues" in results:
            # First by severity if available, then by impact description
            results["performance_issues"].sort(
                key=lambda x: (
                    get_severity_value(x, "MEDIUM"),
                    get_impact_value(x),
                )
            )

        # Sort architectural issues by severity
        if "architectural_issues" in results:
            results["architectural_issues"].sort(key=lambda x: get_severity_value(x, "MEDIUM"))

        return results

    async def get_recent_insights(
        self, limit: int = 10, insight_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent insights from Dream analysis.

        Args:
            limit: Maximum insights to return
            insight_type: Filter by type

        Returns:
            List of recent insights
        """
        try:
            query = """
                SELECT 
                    id, session_id, insight_type, title, description,
                    confidence, impact, created_at, entities_involved,
                    code_references
                FROM dream_insights
            """
            params: list[Any] = []

            if insight_type:
                query += " WHERE insight_type = ?"
                params.append(insight_type.upper())

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            result = await self.db.execute_async(query, tuple(params), FetchType.ALL)

            # result.data is Optional[Union[Dict, List[Dict]]] - need to check type
            insights: List[Dict[str, Any]] = []

            # When using FetchType.ALL, result.data should be List[Dict]
            if result.data is None:
                return insights

            if not isinstance(result.data, list):
                logger.error("Unexpected data type from query", type=type(result.data))
                return insights

            # Now we know result.data is List[Dict]
            rows: List[Dict[str, Any]] = result.data
            for row in rows:
                # Row is already a dict, work with it directly
                insight = row.copy()  # Make a copy to avoid modifying original
                # Parse JSON fields (handle both compressed and uncompressed data)
                if insight.get("entities_involved"):
                    try:
                        # Check if it's compressed data (bytes) or JSON string
                        if isinstance(insight["entities_involved"], bytes):
                            # Decompress zlib data
                            decompressed = zlib.decompress(insight["entities_involved"]).decode()
                            insight["entities_involved"] = json.loads(decompressed)
                        else:
                            # Regular JSON string
                            insight["entities_involved"] = json.loads(insight["entities_involved"])
                    except (json.JSONDecodeError, zlib.error) as e:
                        logger.warning(
                            "Failed to parse entities_involved",
                            field="entities_involved",
                            error=str(e),
                        )
                        insight["entities_involved"] = []

                if insight.get("code_references"):
                    try:
                        # Check if it's compressed data (bytes) or JSON string
                        if isinstance(insight["code_references"], bytes):
                            # Decompress zlib data
                            decompressed = zlib.decompress(insight["code_references"]).decode()
                            insight["code_references"] = json.loads(decompressed)
                        else:
                            # Regular JSON string
                            insight["code_references"] = json.loads(insight["code_references"])
                    except (json.JSONDecodeError, zlib.error) as e:
                        logger.warning(
                            "Failed to parse code_references", field="code_references", error=str(e)
                        )
                        insight["code_references"] = []
                insights.append(insight)

            return insights

        except Exception as e:
            logger.error("Failed to get insights", error=str(e))
            return []

    async def is_analysis_in_progress(self) -> bool:
        """
        Returns True if a deep analysis is currently in progress.
        """
        current_state = await self.state_manager.get_current_state()
        return current_state != DreamState.MONITORING
