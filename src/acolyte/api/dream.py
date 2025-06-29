"""
Dream system - Real vector base optimization.
It's not a gratuitous anthropomorphism, it's a necessary technical optimizer.

✅ INTEGRATED: This file now uses the real Dream module implemented in /dream/
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
import time

# Core imports
from acolyte.core.logging import logger
from acolyte.core.secure_config import Settings
from acolyte.core.exceptions import (
    ValidationError,
    from_exception,
    internal_error,
)

# Dream module imports
from acolyte.dream.orchestrator import DreamOrchestrator, DreamTrigger

router = APIRouter()

# Initialize Dream orchestrator lazily
_dream_orchestrator = None


def get_dream_orchestrator():
    global _dream_orchestrator
    if _dream_orchestrator is None:
        _dream_orchestrator = DreamOrchestrator()
    return _dream_orchestrator


# Configuration
config = Settings()
logger.info("Dream API initialized", module="dream")


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class OptimizationPriority(BaseModel):
    """Specific optimization priority."""

    type: Literal[
        "bugs",
        "security",
        "performance",
        "architecture",
        "patterns",
    ] = Field(..., description="Type of analysis to prioritize")
    weight: float = Field(..., ge=0.0, le=1.0, description="Relative weight (0.0-1.0)")

    @field_validator("weight")
    @classmethod
    def validate_weight_precision(cls, v):
        # Round to 2 decimal places to avoid precision problems
        return round(v, 2)


class OptimizationRequest(BaseModel):
    """Configuration for Dream analysis request."""

    focus_areas: Optional[List[str]] = Field(
        default=None, description="Specific areas to analyze (e.g. 'auth', 'database', 'api')"
    )
    user_query: Optional[str] = Field(
        default=None, description="User query that triggered the request"
    )
    approved: bool = Field(default=False, description="If the user approved the analysis")
    priorities: List[OptimizationPriority] = Field(
        default=[
            OptimizationPriority(type="bugs", weight=0.3),
            OptimizationPriority(type="security", weight=0.3),
            OptimizationPriority(type="performance", weight=0.2),
            OptimizationPriority(type="architecture", weight=0.1),
            OptimizationPriority(type="patterns", weight=0.1),
        ],
        description="Analysis priorities",
    )

    @field_validator("priorities")
    @classmethod
    def validate_priorities(cls, v):
        if not v:
            raise ValueError("At least one priority required")
        if len(v) > 10:
            raise ValueError("Too many priorities (max 10)")

        # Verify that the weights sum approximately 1.0
        total_weight = sum(p.weight for p in v)
        if not 0.9 <= total_weight <= 1.1:
            raise ValueError(f"Priority weights should sum to ~1.0, got {total_weight}")

        # Verify that there are no duplicates
        types_seen = set()
        for priority in v:
            if priority.type in types_seen:
                raise ValueError(f"Duplicate priority type: {priority.type}")
            types_seen.add(priority.type)

        return v


class InsightFilter(BaseModel):
    """Filters for searching insights."""

    insight_type: Optional[
        Literal["PATTERN", "CONNECTION", "OPTIMIZATION", "ARCHITECTURE", "BUG_RISK"]
    ] = Field(None, description="Filter by insight type")
    min_confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence (0.0-1.0)"
    )


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get("/status")
async def optimization_status() -> Dict[str, Any]:
    """
    Current state of the optimization system.

    The 'fatigue' is a technical metaphor that represents:
    - File instability (based on stability_score)
    - Recent activity (commits_last_30_days)
    - Code volatility (code_volatility_index)
    - Detected architectural changes
    """
    try:
        # Get fatigue level from real monitor
        dream_orchestrator = get_dream_orchestrator()
        fatigue_data = await dream_orchestrator.check_fatigue_level()

        # Get current state info
        state_info = await dream_orchestrator.state_manager.get_state_info()

        # Get configuration
        model_context_size = config.get("model.context_size", 32768)

        # Build status response
        status = {
            "state": state_info["current_state"],
            "fatigue_level": fatigue_data["fatigue_level"],
            "fatigue_components": fatigue_data["components"],
            "explanation": fatigue_data["explanation"],
            "should_suggest_optimization": fatigue_data["should_suggest"],
            "is_emergency": fatigue_data["is_emergency"],
            "can_work": True,  # ACOLYTE can always work
            "optimal_duration_minutes": dream_orchestrator.cycle_duration,
            "model_context_size": model_context_size,
            "window_configuration": {
                "strategy": dream_orchestrator.analyzer.window_manager.strategy,
                "window_size": dream_orchestrator.analyzer.window_manager.new_code_size,
                "preserved_context": dream_orchestrator.analyzer.window_manager.preserved_context_size,
            },
            "last_optimization": {
                "timestamp": fatigue_data.get("last_optimization"),
                "hours_ago": state_info.get("hours_since_optimization"),
            },
            "current_session": {
                "session_id": state_info.get("session_id"),
                "state_duration_seconds": state_info.get("state_duration_seconds"),
                "can_abort": state_info.get("can_abort", True),
                "estimated_completion": state_info.get("estimated_completion"),
            },
        }

        # Add suggestion if fatigue is high
        if fatigue_data["should_suggest"]:
            status["suggestion"] = fatigue_data.get("suggestion", "")

        # Add warning for critical levels
        if fatigue_data["is_emergency"]:
            status["warning"] = (
                "Critical fatigue level detected. Optimization strongly recommended."
            )

        logger.info(
            "Dream status request",
            fatigue_level=fatigue_data["fatigue_level"],
            state=state_info["current_state"],
        )

        return status

    except Exception as e:
        logger.error("Dream status failed", error=str(e), exc_info=True)
        error_response = internal_error(
            message="Failed to get optimization status", context={"error_type": type(e).__name__}
        )
        raise HTTPException(status_code=500, detail=error_response.model_dump())


@router.post("/request")
async def request_optimization(request: OptimizationRequest) -> Dict[str, Any]:
    """
    Request Dream analysis.

    This endpoint is used when:
    1. The user explicitly requests a deep analysis
    2. The system detects high fatigue and suggests optimization

    Always requires explicit user approval.
    """
    try:
        # Determine trigger type
        trigger = DreamTrigger.USER_REQUEST
        if request.user_query is None and request.focus_areas is None:
            # If no specific query or areas, assume it's a fatigue suggestion
            trigger = DreamTrigger.FATIGUE_SUGGESTION

        # Request analysis permission
        dream_orchestrator = get_dream_orchestrator()
        request_data = await dream_orchestrator.request_analysis(
            trigger=trigger,
            focus_areas=request.focus_areas,
            user_query=request.user_query,
        )

        return request_data

    except ValidationError as e:
        logger.warning("Dream request validation failed", validation_message=e.message)
        raise HTTPException(status_code=400, detail=from_exception(e).model_dump())
    except Exception as e:
        logger.error("Dream request failed", error=str(e), exc_info=True)
        error_response = internal_error(
            message="Failed to request optimization", context={"error_type": type(e).__name__}
        )
        raise HTTPException(status_code=500, detail=error_response.model_dump())


@router.post("/optimize")
async def start_optimization(
    request_id: str,
    approved: bool,
    background_tasks: BackgroundTasks,
    priorities: Optional[List[OptimizationPriority]] = None,
) -> Dict[str, Any]:
    """
    Start Dream optimization cycle.

    During the analysis, ACOLYTE:
    1. Explores the code with extended context
    2. Detects bugs, vulnerabilities and performance problems
    3. Identifies patterns and anti-patterns
    4. Analyzes the architecture and dependencies
    5. Generates actionable insights

    Requires request_id from a previous request and user approval.
    """
    start_time = time.time()

    logger.info(
        "Dream optimization start request",
        request_id=request_id,
        approved=approved,
    )

    try:
        # Convert priorities to dict format expected by orchestrator
        priority_dict = None
        if priorities:
            priority_dict = {p.type: p.weight for p in priorities}

        # Start analysis (or reject if not approved)
        dream_orchestrator = get_dream_orchestrator()
        result = await dream_orchestrator.start_analysis(
            request_id=request_id, approved=approved, priorities=priority_dict
        )

        processing_time = int((time.time() - start_time) * 1000)
        result["processing_time_ms"] = processing_time

        return result

    except Exception as e:
        logger.error(
            "Dream optimization failed", error=str(e), request_id=request_id, exc_info=True
        )

        error_response = internal_error(
            message="Failed to start optimization",
            error_id=request_id,
            context={"error_type": type(e).__name__},
        )
        raise HTTPException(status_code=500, detail=error_response.model_dump())


@router.get("/insights")
async def get_optimization_insights(
    limit: int = Query(10, ge=1, le=100, description="Maximum insights to return"),
    insight_type: Optional[str] = Query(None, description="Type of insight to filter"),
    min_confidence: float = Query(0.7, ge=0.0, le=1.0, description="Minimum confidence"),
) -> Dict[str, Any]:
    """
    Get insights discovered during Dream analysis.

    Insight types:
    - PATTERN: Detected code patterns
    - CONNECTION: Connections and relationships between code
    - OPTIMIZATION: Optimization opportunities
    - ARCHITECTURE: Architectural insights
    - BUG_RISK: Bug risks identified
    """
    try:
        logger.info(
            "Dream insights request",
            limit=limit,
            insight_type=insight_type,
            min_confidence=min_confidence,
        )

        # Get insights from orchestrator
        dream_orchestrator = get_dream_orchestrator()
        insights = await dream_orchestrator.get_recent_insights(
            limit=limit, insight_type=insight_type
        )

        # Filter by confidence if needed
        if min_confidence > 0:
            insights = [i for i in insights if i.get("confidence", 0) >= min_confidence]

        # Get current fatigue for context
        fatigue_data = await dream_orchestrator.check_fatigue_level()

        # Build response
        result = {
            "insights": insights,
            "stats": {
                "total_returned": len(insights),
                "types": _count_insights_by_type(insights),
            },
            "filters_applied": {
                "limit": limit,
                "type_filter": insight_type,
                "min_confidence": min_confidence,
            },
            "optimization_context": {
                "current_fatigue": fatigue_data["fatigue_level"],
                "last_analysis": fatigue_data.get("last_optimization"),
            },
        }

        logger.info(
            "Dream insights retrieved",
            returned_count=len(insights),
        )

        return result

    except Exception as e:
        logger.error("Dream insights failed", error=str(e), exc_info=True)
        error_response = internal_error(
            message="Failed to retrieve optimization insights",
            context={"error_type": type(e).__name__},
        )
        raise HTTPException(status_code=500, detail=error_response.model_dump())


@router.post("/abort")
async def abort_analysis(reason: str = "User requested abort") -> Dict[str, Any]:
    """
    Abort Dream analysis in progress.

    Only works if there is an active analysis and is not in the WAKING phase.
    """
    try:
        # Check current state
        dream_orchestrator = get_dream_orchestrator()
        state_info = await dream_orchestrator.state_manager.get_state_info()

        if not state_info.get("can_abort", True):
            raise ValidationError(
                message="Cannot abort during final phase",
                context={"current_state": state_info["current_state"]},
            )

        # Abort analysis
        await dream_orchestrator.state_manager.abort_analysis(reason)

        return {
            "status": "aborted",
            "reason": reason,
            "previous_state": state_info["current_state"],
            "message": "Analysis aborted. Returning to normal operation.",
        }

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=from_exception(e).model_dump())
    except Exception as e:
        logger.error("Dream abort failed", error=str(e), exc_info=True)
        error_response = internal_error(
            message="Failed to abort analysis", context={"error_type": type(e).__name__}
        )
        raise HTTPException(status_code=500, detail=error_response.model_dump())


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _count_insights_by_type(insights: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count insights by type."""
    counts = {}
    for insight in insights:
        insight_type = insight.get("insight_type", "UNKNOWN")
        counts[insight_type] = counts.get(insight_type, 0) + 1
    return counts
