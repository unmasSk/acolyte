from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
from acolyte.dream import DreamOrchestrator

router: APIRouter

def get_dream_orchestrator() -> DreamOrchestrator: ...

class OptimizationPriority(BaseModel):
    type: Literal[
        "bugs",
        "security",
        "performance",
        "architecture",
        "patterns",
    ]
    weight: float = Field(..., ge=0.0, le=1.0)

    @field_validator("weight")
    @classmethod
    def validate_weight_precision(cls, v: float) -> float: ...

class OptimizationRequest(BaseModel):
    focus_areas: Optional[List[str]] = ...
    user_query: Optional[str] = ...
    approved: bool = ...
    priorities: List[OptimizationPriority] = ...

    @field_validator("priorities")
    @classmethod
    def validate_priorities(cls, v: List[OptimizationPriority]) -> List[OptimizationPriority]: ...

class InsightFilter(BaseModel):
    insight_type: Optional[
        Literal["PATTERN", "CONNECTION", "OPTIMIZATION", "ARCHITECTURE", "BUG_RISK"]
    ] = ...
    min_confidence: float = ...

async def optimization_status() -> Dict[str, Any]: ...
async def request_optimization(request: OptimizationRequest) -> Dict[str, Any]: ...
async def start_optimization(
    request_id: str,
    approved: bool,
    background_tasks: BackgroundTasks,
    priorities: Optional[List[OptimizationPriority]] = ...,
) -> Dict[str, Any]: ...
async def get_optimization_insights(
    limit: int = Query(...),
    insight_type: Optional[str] = Query(...),
    min_confidence: float = Query(...),
) -> Dict[str, Any]: ...
async def abort_analysis(reason: str = ...) -> Dict[str, Any]: ...
def _count_insights_by_type(insights: List[Dict[str, Any]]) -> Dict[str, int]: ...
