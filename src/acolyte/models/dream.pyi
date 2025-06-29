from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from acolyte.models.base import AcolyteBaseModel, TimestampMixin, IdentifiableMixin

class OptimizationStatus(str, Enum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"

class InsightType(str, Enum):
    PATTERN = "pattern"
    CONNECTION = "connection"
    OPTIMIZATION = "optimization"
    ARCHITECTURE = "architecture"
    BUG_RISK = "bug_risk"

class DreamInsight(AcolyteBaseModel, TimestampMixin, IdentifiableMixin):
    title: str
    description: str
    insight_type: InsightType
    entities_involved: List[str]
    confidence: float
    impact: str
    created_during_session: Optional[str]

class OptimizationMetrics(AcolyteBaseModel):
    time_since_optimization: float
    embedding_fragmentation: float
    query_performance_degradation: float
    new_embeddings_ratio: float

    @property
    def fatigue_level(self) -> float: ...
    @property
    def needs_optimization(self) -> bool: ...

class OptimizationRequest(AcolyteBaseModel):
    duration_minutes: int
    priorities: Dict[str, float]
    force: bool

class OptimizationResult(AcolyteBaseModel, TimestampMixin, IdentifiableMixin):
    session_id: str
    duration_seconds: int
    improvements: Dict[str, str]
    embeddings_reorganized: int
    patterns_found: int
    clusters_created: int
    query_speed_improvement: Optional[float]
    insights: List[Dict[str, Any]]

class DreamState(AcolyteBaseModel):
    status: OptimizationStatus
    metrics: OptimizationMetrics
    last_optimization: Optional[datetime]
    optimization_count: int
    avg_query_time_ms: float
    total_embeddings: int

    def get_recommendation(self) -> str: ...
