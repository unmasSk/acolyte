"""
Models for the optimization system (Dream).
NOT anthropomorphization - it's real technical optimization of embeddings.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import Field
from acolyte.models.base import AcolyteBaseModel, TimestampMixin, IdentifiableMixin


class OptimizationStatus(str, Enum):
    """Optimization process states."""

    IDLE = "idle"  # Not optimizing
    ANALYZING = "analyzing"  # Analyzing need
    OPTIMIZING = "optimizing"  # Actively optimizing
    COMPLETED = "completed"  # Optimization completed


class InsightType(str, Enum):
    """Types of insights discovered during optimization."""

    PATTERN = "pattern"  # Code pattern detected
    CONNECTION = "connection"  # Component relationship
    OPTIMIZATION = "optimization"  # Improvement opportunity
    ARCHITECTURE = "architecture"  # Architectural insight
    BUG_RISK = "bug_risk"  # Potential risk detected


class DreamInsight(AcolyteBaseModel, TimestampMixin, IdentifiableMixin):
    """
    Insight discovered during optimization (Dream mode).
    Patterns and connections found by the analyzer.
    """

    # Identification
    title: str = Field(..., max_length=200, description="Insight title")
    description: str = Field(..., description="Detailed explanation")
    insight_type: InsightType = Field(..., description="Type of insight")

    # Context
    entities_involved: List[str] = Field(
        default_factory=list, description="Related files/functions"
    )

    # Quality
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the insight (0.0-1.0)"
    )
    impact: str = Field(..., description="LOW/MEDIUM/HIGH - Estimated impact")

    # Metadata
    created_during_session: Optional[str] = Field(
        None, description="Optimization session ID that discovered it"
    )


class OptimizationMetrics(AcolyteBaseModel):
    """
    Technical metrics indicating need for optimization.
    'Fatigue' is a metaphor for these technical values.
    """

    # Fatigue components (0-10 each)
    time_since_optimization: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Time since last optimization"
    )
    embedding_fragmentation: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Related embeddings dispersion"
    )
    query_performance_degradation: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Search time degradation"
    )
    new_embeddings_ratio: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Ratio of unorganized embeddings"
    )

    @property
    def fatigue_level(self) -> float:
        """Calculates total 'fatigue' level (optimization need)."""
        components = [
            self.time_since_optimization * 0.2,
            self.embedding_fragmentation * 0.3,
            self.query_performance_degradation * 0.3,
            self.new_embeddings_ratio * 0.2,
        ]
        return sum(components)

    @property
    def needs_optimization(self) -> bool:
        """Determines if optimization is recommended."""
        return self.fatigue_level > 7.0


class OptimizationRequest(AcolyteBaseModel):
    """Request to start optimization."""

    duration_minutes: int = Field(default=20, ge=5, le=60, description="Optimization duration")
    priorities: Dict[str, float] = Field(
        default_factory=lambda: {
            "embedding_reorganization": 0.4,
            "pattern_detection": 0.3,
            "context_clustering": 0.3,
        },
        description="What to optimize and with what weight",
    )
    force: bool = Field(default=False, description="Force even if fatigue is low")


class OptimizationResult(AcolyteBaseModel, TimestampMixin, IdentifiableMixin):
    """Result of an optimization process."""

    # Identification
    session_id: str = Field(..., description="Optimization session ID")
    duration_seconds: int = Field(..., description="Actual duration")

    # Improvements achieved
    improvements: Dict[str, str] = Field(default_factory=dict, description="Improvements achieved")

    # Statistics
    embeddings_reorganized: int = Field(default=0, description="Embeddings reorganized")
    patterns_found: int = Field(default=0, description="Patterns detected")
    clusters_created: int = Field(default=0, description="Clusters created")

    # Performance
    query_speed_improvement: Optional[float] = Field(
        default=None, description="Search speed improvement (%)"
    )

    # Generated insights
    insights: List[Dict[str, Any]] = Field(
        default_factory=list, description="Patterns and suggestions found"
    )


class DreamState(AcolyteBaseModel):
    """
    Current state of the optimization system.
    Continuous metrics tracking.
    """

    status: OptimizationStatus = Field(default=OptimizationStatus.IDLE, description="Current state")

    # Current metrics
    metrics: OptimizationMetrics = Field(
        default_factory=lambda: OptimizationMetrics(), description="Current technical metrics"
    )

    # History
    last_optimization: Optional[datetime] = Field(
        default=None, description="Last optimization performed"
    )
    optimization_count: int = Field(default=0, description="Total optimizations performed")

    # Current performance
    avg_query_time_ms: float = Field(default=0.0, description="Average search time")
    total_embeddings: int = Field(default=0, description="Total embeddings in system")

    def get_recommendation(self) -> str:
        """Generates recommendation based on metrics."""
        fatigue = self.metrics.fatigue_level

        if fatigue < 3:
            return "System optimized, no action required"
        elif fatigue < 5:
            return "Good performance, optimization not urgent"
        elif fatigue < 7:
            return "Consider optimization soon for better performance"
        elif fatigue < 9:
            return "Optimization recommended - notable degradation"
        else:
            return "Urgent optimization - significantly degraded performance"
