"""
System of local observability.

For local debugging and metrics, without external telemetry.
Only for internal system monitoring of the mono-user system.
"""

from contextlib import contextmanager
from typing import Dict, Any, Optional, Iterator

from acolyte.core.logging import AsyncLogger

class LocalTracer:
    """
    Simple local tracing system.

    CLARIFICATION LocalTracer vs MetricsCollector:
    =============================================
    LocalTracer: For specific operations with context
    - Individual spans with duration and attributes
    - Useful for debugging "why X operation is slow"
    - Generates detailed logs with span_id for correlation

    MetricsCollector: For aggregated system statistics
    - Numeric counters and gauges
    - Useful for dashboards and general monitoring
    - No specific operation context

    BOTH ARE NECESSARY:
    - LocalTracer = Deep performance debugging
    - MetricsCollector = General system overview

    COMPLEMENTARY USE EXAMPLE:
    - MetricsCollector.increment("database_queries")
    - LocalTracer.span("database_query", {"table": "conversations", "query_type": "SELECT"})

    For local debugging and metrics, without external telemetry.
    """

    service_name: str
    logger: AsyncLogger

    def __init__(self, service_name: str = "acolyte") -> None: ...
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[None]: ...

class MetricsCollector:
    """
    Local metrics collector.

    Only for internal system monitoring, without export.
    """

    metrics: Dict[str, float]
    logger: AsyncLogger

    def __init__(self) -> None: ...
    def increment(self, name: str, value: float = 1.0) -> None: ...
    def gauge(self, name: str, value: float) -> None: ...
    def record(self, name: str, value: float) -> None: ...
    def get_metrics(self) -> Dict[str, float]: ...

# Global instances
tracer: LocalTracer
metrics: MetricsCollector
