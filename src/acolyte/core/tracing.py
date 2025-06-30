"""
System of local observability.

For local debugging and metrics, without external telemetry.
Only for internal system monitoring of the mono-user system.
"""

import time
from contextlib import contextmanager
from typing import Dict, Any, Optional, Iterator

from acolyte.core.logging import AsyncLogger
from acolyte.core.id_generator import generate_id


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

    def __init__(self, service_name: str = "acolyte") -> None:
        self.service_name = service_name
        self.logger = AsyncLogger("tracing")

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[None]:
        """
        Creates a span to measure operation.

        Usage:
        ```
        with tracer.span("database_query", {"query": sql}):
            result = await db.execute(sql)
        ```
        """
        span_id = generate_id()
        start = time.perf_counter()

        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.logger.debug(
                f"Span completed: {name}",
                span_id=span_id,
                duration_ms=duration * 1000,
                **(attributes or {}),
            )


class MetricsCollector:
    """
    Local metrics collector.

    Only for internal system monitoring, without export.
    """

    def __init__(self) -> None:
        self.metrics: Dict[str, float] = {}
        self.logger = AsyncLogger("metrics")

    def increment(self, name: str, value: float = 1.0) -> None:
        """Increments counter."""
        self.metrics[name] = self.metrics.get(name, 0) + value

    def gauge(self, name: str, value: float) -> None:
        """Sets current value."""
        self.metrics[name] = value

    def record(self, name: str, value: float) -> None:
        """Records a measurement (alias of gauge for compatibility)."""
        self.gauge(name, value)

    def get_metrics(self) -> Dict[str, float]:
        """Gets all metrics."""
        return self.metrics.copy()


# Global instances
tracer = LocalTracer()
metrics = MetricsCollector()  # Global metrics without namespace
