from typing import Dict, List, Tuple, TypedDict, Optional
from acolyte.core.tracing import MetricsCollector

class OperationStats(TypedDict):
    count: int
    avg: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float

class GlobalStats(TypedDict):
    total_operations: int
    avg: float
    p95: float
    meets_sla: bool

class PerformanceStatsDict(TypedDict, total=False):
    _global: GlobalStats
    encode: OperationStats
    encode_batch: OperationStats
    model_load: OperationStats
    cache_lookup: OperationStats
    tokenization: OperationStats

class SearchQualityReport(TypedDict):
    mrr: float
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    total_clicks: int
    unique_queries: int
    queries_with_feedback: int
    avg_search_time_ms: float
    p95_search_time_ms: float

class CacheMetrics(TypedDict):
    hit_rate: float
    hits: int
    misses: int

class OperationsMetrics(TypedDict):
    total_embeddings: int
    total_errors: int
    error_rate: float

class HealthMetrics(TypedDict):
    meets_sla: bool
    quality_threshold: bool

class EmbeddingsMetricsSummary(TypedDict):
    performance: PerformanceStatsDict
    quality: SearchQualityReport
    cache: CacheMetrics
    operations: OperationsMetrics
    health: HealthMetrics

class PerformanceMetrics:
    def __init__(self) -> None: ...
    def start_operation(self, operation: str) -> str: ...
    def end_operation(self, op_id: str) -> float: ...
    def record_latency(self, operation: str, latency_ms: float) -> None: ...
    def get_p95(self, operation: Optional[str] = ...) -> float: ...
    def get_stats(self) -> PerformanceStatsDict: ...
    def check_sla_compliance(self) -> Tuple[bool, Dict[str, bool]]: ...

class SearchQualityMetrics:
    def __init__(self) -> None: ...
    def record_search_results(
        self, query: str, results: List[str], search_time_ms: Optional[float] = ...
    ) -> None: ...
    def record_click(self, query: str, clicked_result: str) -> None: ...
    def record_relevance_feedback(self, query: str, relevance_list: List[bool]) -> None: ...
    def calculate_mrr(self) -> float: ...
    def calculate_precision_at_k(self, k: int) -> float: ...
    def calculate_recall_at_k(self, k: int) -> float: ...
    def get_quality_report(self) -> SearchQualityReport: ...

class EmbeddingsMetrics:
    collector: MetricsCollector
    performance: PerformanceMetrics
    search_quality: SearchQualityMetrics
    def __init__(self) -> None: ...
    def record_operation(self, operation: str, latency_ms: float, success: bool = ...) -> None: ...
    def record_cache_hit(self) -> None: ...
    def record_cache_miss(self) -> None: ...
    def get_cache_hit_rate(self) -> float: ...
    def get_p95_latency(self) -> float: ...
    def get_summary(self) -> EmbeddingsMetricsSummary: ...
    def log_summary(self) -> None: ...
