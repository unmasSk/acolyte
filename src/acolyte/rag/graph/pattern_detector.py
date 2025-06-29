"""
Pattern Detector - Pattern detection in the graph.

Finds problematic patterns, refactoring opportunities
and unusual connections in the code.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta

from acolyte.core.database import get_db_manager, FetchType
from acolyte.core.logging import logger
from acolyte.core.exceptions import DatabaseError


class PatternDetector:
    """
    Detects patterns in the neural graph.

    Used mainly by Dream for deep analysis.
    """

    def __init__(self):
        """Initialize detector with database connection."""
        self.db = get_db_manager()

    async def detect_patterns(self, pattern_type: str) -> List[Dict[str, Any]]:
        """
        Detect a specific type of pattern.

        Args:
            pattern_type: tight_coupling, change_hotspots, architectural_boundaries

        Returns:
            List of detected patterns
        """
        if pattern_type == "tight_coupling":
            return await self._detect_tight_coupling()
        elif pattern_type == "change_hotspots":
            return await self._detect_change_hotspots()
        elif pattern_type == "architectural_boundaries":
            return await self._detect_architectural_violations()
        else:
            logger.warning("Unrecognized pattern type", pattern_type=pattern_type)
            return []

    async def _detect_tight_coupling(
        self, min_strength: float = 0.8, min_connections: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Detect tightly coupled files.

        Args:
            min_strength: Minimum connection strength
            min_connections: Minimum number of strong connections

        Returns:
            List of coupled groups
        """
        try:
            # Find pairs of files with multiple strong relationship types
            result = await self.db.execute_async(
                """
                WITH strong_pairs AS (
                    SELECT 
                        n1.path as source_path,
                        n2.path as target_path,
                        COUNT(DISTINCT e.relation_type) as relation_count,
                        AVG(e.strength) as avg_strength,
                        GROUP_CONCAT(e.relation_type) as relations
                    FROM code_graph_edges e
                    JOIN code_graph_nodes n1 ON e.source_id = n1.id
                    JOIN code_graph_nodes n2 ON e.target_id = n2.id
                    WHERE e.strength >= ?
                    GROUP BY n1.id, n2.id
                    HAVING relation_count >= ?
                )
                SELECT * FROM strong_pairs
                ORDER BY relation_count DESC, avg_strength DESC
                """,
                (min_strength, min_connections),
                FetchType.ALL,
            )
            results = result.data or []

            # Each row is always a dict[str, Any] per execute_async contract, ignore false linter positives.
            patterns = []
            for row in results:
                patterns.append(
                    {
                        "type": "TIGHT_COUPLING",
                        "severity": "HIGH" if row["avg_strength"] > 0.9 else "MEDIUM",  # type: ignore[index]
                        "files": [row["source_path"], row["target_path"]],  # type: ignore[index]
                        "coupling_score": row["avg_strength"],  # type: ignore[index]
                        "relation_types": row["relations"].split(","),  # type: ignore[index]
                        "recommendation": f"Consider introducing an interface between {row['source_path']} and {row['target_path']}",  # type: ignore[index]
                    }
                )

            return patterns

        except Exception as e:
            logger.error("Error detecting tight coupling", error=str(e))
            raise DatabaseError(f"Could not detect tight coupling: {e}")

    async def _detect_change_hotspots(
        self, days_window: int = 30, min_changes: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Detect files that change frequently.

        Args:
            days_window: Time window to analyze
            min_changes: Minimum changes to consider hotspot

        Returns:
            List of hotspots
        """
        try:
            # Count recent modifications per file
            window_start = datetime.now() - timedelta(days=days_window)

            result = await self.db.execute_async(
                """
                SELECT 
                    n.path,
                    n.name,
                    COUNT(DISTINCT e.metadata) as modification_count,
                    COUNT(DISTINCT other.id) as files_modified_with
                FROM code_graph_nodes n
                JOIN code_graph_edges e ON (
                    n.id = e.source_id OR n.id = e.target_id
                )
                LEFT JOIN code_graph_nodes other ON (
                    (e.source_id = n.id  AND e.target_id = other.id) OR
                    (e.target_id = n.id  AND e.source_id = other.id)
                )
                WHERE e.relation_type = 'MODIFIES_TOGETHER'
                AND e.last_reinforced >= ?
                GROUP BY n.id
                HAVING modification_count >= ?
                ORDER BY modification_count DESC
                """,
                (window_start, min_changes),
                FetchType.ALL,
            )
            results = result.data or []

            # Each row is always a dict[str, Any] per execute_async contract, ignore false linter positives.
            patterns = []
            for row in results:
                severity = "CRITICAL" if row["modification_count"] > min_changes * 2 else "HIGH"  # type: ignore[index]

                patterns.append(
                    {
                        "type": "CHANGE_HOTSPOT",
                        "severity": severity,
                        "file": row["path"],  # type: ignore[index]
                        "change_frequency": row["modification_count"],  # type: ignore[index]
                        "co_modified_files": row["files_modified_with"],  # type: ignore[index]
                        "recommendation": f"{row['path']} changes very frequently. Consider breaking it into smaller, more stable modules.",  # type: ignore[index]
                    }
                )

            return patterns

        except Exception as e:
            logger.error("Error detecting hotspots", error=str(e))
            raise DatabaseError(f"Could not detect hotspots: {e}")

    async def _detect_architectural_violations(self) -> List[Dict[str, Any]]:
        """
        Detect architectural boundary violations.

        Returns:
            List of detected violations
        """
        try:
            # Detect cross-layer imports
            # Assume convention: src/domain should not import from src/api
            result = await self.db.execute_async(
                """
                SELECT 
                    n1.path as source,
                    n2.path as target,
                    e.relation_type
                FROM code_graph_edges e
                JOIN code_graph_nodes n1 ON e.source_id = n1.id
                JOIN code_graph_nodes n2 ON e.target_id = n2.id
                WHERE e.relation_type = 'IMPORTS'
                AND (
                    -- Domain importing from API
                    (n1.path LIKE '%/domain/%' AND n2.path LIKE '%/api/%')
                    OR
                    -- Core importing from upper modules
                    (n1.path LIKE '%/core/%' AND (
                        n2.path LIKE '%/api/%' OR 
                        n2.path LIKE '%/services/%' OR
                        n2.path LIKE '%/rag/%'
                    ))
                    OR
                    -- Models importing from services
                    (n1.path LIKE '%/models/%' AND n2.path LIKE '%/services/%')
                )
                """,
                (),
                FetchType.ALL,
            )
            results = result.data or []

            # Each row is always a dict[str, Any] per execute_async contract, ignore false linter positives.
            patterns = []
            for row in results:
                patterns.append(
                    {
                        "type": "ARCHITECTURAL_VIOLATION",
                        "severity": "HIGH",
                        "source": row["source"],  # type: ignore[index]
                        "target": row["target"],  # type: ignore[index]
                        "violation": "Cross-layer dependency",
                        "recommendation": f"{row['source']} should not import from {row['target']}. Consider inverting the dependency.",  # type: ignore[index]
                    }
                )

            return patterns

        except Exception as e:
            logger.error("Error detecting architectural violations", error=str(e))
            raise DatabaseError(f"Could not detect violations: {e}")

    async def find_bug_patterns(self, min_occurrences: int = 2) -> List[Dict[str, Any]]:
        """
        Find recurring bug patterns.

        Args:
            min_occurrences: Minimum pattern occurrences

        Returns:
            List of bug patterns
        """
        try:
            # Search for BUG_PATTERN type relationships
            result = await self.db.execute_async(
                """
                SELECT 
                    e.metadata,
                    COUNT(*) as occurrences,
                    GROUP_CONCAT(n1.path || ' -> ' || n2.path) as instances
                FROM code_graph_edges e
                JOIN code_graph_nodes n1 ON e.source_id = n1.id
                JOIN code_graph_nodes n2 ON e.target_id = n2.id
                WHERE e.relation_type = 'BUG_PATTERN'
                GROUP BY e.metadata
                HAVING occurrences >= ?
                ORDER BY occurrences DESC
                """,
                (min_occurrences,),
                FetchType.ALL,
            )
            results = result.data or []

            # Each row is always a dict[str, Any] per execute_async contract, ignore false linter positives.
            patterns = []
            for row in results:
                patterns.append(
                    {
                        "type": "RECURRING_BUG",
                        "severity": "HIGH" if row["occurrences"] > 5 else "MEDIUM",  # type: ignore[index]
                        "pattern": row["metadata"],  # type: ignore[index]
                        "occurrences": row["occurrences"],  # type: ignore[index]
                        "instances": row["instances"].split(","),  # type: ignore[index]
                        "recommendation": "This bug pattern appears multiple times. Consider adding a linting rule or test.",
                    }
                )

            return patterns

        except Exception as e:
            logger.error("Error finding bug patterns", error=str(e))
            raise DatabaseError(f"Could not find bug patterns: {e}")

    async def analyze_for_dream(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Complete analysis for Dream mode.

        Runs all detectors and groups results.

        Returns:
            Dictionary with all found patterns
        """
        logger.info("[UNTESTED PATH] Starting complete pattern analysis for Dream")
        logger.warning("[UNTESTED BRANCH] analyze_for_dream method called")

        analysis = {
            "tight_coupling": await self.detect_patterns("tight_coupling"),
            "change_hotspots": await self.detect_patterns("change_hotspots"),
            "architectural_violations": await self.detect_patterns("architectural_boundaries"),
            "bug_patterns": await self.find_bug_patterns(),
        }

        # Count totals
        total_issues = sum(len(patterns) for patterns in analysis.values())
        critical_count = sum(
            1
            for patterns in analysis.values()
            for p in patterns
            if p.get("severity") in ["CRITICAL", "HIGH"]
        )

        logger.info("Analysis completed", total_issues=total_issues, critical_count=critical_count)

        return analysis
