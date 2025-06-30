"""
Relations Manager - Graph relationship management.

Handles specific operations on relationships, metrics and
optimized queries of the neural graph.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from acolyte.core.database import get_db_manager, FetchType
from acolyte.core.logging import logger
from acolyte.core.exceptions import DatabaseError


class RelationsManager:
    """
    Manages neural graph relationships in SQLite.

    Advanced operations on edges and metrics.
    """

    def __init__(self):
        """Initialize manager with database connection."""
        self.db = get_db_manager()

    async def get_file_dependencies(
        self, file_path: str, relation_types: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Get all dependencies for a file.

        Args:
            file_path: File path
            relation_types: Relationship types to include (default: all)

        Returns:
            Dict with dependency lists by type
        """
        if not relation_types:
            relation_types = ["IMPORTS", "CALLS", "EXTENDS", "IMPLEMENTS", "USES"]

        try:
            placeholders = ",".join("?" * len(relation_types))
            query = f"""
            SELECT 
                e.relation_type,
                n_target.path as target_path
            FROM code_graph_nodes n_source
            JOIN code_graph_edges e ON n_source.id = e.source_id
            JOIN code_graph_nodes n_target ON e.target_id = n_target.id
            WHERE n_source.path = ?
            AND e.relation_type IN ({placeholders})
            ORDER BY e.relation_type, e.strength DESC
            """
            result = await self.db.execute_async(query, [file_path, *relation_types], FetchType.ALL)
            results = result.data or []

            # Each row is always a dict[str, Any] per execute_async contract, ignore false linter positives.
            dependencies = {}
            for row in results:
                rel_type = row["relation_type"]  # type: ignore[index]
                if rel_type not in dependencies:
                    dependencies[rel_type] = []
                dependencies[rel_type].append(row["target_path"])  # type: ignore[index]

            return dependencies

        except Exception as e:
            logger.error("Error getting dependencies", error=str(e))
            raise DatabaseError(f"Could not get dependencies: {e}") from e

    async def find_co_modified(
        self, file_path: str, min_strength: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find files that are modified together with this one.

        Args:
            file_path: Reference file
            min_strength: Minimum co-modification strength

        Returns:
            List of (file, strength)
        """
        try:
            result = await self.db.execute_async(
                """
                SELECT 
                    n_other.path,
                    e.strength
                FROM code_graph_nodes n
                JOIN code_graph_edges e ON (
                    (e.source_id = n.id OR e.target_id = n.id)
                    AND e.relation_type = 'MODIFIES_TOGETHER'
                )
                JOIN code_graph_nodes n_other ON (
                    CASE 
                        WHEN e.source_id = n.id THEN e.target_id = n_other.id
                        ELSE e.source_id = n_other.id
                    END
                )
                WHERE n.path = ?
                AND e.strength >= ?
                ORDER BY e.strength DESC
                """,
                (file_path, min_strength),
                FetchType.ALL,
            )
            results = result.data or []

            # Each row is always a dict[str, Any] per execute_async contract, ignore false linter positives.
            return [(row["path"], row["strength"]) for row in results]  # type: ignore[index,assignment]

        except Exception as e:
            logger.error("Error finding co-modified files", error=str(e))
            raise DatabaseError(f"Could not find co-modified files: {e}")

    async def decay_old_relations(
        self, days_threshold: int = 30, decay_factor: float = 0.05
    ) -> int:
        """
        Reduce strength of old relationships.

        Args:
            days_threshold: Days without reinforcement to apply decay
            decay_factor: How much to reduce (default 0.05)

        Returns:
            Number of affected relationships
        """
        threshold_date = (datetime.now() - timedelta(days=days_threshold)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        try:
            result = await self.db.execute_async(
                """
                UPDATE code_graph_edges
                SET strength = MAX(0.1, strength - ?)
                WHERE last_reinforced < ?
                AND relation_type IN ('MODIFIES_TOGETHER', 'BUG_PATTERN')
                """,
                (decay_factor, threshold_date),
            )

            affected = getattr(result, "rowcount", 0) or 0
            logger.info("Decay applied to relations", affected=affected)

            return int(affected)

        except Exception as e:
            logger.error("Error applying decay", error=str(e))
            raise DatabaseError(f"Could not apply decay: {e}")

    async def update_metrics(self) -> None:
        """
        Update global graph metrics.

        Calculates total nodes, edges, average connectivity.
        """
        try:
            # Total nodes
            nodes_result = await self.db.execute_async(
                "SELECT COUNT(*) as count FROM code_graph_nodes", (), FetchType.ONE
            )
            nodes_data = nodes_result.data
            total_nodes = nodes_data["count"] if nodes_data else 0  # type: ignore[index]

            # Total edges
            edges_result = await self.db.execute_async(
                "SELECT COUNT(*) as count FROM code_graph_edges", (), FetchType.ONE
            )
            edges_data = edges_result.data
            total_edges = edges_data["count"] if edges_data else 0  # type: ignore[index]

            # Average connectivity
            avg_connectivity = 0.0
            if total_nodes > 0:
                connectivity_result = await self.db.execute_async(
                    """
                    SELECT AVG(connection_count) as avg_conn
                    FROM (
                        SELECT n.id, COUNT(*) as connection_count
                        FROM code_graph_nodes n
                        LEFT JOIN code_graph_edges e ON (
                            e.source_id = n.id OR e.target_id = n.id
                        )
                        GROUP BY n.id
                    )
                    """,
                    (),
                    FetchType.ONE,
                )
                connectivity_data = connectivity_result.data
                avg_connectivity = connectivity_data["avg_conn"] if connectivity_data else 0.0  # type: ignore[index]

            # Update metrics table
            await self.db.execute_async(
                """
                INSERT INTO code_graph_metrics 
                (id, total_nodes, total_edges, avg_connectivity, last_analysis)
                VALUES (1, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    total_nodes = excluded.total_nodes,
                    total_edges = excluded.total_edges,
                    avg_connectivity = excluded.avg_connectivity,
                    last_analysis = excluded.last_analysis,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (total_nodes, total_edges, avg_connectivity),
            )

            logger.info("Metrics updated", total_nodes=total_nodes, total_edges=total_edges)

        except Exception as e:
            logger.error("Error updating metrics", error=str(e))
            raise DatabaseError(f"Could not update metrics: {e}")

    async def find_strongest_clusters(
        self, min_nodes: int = 3, min_avg_strength: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Find strongly connected clusters.

        Args:
            min_nodes: Minimum nodes in cluster
            min_avg_strength: Minimum average strength

        Returns:
            List of clusters with their nodes
        """
        # Simplified implementation: find highly connected files
        try:
            result = await self.db.execute_async(
                """
                WITH node_connections AS (
                    SELECT 
                        n.id,
                        n.path,
                        n.name,
                        COUNT(DISTINCT e.source_id) + COUNT(DISTINCT e.target_id) as total_connections,
                        AVG(e.strength) as avg_strength
                    FROM code_graph_nodes n
                    JOIN code_graph_edges e ON (n.id = e.source_id OR n.id = e.target_id)
                    GROUP BY n.id
                    HAVING total_connections >= ?
                    AND avg_strength >= ?
                )
                SELECT * FROM node_connections
                ORDER BY total_connections DESC, avg_strength DESC
                LIMIT 10
                """,
                (min_nodes, min_avg_strength),
                FetchType.ALL,
            )
            results = result.data or []

            clusters = []
            for node in results:
                # For each central node, find its neighbors
                neighbors_result = await self.db.execute_async(
                    """
                    SELECT DISTINCT 
                        n.path,
                        e.strength
                    FROM code_graph_edges e
                    JOIN code_graph_nodes n ON (
                        CASE
                            WHEN e.source_id = ? THEN e.target_id = n.id
                            ELSE e.source_id = n.id
                        END
                    )
                    WHERE e.strength >= ?
                    """,
                    (node["id"], min_avg_strength),  # type: ignore[index]
                    FetchType.ALL,
                )
                neighbors = neighbors_result.data or []

                # Each node and neighbor is dict[str, Any], ignore false linter positives.
                clusters.append(
                    {
                        "central_node": node["path"],  # type: ignore[index]
                        "total_connections": node["total_connections"],  # type: ignore[index]
                        "avg_strength": node["avg_strength"],  # type: ignore[index]
                        "members": [n["path"] for n in neighbors],  # type: ignore[index]
                    }
                )

            return clusters

        except Exception as e:
            logger.error("Error finding clusters", error=str(e))
            raise DatabaseError(f"Could not find clusters: {e}")
