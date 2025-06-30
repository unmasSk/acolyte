"""
Neural Graph - Code relationship graph.

Manages nodes (files, functions, classes) and their connections
(imports, calls, extends) stored in SQLite.
"""

from typing import Dict, Any, List, Optional
import json

from acolyte.core.database import get_db_manager, FetchType
from acolyte.core.logging import logger
from acolyte.core.tracing import MetricsCollector
from acolyte.core.exceptions import DatabaseError, NotFoundError
from acolyte.core.id_generator import generate_id


class NeuralGraph:
    """
    Main neural graph class.

    Maintains structural relationships between code in SQLite.
    """

    def __init__(self):
        """Initialize graph with database connection."""
        self.db = get_db_manager()
        self.metrics = MetricsCollector()

    async def add_node(
        self, node_type: str, path: str, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add or update a node in the graph.

        Args:
            node_type: FILE, FUNCTION, CLASS, MODULE
            path: Full path or path::function_name
            name: Element name
            metadata: Additional metadata (optional)

        Returns:
            Node ID
        """
        node_id = generate_id()
        metadata_json = json.dumps(metadata or {})

        try:
            await self.db.execute_async(
                """
                INSERT INTO code_graph_nodes (id, node_type, path, name, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(node_type, path) DO UPDATE SET
                    name = excluded.name,
                    last_seen = CURRENT_TIMESTAMP,
                    metadata = excluded.metadata
                """,
                (node_id, node_type, path, name, metadata_json),
            )

            # Get real ID if it was an update
            result = await self.db.execute_async(
                "SELECT id FROM code_graph_nodes WHERE node_type = ? AND path = ?",
                (node_type, path),
                FetchType.ONE,
            )
            data: Optional[Dict[str, Any]] = result.data  # type: ignore
            self.metrics.increment("graph.nodes.added")
            return data["id"] if data else node_id

        except Exception as e:
            logger.error("Error adding node", error=str(e))
            raise DatabaseError(f"Could not add node: {e}")

    async def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        discovered_by: str = "STATIC_ANALYSIS",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add or strengthen a relationship between nodes.

        Args:
            source: Source node ID or path
            target: Target node ID or path
            relation: IMPORTS, CALLS, EXTENDS, IMPLEMENTS, USES, MODIFIES_TOGETHER, BUG_PATTERN
            discovered_by: GIT_ACTIVITY, DREAM_ANALYSIS, USER_ACTIVITY, STATIC_ANALYSIS
            metadata: Additional info (line, commit, etc.)
        """
        # Resolve IDs if they are paths
        source_id = await self._resolve_node_id(source)
        target_id = await self._resolve_node_id(target)

        if not source_id or not target_id:
            logger.warning("[UNTESTED PATH] Node not found in strengthen_edge")
            raise NotFoundError(f"Node not found: {source} or {target}")

        metadata_json = json.dumps(metadata or {})

        try:
            await self.db.execute_async(
                """
                INSERT INTO code_graph_edges 
                (source_id, target_id, relation_type, strength, discovered_by, metadata)
                VALUES (?, ?, ?, 0.5, ?, ?)
                ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                    strength = MIN(1.0, strength + 0.1),
                    last_reinforced = CURRENT_TIMESTAMP,
                    metadata = excluded.metadata
                """,
                (source_id, target_id, relation, discovered_by, metadata_json),
            )

            self.metrics.increment("graph.edges.added")

        except Exception as e:
            logger.error("Error adding edge", error=str(e))
            raise DatabaseError(f"Could not add edge: {e}")

    async def strengthen_edge(
        self, source: str, target: str, relation: str, delta: float = 0.1
    ) -> None:
        """
        Strengthen an existing connection.

        Args:
            source: Source ID or path
            target: Target ID or path
            relation: Relationship type
            delta: Strength increment (default 0.1)
        """
        source_id = await self._resolve_node_id(source)
        target_id = await self._resolve_node_id(target)

        if not source_id or not target_id:
            raise NotFoundError(f"Node not found: {source} or {target}")

        try:
            await self.db.execute_async(
                """
                UPDATE code_graph_edges 
                SET strength = MIN(1.0, strength + ?),
                    last_reinforced = CURRENT_TIMESTAMP
                WHERE source_id = ? AND target_id = ? AND relation_type = ?
                """,
                (delta, source_id, target_id, relation),
            )

            self.metrics.increment("graph.edges.strengthened")

        except Exception as e:
            logger.error("Error strengthening edge", error=str(e))
            raise DatabaseError(f"Could not strengthen edge: {e}")

    async def find_related(
        self, node: str, max_distance: int = 2, min_strength: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find related nodes up to a certain distance.

        Args:
            node: Source node ID or path
            max_distance: Maximum hops (default 2)
            min_strength: Minimum connection strength

        Returns:
            List of related nodes with connection info
        """
        node_id = await self._resolve_node_id(node)
        if not node_id:
            logger.warning("[UNTESTED PATH] Node not found in find_related")
            raise NotFoundError(f"Node not found: {node}")

        # BFS implementation for recursive search
        visited = {node_id}
        current_level = [
            (node_id, None, None, 1.0)
        ]  # (id, relation_type, from_node, accumulated_strength)
        all_results = []

        for distance in range(1, max_distance + 1):
            next_level = []

            # Process all nodes at current distance
            for current_id, prev_relation, prev_node, prev_strength in current_level:
                try:
                    # Find neighbors of current node
                    result = await self.db.execute_async(
                        """
                        SELECT 
                            n.id, n.node_type, n.path, n.name,
                            e.relation_type, e.strength,
                            CASE 
                                WHEN e.source_id = ? THEN 'outgoing'
                                ELSE 'incoming'
                            END as direction
                        FROM code_graph_edges e
                        JOIN code_graph_nodes n ON (
                            CASE 
                                WHEN e.source_id = ? THEN e.target_id = n.id
                                WHEN e.target_id = ? THEN e.source_id = n.id
                            END
                        )
                        WHERE e.strength >= ?
                        """,
                        (current_id, current_id, current_id, min_strength),
                        FetchType.ALL,
                    )
                    results: List[Dict[str, Any]] = result.data or []  # type: ignore

                    for row in results:
                        neighbor_id = row["id"]
                        if neighbor_id in visited:
                            continue
                        visited.add(neighbor_id)
                        strength_val = float(row["strength"])
                        accumulated_strength = prev_strength * strength_val
                        if accumulated_strength >= min_strength:
                            result_dict = dict(row)
                            result_dict["distance"] = distance
                            result_dict["accumulated_strength"] = accumulated_strength
                            result_dict["path_from_origin"] = (
                                f"{prev_node or node} -> {row['path']}"
                                if prev_node
                                else row['path']
                            )
                            all_results.append(result_dict)
                            if distance < max_distance:
                                next_level.append(
                                    (
                                        neighbor_id,
                                        row["relation_type"],
                                        row["path"],
                                        accumulated_strength,
                                    )
                                )

                except Exception as e:
                    logger.error("Error in BFS at distance", distance=distance, error=str(e))
                    continue

            # Move to next distance level
            current_level = next_level

            # Stop if no more nodes to explore
            if not current_level:
                break

        # Sort by accumulated strength (best connections first)
        all_results.sort(key=lambda x: x["accumulated_strength"], reverse=True)

        return all_results

    async def predict_impact(self, changed_node: str) -> Dict[str, Any]:
        """
        Predict the impact of changing a node.

        Args:
            changed_node: Modified node ID or path

        Returns:
            Dictionary with affected files and impact level
        """
        node_id = await self._resolve_node_id(changed_node)
        if not node_id:
            logger.warning("[UNTESTED PATH] Node not found in predict_impact")
            raise NotFoundError(f"Node not found: {changed_node}")

        try:
            # Find all nodes that depend on this node
            result = await self.db.execute_async(
                """
                SELECT 
                    n.path, n.name, e.relation_type, e.strength
                FROM code_graph_edges e
                JOIN code_graph_nodes n ON e.source_id = n.id
                WHERE e.target_id = ? 
                AND e.relation_type IN ('IMPORTS', 'CALLS', 'EXTENDS', 'IMPLEMENTS', 'USES')
                ORDER BY e.strength DESC
                """,
                (node_id,),
                FetchType.ALL,
            )
            dependents: List[Dict[str, Any]] = result.data or []  # type: ignore

            # Calculate impact
            high_impact = []
            medium_impact = []
            low_impact = []

            for dep in dependents:
                strength_val = float(dep["strength"])
                if strength_val > 0.7:
                    high_impact.append(dep["path"])
                elif strength_val > 0.4:
                    medium_impact.append(dep["path"])
                else:
                    low_impact.append(dep["path"])

            return {
                "total_affected": len(dependents),
                "high_impact": high_impact,
                "medium_impact": medium_impact,
                "low_impact": low_impact,
                "details": [dict(dep) for dep in dependents],
            }

        except Exception as e:
            logger.error("Error predicting impact", error=str(e))
            raise DatabaseError(f"Could not predict impact: {e}")

    async def _resolve_node_id(self, node_ref: str) -> Optional[str]:
        """
        Resolve an ID from path or ID.

        Args:
            node_ref: Hexadecimal ID or node path

        Returns:
            Node ID or None if it doesn't exist
        """
        # If it looks like a 32-char hex ID, verify it exists
        if len(node_ref) == 32 and all(c in "0123456789abcdef" for c in node_ref):
            result = await self.db.execute_async(
                "SELECT id FROM code_graph_nodes WHERE id = ?", (node_ref,), FetchType.ONE
            )
            data: Optional[Dict[str, Any]] = result.data  # type: ignore
            return data["id"] if data else None

        # Otherwise, search by path
        result = await self.db.execute_async(
            "SELECT id FROM code_graph_nodes WHERE path = ?", (node_ref,), FetchType.ONE
        )
        data: Optional[Dict[str, Any]] = result.data  # type: ignore
        return data["id"] if data else None
