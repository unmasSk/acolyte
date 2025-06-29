"""
Fatigue Monitor - Calculates system fatigue based on real Git metrics.

Uses GitMetadata fields to compute a fatigue score that indicates
when the system needs deep analysis and optimization.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import timedelta, datetime

from acolyte.core.logging import logger
from acolyte.core.utils.datetime_utils import utc_now, parse_iso_datetime
from acolyte.core.database import get_db_manager, FetchType
from acolyte.core.secure_config import Settings
from acolyte.models.common.metadata import GitMetadata
from acolyte.rag.enrichment import EnrichmentService
from acolyte.rag.retrieval.hybrid_search import HybridSearch, SearchFilters


class FatigueLevel:
    """Fatigue level constants and helpers."""

    # Fatigue level thresholds
    HEALTHY = 3.0
    MODERATE = 6.0
    HIGH = 7.5
    CRITICAL = 9.5

    @staticmethod
    def get_description(level: float) -> str:
        """Get human-readable description of fatigue level."""
        if level <= FatigueLevel.HEALTHY:
            return "Healthy system, fast searches"
        elif level <= FatigueLevel.MODERATE:
            return "Moderate accumulation of changes"
        elif level < FatigueLevel.HIGH:
            return "Need to reorganize memory soon"
        elif level < FatigueLevel.CRITICAL:
            return "Need to optimize NOW to maintain performance"
        else:
            return "Critical level - urgent optimization required"


class FatigueMonitor:
    """
    Monitors system fatigue based on real Git activity metrics.

    Fatigue represents:
    - Code churn and instability
    - Recent activity levels
    - Architectural changes
    - Time since last optimization
    """

    def __init__(self, weaviate_client: Any) -> None:
        """
        Initialize fatigue monitor.

        Args:
            weaviate_client: Weaviate client for search operations
        """
        self.db = get_db_manager()
        self.config = Settings()
        self.enrichment_service = EnrichmentService()

        # Initialize HybridSearch for querying code chunks
        self.search = HybridSearch(
            weaviate_client=weaviate_client,
            enable_compression=False,  # We don't need compression for metadata queries
        )

        # Get thresholds from config
        self.threshold = self.config.get("dream.fatigue_threshold", 7.5)
        self.emergency_threshold = self.config.get("dream.emergency_threshold", 9.5)

        logger.info("Fatigue monitor initialized", module="dream")

    async def calculate_fatigue(self) -> Dict[str, Any]:
        """
        Calculate current fatigue level using real Git metrics.

        Returns:
            Dict with total fatigue, components, and explanation
        """
        try:
            # Get individual fatigue components
            components = await self._calculate_fatigue_components()

            # Calculate total fatigue (0-10 scale)
            total_fatigue = sum(components.values())
            total_fatigue = min(total_fatigue, 10.0)  # Cap at 10

            # Generate explanation
            explanation = self._generate_fatigue_explanation(components, total_fatigue)

            # Check for triggers
            triggers = await self._check_fatigue_triggers()

            return {
                "total": round(total_fatigue, 1),
                "components": {k: round(v, 2) for k, v in components.items()},
                "explanation": explanation,
                "triggers": triggers,
                "threshold": self.threshold,
                "is_high": total_fatigue > self.threshold,
                "is_critical": total_fatigue > self.emergency_threshold,
            }

        except Exception as e:
            logger.error("Failed to calculate fatigue", error=str(e))
            # Return moderate defaults (5.0 = middle of 0-10 scale)
            # This avoids false alarms (too high) or ignoring issues (too low)
            return {
                "total": 5.0,  # Middle value: neither healthy nor critical
                "components": {},
                "explanation": "Unable to calculate current fatigue level",
                "triggers": [],
                "error": str(e),
            }

    async def _calculate_fatigue_components(self) -> Dict[str, float]:
        """
        Calculate individual fatigue components.

        Returns:
            Dict with component scores (each 0-3 range approx)
        """
        components = {}

        # 1. Time since last optimization (10% weight = max 1.0)
        time_factor = await self._calculate_time_factor()
        components["time_factor"] = time_factor * 1.0

        # 2. File instability (30% weight = max 3.0)
        instability = await self._calculate_file_instability()
        components["file_instability"] = instability * 3.0

        # 3. Recent activity (30% weight = max 3.0)
        activity = await self._calculate_recent_activity()
        components["recent_activity"] = activity * 3.0

        # 4. Code volatility (20% weight = max 2.0)
        volatility = await self._calculate_code_volatility()
        components["code_volatility"] = volatility * 2.0

        # 5. Architectural changes (10% weight = max 1.0)
        architectural = await self._calculate_architectural_changes()
        components["architectural_changes"] = architectural * 1.0

        return components

    async def _calculate_time_factor(self) -> float:
        """
        Calculate fatigue from time since last optimization.

        Returns:
            Score 0-1 based on days elapsed
        """
        try:
            result = await self.db.execute_async(
                "SELECT last_optimization FROM dream_state WHERE id = 1", (), FetchType.ONE
            )

            if not result.data or not isinstance(result.data, dict):
                # Never optimized or wrong data type
                return 1.0

            last_opt_str = result.data.get("last_optimization")

            if not last_opt_str:
                return 1.0

            # Parse datetime
            last_opt = parse_iso_datetime(last_opt_str)
            # utc_now() already returns UTC time

            # Calculate days elapsed
            days_elapsed = (utc_now() - last_opt).days

            # Score based on days (0-30 days maps to 0-1)
            score = min(days_elapsed / 30.0, 1.0)

            return score

        except Exception as e:
            logger.warning("Failed to calculate time factor", error=str(e))
            return 1.0  # Default to max value on error

    async def _calculate_file_instability(self) -> float:
        """
        Calculate average instability of recently modified files.

        Returns:
            Score 0-1 based on stability scores
        """
        try:
            # Search for recently modified files
            date_from = (utc_now() - timedelta(days=7)).isoformat()
            filters = SearchFilters(date_from=date_from)

            # Get chunks from recently modified files
            search_results = await self.search.search(
                query="",  # Empty query to get all matching the filter
                max_chunks=50,
                filters=filters,
            )

            if not search_results:
                return 0.0

            # Extract unique file paths
            file_paths = list({result.chunk.metadata.file_path for result in search_results})

            # Batch enrich all files at once
            all_metadata = await self.enrichment_service.enrich_files_batch(file_paths)

            # Calculate average instability
            total_instability = 0.0
            count = 0

            for file_path in file_paths:
                metadata = all_metadata.get(file_path, {})
                git_metadata = metadata.get("git_metadata")

                if git_metadata:
                    # Get stability score from GitMetadata
                    stability_score = git_metadata.get("stability_score", 0.5)
                    # Instability = 1 - stability_score
                    instability = 1.0 - stability_score
                    total_instability += instability
                    count += 1

            if count == 0:
                return 0.0

            # Average instability (already 0-1)
            return total_instability / count

        except Exception as e:
            logger.warning("Failed to calculate file instability", error=str(e))
            return 0.0  # Default to 0 on error

    async def _calculate_recent_activity(self) -> float:
        """
        Calculate activity level based on recent commits.

        Returns:
            Score 0-1 based on commit frequency
        """
        try:
            # Get files with activity in last 30 days
            date_from = (utc_now() - timedelta(days=30)).isoformat()
            filters = SearchFilters(date_from=date_from)

            # Get all active files
            search_results = await self.search.search(
                query="",  # Empty query to get all matching the filter
                max_chunks=200,  # More chunks for activity
                filters=filters,
            )

            if not search_results:
                return 0.0

            # Extract unique files
            file_paths = list({result.chunk.metadata.file_path for result in search_results})

            # Batch enrich
            all_metadata = await self.enrichment_service.enrich_files_batch(file_paths)

            # Calculate score based on commits
            total_commits = 0
            high_activity_files = 0

            if file_paths:
                for file_path in file_paths:
                    metadata = all_metadata.get(file_path, {})
                    git_metadata = metadata.get("git_metadata")

                    if git_metadata:
                        commits = git_metadata.get("commits_last_30_days", 0)
                        total_commits += commits
                        if commits > 10:  # High activity threshold
                            high_activity_files += 1

            # Score based on activity (normalize to 0-1)
            # Consider both number of active files and commit frequency
            file_score = min(len(file_paths) / 100.0, 1.0)  # 100 files = max
            commit_score = min(total_commits / 200.0, 1.0)  # 200 commits = max
            high_activity_score = min(high_activity_files / 10.0, 1.0)  # 10 files = max

            # Weighted average
            return file_score * 0.3 + commit_score * 0.4 + high_activity_score * 0.3

        except Exception as e:
            logger.warning("Failed to calculate recent activity", error=str(e))
            return 0.3  # Default low value

    async def _calculate_code_volatility(self) -> float:
        """
        Calculate average code volatility index.

        Returns:
            Score 0-1 based on volatility
        """
        try:
            # Get recently modified files
            date_from = (utc_now() - timedelta(days=7)).isoformat()
            filters = SearchFilters(date_from=date_from)

            search_results = await self.search.search(
                query="",
                max_chunks=30,
                filters=filters,
            )

            if not search_results:
                return 0.0

            # Extract unique file paths
            file_paths = list({result.chunk.metadata.file_path for result in search_results})

            # Batch enrich all files at once
            all_metadata = await self.enrichment_service.enrich_files_batch(file_paths)

            total_volatility = 0.0
            count = 0

            for file_path in file_paths:
                metadata = all_metadata.get(file_path, {})
                git_metadata = metadata.get("git_metadata")

                if git_metadata:
                    volatility = git_metadata.get("code_volatility_index", 0.0)
                    total_volatility += volatility
                    count += 1

            if count == 0:
                return 0.0

            # Average volatility (already 0-1)
            return total_volatility / count

        except Exception as e:
            logger.warning("Failed to calculate code volatility", error=str(e))
            return 0.2  # Default low value

    async def _calculate_architectural_changes(self) -> float:
        """
        Detect architectural changes (package.json, requirements.txt, etc).

        Returns:
            Score 0-1 based on architectural change detection
        """
        try:
            # Check for changes in key architectural files
            arch_files = [
                "package.json",
                "package-lock.json",
                "requirements.txt",
                "pyproject.toml",
                "poetry.lock",
                "go.mod",
                "go.sum",
                "Cargo.toml",
                "Cargo.lock",
                "pom.xml",
                "build.gradle",
                "Gemfile",
                "Gemfile.lock",
                ".gitmodules",
                "Dockerfile",
                "docker-compose.yml",
            ]

            # Search for these files modified recently
            date_from = (utc_now() - timedelta(days=7)).isoformat()

            # PERFORMANCE FIX: Batch search instead of 16 individual searches
            # Get all recently modified files in one query
            filters = SearchFilters(date_from=date_from)
            recent_results = await self.search.search(query="", max_chunks=100, filters=filters)

            # Create set for O(1) lookup of architectural files
            arch_files_set = set(arch_files)

            # Count how many architectural files were found in results
            arch_changes = 0
            found_arch_files = set()

            for result in recent_results:
                file_path = result.chunk.metadata.file_path
                # Extract just the filename from the path
                file_name = (
                    file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
                )

                if file_name in arch_files_set and file_name not in found_arch_files:
                    found_arch_files.add(file_name)
                    arch_changes += 1

            logger.debug(
                "Architectural files scan completed",
                total_files_scanned=len(recent_results),
                arch_files_found=arch_changes,
                files=list(found_arch_files)[:5],  # Log first 5 for debugging
            )

            # Check for directory restructuring by sampling recent files
            # OPTIMIZATION: Reuse results from above search, just take first 10 for restructuring analysis
            restructure_sample = recent_results[:10] if recent_results else []

            total_restructures = 0
            if restructure_sample:
                # Extract file paths for enrichment
                file_paths = list(
                    {result.chunk.metadata.file_path for result in restructure_sample}
                )

                # Batch enrich all files at once
                all_metadata = await self.enrichment_service.enrich_files_batch(file_paths)

                for file_path in file_paths:
                    metadata = all_metadata.get(file_path, {})
                    git_metadata = metadata.get("git_metadata")

                    if git_metadata:
                        restructures = git_metadata.get("directories_restructured", 0)
                        total_restructures += restructures

            # Score based on changes
            arch_score = min(arch_changes / 3.0, 1.0)  # 3 files = max
            restructure_score = min(total_restructures / 5.0, 1.0)  # 5 restructures = max

            # Weighted average
            return arch_score * 0.7 + restructure_score * 0.3

        except Exception as e:
            logger.warning("Failed to calculate architectural changes", error=str(e))
            return 0.0

    async def _get_file_git_metadata(self, file_path: str) -> Optional[GitMetadata]:
        """
        Get Git metadata for a file from enrichment service.

        Args:
            file_path: Path to file

        Returns:
            GitMetadata object or None
        """
        try:
            # Get from enrichment service (it has caching)
            metadata = await self.enrichment_service.enrich_file(file_path)
            return metadata.get("git_metadata")
        except Exception as e:
            logger.debug("Could not get Git metadata", file_path=file_path, error=str(e))
            return None

    def _generate_fatigue_explanation(self, components: Dict[str, float], total: float) -> str:
        """
        Generate human-readable explanation of fatigue.

        Args:
            components: Individual component scores
            total: Total fatigue score

        Returns:
            Explanation string
        """
        reasons = []

        # Check each component
        if components.get("file_instability", 0) > 1.5:
            reasons.append("unstable files with frequent changes")

        if components.get("recent_activity", 0) > 2.0:
            reasons.append("high recent code activity")

        if components.get("code_volatility", 0) > 1.0:
            reasons.append("high code volatility")

        if components.get("architectural_changes", 0) > 0.5:
            reasons.append("architectural changes detected")

        if components.get("time_factor", 0) > 0.5:
            reasons.append("time since last optimization")

        if not reasons:
            return "Stable system with normal activity"

        base = "Fatigue level {:.1f} due to: ".format(total)
        return base + ", ".join(reasons)

    async def _check_fatigue_triggers(self) -> List[Dict[str, Any]]:
        """
        Check for specific triggers that indicate optimization need.

        Returns:
            List of detected triggers
        """
        triggers = []

        try:
            # Get recent files for analysis
            date_from = (utc_now() - timedelta(days=7)).isoformat()
            filters = SearchFilters(date_from=date_from)

            search_results = await self.search.search(query="", max_chunks=10, filters=filters)

            if search_results:
                # Extract unique file paths
                file_paths = list({result.chunk.metadata.file_path for result in search_results})

                # Batch enrich all files at once
                all_metadata = await self.enrichment_service.enrich_files_batch(file_paths)

                high_conflict_files = []
                rapid_change_files = []

                for file_path in file_paths:
                    metadata = all_metadata.get(file_path, {})
                    git_metadata = metadata.get("git_metadata")

                    if git_metadata:
                        # Check for high conflicts
                        if git_metadata.get("merge_conflicts_count", 0) > 3:
                            high_conflict_files.append(file_path)

                        # Check for rapid changes
                        commits_30d = git_metadata.get("commits_last_30_days", 0)
                        stability = git_metadata.get("stability_score", 1.0)
                        if commits_30d > 20 and stability < 0.3:
                            rapid_change_files.append(file_path)

                if high_conflict_files:
                    triggers.append(
                        {
                            "type": "high_conflicts",
                            "severity": "high",
                            "message": "{} files with conflict history".format(
                                len(high_conflict_files)
                            ),
                            "files": high_conflict_files[:3],  # Show first 3
                        }
                    )

                if rapid_change_files:
                    triggers.append(
                        {
                            "type": "rapid_changes",
                            "severity": "medium",
                            "message": "{} files changing rapidly".format(len(rapid_change_files)),
                            "files": rapid_change_files[:3],
                        }
                    )

        except Exception as e:
            logger.warning("Failed to check triggers", error=str(e))

        return triggers

    async def reduce_fatigue(self, factor: float = 0.3) -> None:
        """
        Reduce fatigue level after optimization.

        Args:
            factor: Multiplication factor (0.3 = reduce by 70%)
        """
        try:
            await self.db.execute_async(
                """
                UPDATE dream_state
                SET fatigue_level = fatigue_level * ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
                """,
                (factor,),
            )

            logger.info("Fatigue reduced", reduction_percent=(1 - factor) * 100)

        except Exception as e:
            logger.error("Failed to reduce fatigue", error=str(e))

    async def get_fatigue_history(self, days: int = 7) -> List[Tuple[datetime, float]]:
        """
        Get estimated fatigue level history.

        Since we don't track historical fatigue values, this method
        reconstructs an estimated history based on:
        - Current fatigue level
        - Last optimization time
        - Recent code activity patterns

        Args:
            days: Days of history to retrieve

        Returns:
            List of (timestamp, fatigue_level) tuples
        """
        try:
            # Get current fatigue
            current_fatigue = await self.calculate_fatigue()
            current_level = current_fatigue["total"]

            # Get last optimization time
            result = await self.db.execute_async(
                "SELECT last_optimization FROM dream_state WHERE id = 1", (), FetchType.ONE
            )

            last_optimization = None
            if result.data and isinstance(result.data, dict):
                last_opt_str = result.data.get("last_optimization")
                if last_opt_str:
                    last_optimization = parse_iso_datetime(last_opt_str)

            # Build history points
            history: List[Tuple[datetime, float]] = []
            now = utc_now()

            # Calculate points for each day
            for day in range(days + 1):
                timestamp = now - timedelta(days=day)

                # Estimate fatigue based on days since optimization
                if last_optimization and timestamp < last_optimization:
                    # Before last optimization - estimate high fatigue
                    # Fatigue was likely high before optimization
                    estimated_level = min(8.5 + (day * 0.1), 10.0)
                elif last_optimization and timestamp.date() == last_optimization.date():
                    # Day of optimization - show reduction
                    # Fatigue drops by 70% after optimization
                    estimated_level = current_level * 0.3
                else:
                    # After optimization - gradual increase
                    if last_optimization:
                        days_since_opt = (timestamp - last_optimization).days
                        # Fatigue increases roughly 0.2 per day
                        growth_rate = 0.2
                        estimated_level = min(3.0 + (days_since_opt * growth_rate), current_level)
                    else:
                        # No optimization history - linear interpolation
                        # Assume fatigue grows from 5.0 at rate of 0.5/day
                        estimated_level = min(5.0 + ((days - day) * 0.5), current_level)

                # Add to history (in reverse chronological order)
                history.insert(0, (timestamp, round(estimated_level, 1)))

            return history

        except Exception as e:
            logger.error("Failed to get fatigue history", error=str(e))
            # Return simple fallback - current level for all days
            return [(utc_now() - timedelta(days=d), 5.0) for d in range(days + 1)]
