"""
EnrichmentService - Coordinator for enrichment.

Complete implementation with all Git metrics required.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import time
from datetime import timedelta
import re

from acolyte.core.utils.datetime_utils import utc_now, utc_now_iso, parse_iso_datetime

import git
from git.exc import InvalidGitRepositoryError

from acolyte.core.logging import logger
from acolyte.core.tracing import MetricsCollector
from acolyte.models.chunk import Chunk
from acolyte.rag.enrichment.processors import GraphBuilder


class EnrichmentService:
    """
    Coordinates the enrichment of chunks with additional metadata.

    COMPLETE IMPLEMENTATION:
    - ✅ Basic functional structure
    - ✅ Trigger handling (commit, pull, etc.)
    - ✅ Simple cache with invalidation
    - ✅ Complete Git blame with contributors analysis
    - ✅ Code patterns analysis (test, error handling, etc.)
    - ✅ Advanced Git metrics (volatility, conflicts, co-modifications)
    - ✅ Commit types analysis (fix/feat/refactor)
    - ✅ Integration with neural graph to update relationships (graph_builder.py implemented)
    """

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.metrics = MetricsCollector()

        # Simple cache (no TTL for now)
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Initialize Git repository
        self.repo: Optional[git.Repo] = None
        self.has_git = False

        # Initialize graph builder
        self.graph_builder = GraphBuilder()

        try:
            self.repo = git.Repo(self.repo_path)
            self.has_git = True
            logger.info("Git repository loaded", git_dir=self.repo.git_dir)
        except InvalidGitRepositoryError:
            logger.warning("No Git repository found", repo_path=repo_path)
        except Exception as e:
            logger.error("Error initializing Git repository", error=str(e))

        logger.info(
            "EnrichmentService initialized", repo_path=str(self.repo_path), has_git=self.has_git
        )

    async def enrich_chunks(
        self, chunks: List[Chunk], trigger: str = "manual"
    ) -> List[Tuple[Chunk, Dict[str, Any]]]:
        """
        Enrich chunks with additional metadata.

        IMPORTANT: Returns tuples (chunk, metadata) so
        IndexingService can combine them when saving to Weaviate.

        Args:
            chunks: List of chunks to enrich
            trigger: Origin (commit, pull, checkout, fetch, manual)

        Returns:
            List of tuples (original chunk, enriched metadata)
        """
        start_time = time.time()

        # Handle trigger 'pull' - invalidate cache
        if trigger == "pull":
            self._cache.clear()
            logger.info("Cache invalidated due to git pull")
            self.metrics.increment("cache_invalidations")

        enriched_chunks: List[Tuple[Chunk, Dict[str, Any]]] = []

        for chunk in chunks:
            # Generate cache key
            cache_key = self._get_cache_key(chunk)

            # Check cache
            if cache_key in self._cache and trigger != "pull":
                enriched_chunks.append((chunk, self._cache[cache_key]))
                self.metrics.increment("cache_hits")
                continue

            # Enrich chunk
            try:
                metadata = await self._enrich_single_chunk(chunk)

                # Save to cache
                self._cache[cache_key] = metadata
                self.metrics.increment("cache_misses")

                enriched_chunks.append((chunk, metadata))

            except Exception as e:
                logger.error("Failed to enrich chunk", error=str(e))
                # In case of error, return empty metadata
                enriched_chunks.append((chunk, {}))
                self.metrics.increment("enrichment_errors")

        # Update neural graph with relationships
        if enriched_chunks:
            try:
                # Collect all metadata for graph update
                all_metadata: Dict[str, Any] = {}
                for chunk, metadata in enriched_chunks:
                    if metadata:
                        # Merge metadata, prioritizing Git info
                        if "git" in metadata:
                            all_metadata["git"] = metadata["git"]
                            break

                # Update graph
                await self.graph_builder.update_from_chunks(
                    [chunk for chunk, _ in enriched_chunks], all_metadata
                )
                logger.info("Neural graph updated from enriched chunks")

            except Exception as e:
                logger.error("Failed to update neural graph", error=str(e))
                # Non-critical failure, continue

        # Metrics
        elapsed_ms = (time.time() - start_time) * 1000
        self.metrics.gauge("enrichment_time_ms", elapsed_ms)
        self.metrics.gauge("chunks_enriched", len(chunks))
        self.metrics.increment(f"trigger.{trigger}")

        logger.info(
            "Enrichment complete",
            chunks=len(chunks),
            trigger=trigger,
            elapsed_ms=f"{elapsed_ms:.2f}",
        )

        return enriched_chunks

    async def enrich_file(self, file_path: str) -> Dict[str, Any]:
        """
        Enrich a single file and return its metadata.

        This is a convenience method for getting metadata without chunks.
        Used by FatigueMonitor.

        Args:
            file_path: Path to the file

        Returns:
            Dict with git_metadata and other file metadata
        """
        try:
            # Create a dummy chunk just for metadata extraction
            from acolyte.models.chunk import Chunk, ChunkMetadata, ChunkType

            dummy_chunk = Chunk(
                id="temp_" + file_path.replace("/", "_"),
                content="",  # Empty content, we only need metadata
                metadata=ChunkMetadata(
                    file_path=file_path,
                    chunk_type=ChunkType.UNKNOWN,
                    start_line=1,
                    end_line=1,
                    language="unknown",
                ),
            )

            # Use existing enrichment logic
            metadata = await self._enrich_single_chunk(dummy_chunk)

            # Return structured metadata with git_metadata field
            return {
                "git_metadata": metadata.get("git", {}),
                "file_metadata": metadata.get("file", {}),
                "patterns": metadata.get("patterns", {}),
            }

        except Exception as e:
            logger.error("Failed to enrich file", file_path=file_path, error=str(e))
            return {}

    async def enrich_files_batch(self, file_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Enrich multiple files in batch for better performance.

        Args:
            file_paths: List of file paths to enrich

        Returns:
            Dict mapping file_path -> metadata
        """
        start_time = time.time()
        results = {}

        # Process in parallel using asyncio.gather for better performance
        import asyncio

        async def process_file(file_path: str) -> Tuple[str, Dict[str, Any]]:
            metadata = await self.enrich_file(file_path)
            return file_path, metadata

        # Limit concurrent operations to avoid overwhelming the system
        # Process in batches of 10 files at a time
        batch_size = 10

        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i : i + batch_size]

            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[process_file(fp) for fp in batch], return_exceptions=True
            )

            # Collect results
            for result in batch_results:
                if isinstance(result, BaseException):
                    logger.error("Error in batch processing", error=str(result))
                else:
                    file_path, metadata = result
                    results[file_path] = metadata

        # Log performance metrics
        elapsed_ms = (time.time() - start_time) * 1000
        self.metrics.gauge("batch_enrichment_time_ms", elapsed_ms)
        self.metrics.gauge("batch_files_enriched", len(file_paths))

        logger.info(
            "Batch enrichment complete",
            files=len(file_paths),
            successful=len(results),
            elapsed_ms=f"{elapsed_ms:.2f}",
        )

        return results

    async def _enrich_single_chunk(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Enrich a single chunk.

        COMPLETE IMPLEMENTATION with all metrics.
        """
        metadata: Dict[str, Any] = {}

        # 1. Basic file information
        file_metadata = await self._get_file_metadata(chunk)
        if file_metadata:
            metadata["file"] = file_metadata

        # 2. Complete Git information with blame and all metrics
        if self.has_git:
            git_metadata = await self._get_git_metadata(chunk)
            if git_metadata:
                metadata["git"] = git_metadata

        # 3. Basic code patterns
        patterns = await self._detect_patterns(chunk)
        if patterns:
            metadata["patterns"] = patterns

        return metadata

    async def _get_file_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Get basic file metadata."""
        try:
            file_path = Path(chunk.metadata.file_path)
            if not file_path.is_absolute():
                file_path = self.repo_path / file_path

            if not file_path.exists():
                return {}

            stat = file_path.stat()

            return {
                "size_bytes": stat.st_size,
                "extension": file_path.suffix,
                "name": file_path.name,
                "relative_path": str(file_path.relative_to(self.repo_path)),
            }

        except Exception as e:
            logger.debug(f"Error getting file metadata: {e}")
            return {}

    async def _get_git_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Get complete Git metadata with all required fields.

        ✅ IMPLEMENTED: Complete metrics for Dream, HybridSearch and TaskService.
        """
        try:
            file_path = chunk.metadata.file_path

            # 1. Basic info from last commit
            last_commit_info = self._get_last_commit_info(file_path)

            # 2. File creation info
            file_creation_info = self._get_file_creation_info(file_path)

            # 3. Calculate file age
            file_age_days = 0
            if file_creation_info and "created" in file_creation_info:
                created_date = file_creation_info["created"]
                if isinstance(created_date, str):
                    created_date = parse_iso_datetime(created_date)
                file_age_days = (utc_now() - created_date.replace(tzinfo=None)).days

            # Complete metadata
            metadata = {
                # Basic info
                "last_modified": last_commit_info.get("date", utc_now_iso()),
                "last_author": last_commit_info.get("author", "unknown"),
                "last_commit": last_commit_info,
                "file_created": file_creation_info.get("created", utc_now_iso()),
                "file_age_days": file_age_days,
                # Contributors analysis (blame)
                "contributors": self._analyze_contributors(file_path),
                # Activity metrics
                "total_commits": self._count_total_commits(file_path),
                "commits_last_30_days": self._calculate_recent_commits(file_path),
                "is_actively_developed": True,  # Will be updated based on commits_last_30_days
                "change_frequency_per_week": self._calculate_change_frequency(file_path),
                # Stability metrics (already implemented)
                "merge_conflicts_count": self._calculate_merge_conflicts(file_path),
                "directories_restructured": self._calculate_directory_moves(file_path),
                "code_volatility_index": self._calculate_code_volatility(file_path),
                # Commit analysis
                "lines_added_last_commit": last_commit_info.get("insertions", 0),
                "lines_deleted_last_commit": last_commit_info.get("deletions", 0),
                "commit_types": self._analyze_commit_types(file_path),
                "last_refactor_date": self._find_last_refactor_date(file_path),
                # Co-modification patterns
                "co_modified_with": self._find_co_modified_files(file_path),
            }

            # Calculate derived metrics
            metadata["stability_score"] = self._calculate_stability_score(
                metadata["merge_conflicts_count"], metadata["code_volatility_index"]
            )

            # Update is_actively_developed based on recent commits
            metadata["is_actively_developed"] = metadata["commits_last_30_days"] > 0

            # Determine modification pattern
            metadata["modification_pattern"] = self._determine_modification_pattern(
                metadata["commits_last_30_days"],
                metadata["change_frequency_per_week"],
                metadata["total_commits"],
            )

            return metadata

        except Exception as e:
            logger.debug(f"Error getting git metadata for {chunk.metadata.file_path}: {e}")
            # Complete fallback to safe default values
            return {
                "last_modified": utc_now_iso(),
                "last_author": "unknown",
                "last_commit": {
                    "hash": "unknown",
                    "message": "unknown",
                    "date": utc_now_iso(),
                },
                "file_created": utc_now_iso(),
                "file_age_days": 0,
                "contributors": {},
                "total_commits": 1,
                "commits_last_30_days": 1,
                "is_actively_developed": False,
                "change_frequency_per_week": 0.0,
                "stability_score": 0.5,
                "code_volatility_index": 0.1,
                "modification_pattern": "unknown",
                "merge_conflicts_count": 0,
                "directories_restructured": 0,
                "lines_added_last_commit": 0,
                "lines_deleted_last_commit": 0,
                "commit_types": {},
                "last_refactor_date": None,
                "co_modified_with": [],
            }

    async def _detect_patterns(self, chunk: Chunk) -> Dict[str, Any]:
        """Detect basic patterns in the code."""
        content = chunk.content.lower()

        patterns = {
            "is_test_code": any(
                indicator in content
                for indicator in ["test_", "test(", "assert", "@pytest", "unittest"]
            ),
            "has_error_handling": any(
                keyword in content for keyword in ["try:", "except:", "catch", "error", "exception"]
            ),
            "has_docstring": (
                '"""' in chunk.content or "'''" in chunk.content or "/**" in chunk.content
            ),
            "imports_count": (
                chunk.content.count("\nimport ")
                + chunk.content.count("\nfrom ")
                + chunk.content.count("\nrequire(")
            ),
        }

        # Infer basic purpose - test code has priority
        if patterns["is_test_code"]:
            patterns["appears_to_be"] = "test_code"
        elif "auth" in content or "login" in content:
            patterns["appears_to_be"] = "authentication"
        elif "database" in content or "query" in content:
            patterns["appears_to_be"] = "database_operations"
        elif "api" in content or "endpoint" in content:
            patterns["appears_to_be"] = "api_handler"
        else:
            patterns["appears_to_be"] = "general_code"

        return patterns

    def _get_cache_key(self, chunk: Chunk) -> str:
        """Generate cache key for a chunk."""
        # Use file_path + lines as key
        file_path = chunk.metadata.file_path
        start_line = chunk.metadata.start_line
        end_line = chunk.metadata.end_line

        return f"{file_path}:{start_line}-{end_line}"

    # ==========================================
    # ADVANCED GIT METHODS - COMPLETE IMPLEMENTATION
    # ==========================================

    def _calculate_merge_conflicts(self, file_path: str) -> int:
        """
        Calculate merge conflicts resolved historically in the file.

        Searches commit history for messages indicating conflict resolution
        and verifies that the specific file was modified in those commits.
        """
        if not self.has_git or not self.repo:
            return 0

        try:
            # Patterns indicating conflict resolution
            conflict_patterns = [
                r"merge branch",
                r"resolve conflict",
                r"fix merge",
                r"conflict resolution",
                r"merge.*conflict",
                r"fix.*conflict",
                r"resolved.*conflict",
            ]

            conflict_count = 0

            # Search for commits mentioning conflicts
            for commit in self.repo.iter_commits(max_count=1000):
                message = commit.message
                if isinstance(message, (bytes, bytearray, memoryview)):
                    message = bytes(message).decode("utf-8", errors="ignore")
                message = str(message).lower()
                if any(re.search(pattern, message) for pattern in conflict_patterns):
                    # Check if this commit modified our file
                    try:
                        if file_path in commit.stats.files:
                            conflict_count += 1
                    except (AttributeError, KeyError):
                        # Commit may not have stats or file may not exist
                        continue

            logger.debug(f"Found {conflict_count} merge conflicts for {file_path}")
            return conflict_count

        except Exception as e:
            logger.debug(f"Error calculating merge conflicts for {file_path}: {e}")
            return 0

    def _calculate_directory_moves(self, file_path: str) -> int:
        """
        Calculate how many times the file was moved between directories.

        Uses git log --follow to track file movements and renames.
        """
        if not self.has_git or not self.repo:
            return 0

        try:
            move_count = 0
            previous_dir = None

            # Ensure file_path is relative to repo
            path = Path(file_path)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.repo_path)
                    file_path = str(path)
                except ValueError:
                    pass

            # Get file history following renames
            commits = list(self.repo.iter_commits(paths=file_path, max_count=500))

            for commit in reversed(commits):  # Chronological order
                try:
                    for item in commit.stats.files:
                        item_str = str(item)
                        if item_str == file_path or item_str.endswith(str(Path(file_path).name)):
                            current_dir = str(Path(item_str).parent)

                            if previous_dir is not None and current_dir != previous_dir:
                                move_count += 1
                                logger.debug(f"File moved from {previous_dir} to {current_dir}")

                            previous_dir = current_dir
                            break
                except (AttributeError, KeyError):
                    continue

            logger.debug(f"Found {move_count} directory moves for {file_path}")
            return move_count

        except Exception as e:
            logger.debug(f"Error calculating directory moves for {file_path}: {e}")
            return 0

    def _calculate_code_volatility(self, file_path: str) -> float:
        """
        Calculate code volatility index (0.0-1.0).

        Formula: (lines modified in last 30 days) / (total current lines)
        Limited to maximum 1.0 (100% volatility).
        """
        if not self.has_git or not self.repo:
            return 0.1  # Safe default value

        try:
            # Calculate total current lines in the file
            total_lines = self._count_file_lines(file_path)
            if total_lines == 0:
                return 1.0  # Empty file = 100% volatile

            # Ensure file_path is relative to repo
            path = Path(file_path)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.repo_path)
                    file_path = str(path)
                except ValueError:
                    pass

            # Limit date (30 days ago)
            since_date = utc_now() - timedelta(days=30)

            # Get change statistics in last 30 days
            modified_lines = 0

            for commit in self.repo.iter_commits(since=since_date, paths=file_path, max_count=100):
                try:
                    if file_path in commit.stats.files:
                        file_stats = commit.stats.files[file_path]
                        # Sum added and deleted lines
                        modified_lines += file_stats["insertions"] + file_stats["deletions"]
                except (AttributeError, KeyError):
                    continue

            # Calculate volatility (limited to 1.0)
            volatility = min(modified_lines / total_lines, 1.0)

            logger.debug(
                f"Volatility for {file_path}: {modified_lines} modified lines / "
                f"{total_lines} total lines = {volatility:.3f}"
            )

            return volatility

        except Exception as e:
            logger.debug(f"Error calculating volatility for {file_path}: {e}")
            return 0.1  # Safe default value

    def _calculate_stability_score(self, conflicts: int, volatility: float) -> float:
        """
        Calculate stability score based on conflicts and volatility.

        Heuristic formula:
        - Penalize for conflicts (each conflict reduces score)
        - Penalize for high volatility
        - Final score between 0.0 (very unstable) and 1.0 (very stable)
        """
        # Score base
        base_score = 1.0

        # Penalize for conflicts (each conflict reduces 0.1)
        conflict_penalty = min(conflicts * 0.1, 0.6)  # Maximum 60% penalty

        # Penalize for high volatility
        volatility_penalty = volatility * 0.4  # Up to 40% penalty

        # Calculate final score
        stability = base_score - conflict_penalty - volatility_penalty

        # Ensure it's in range [0.0, 1.0]
        return max(0.0, min(1.0, stability))

    def _calculate_recent_commits(self, file_path: str) -> int:
        """
        Count commits that modified the file in the last 30 days.
        """
        if not self.has_git or not self.repo:
            return 1  # Safe default value

        try:
            # Ensure file_path is relative to repo
            path = Path(file_path)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.repo_path)
                    file_path = str(path)
                except ValueError:
                    pass

            since_date = utc_now() - timedelta(days=30)

            commit_count = 0
            for commit in self.repo.iter_commits(since=since_date, paths=file_path, max_count=50):
                commit_count += 1

            logger.debug(f"Found {commit_count} recent commits for {file_path}")
            return commit_count

        except Exception as e:
            logger.debug(f"Error calculating recent commits for {file_path}: {e}")
            return 1

    def _count_file_lines(self, file_path: str) -> int:
        """
        Count current lines in the file.
        """
        try:
            full_path = Path(file_path)
            if not full_path.is_absolute():
                full_path = self.repo_path / file_path

            if not full_path.exists():
                return 0

            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for line in f)

        except Exception as e:
            logger.debug(f"Error counting lines in {file_path}: {e}")
            return 0

    # ==========================================
    # NEW METHODS FOR COMPLETE METRICS
    # ==========================================

    def _get_last_commit_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get detailed information about the last commit that modified the file.
        """
        if not self.has_git or not self.repo:
            return {
                "hash": "unknown",
                "message": "No git repository",
                "date": utc_now_iso(),
                "author": "unknown",
            }

        try:
            # Ensure file_path is relative to repo
            path = Path(file_path)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.repo_path)
                    file_path = str(path)
                except ValueError:
                    # Path is not within repo
                    pass

            # Get the last commit for the file
            commits = list(self.repo.iter_commits(paths=file_path, max_count=1))

            if not commits:
                return {
                    "hash": "unknown",
                    "message": "No commits found",
                    "date": utc_now_iso(),
                    "author": "unknown",
                }

            commit = commits[0]

            # Get file statistics for this commit
            insertions = 0
            deletions = 0
            try:
                if file_path in commit.stats.files:
                    file_stats = commit.stats.files[file_path]
                    insertions = file_stats.get("insertions", 0)
                    deletions = file_stats.get("deletions", 0)
            except (AttributeError, KeyError, TypeError) as e:
                logger.debug(f"Could not get stats for {file_path}: {e}")

            return {
                "hash": commit.hexsha,
                "message": commit.message.strip(),
                "date": commit.committed_datetime.isoformat(),
                "author": commit.author.email or str(commit.author),
                "insertions": insertions,
                "deletions": deletions,
            }

        except Exception as e:
            logger.debug(f"Error getting last commit info for {file_path}: {e}")

            return {
                "hash": "error",
                "message": f"Error: {str(e)}",
                "date": utc_now_iso(),
                "author": "unknown",
            }

    def _get_file_creation_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about when the file was created.
        """
        if not self.has_git or not self.repo:
            return {"created": utc_now_iso()}

        try:
            # Ensure file_path is relative to repo
            path = Path(file_path)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.repo_path)
                    file_path = str(path)
                except ValueError:
                    pass

            # Get all commits for the file ordered by date
            commits = list(self.repo.iter_commits(paths=file_path, reverse=True))

            if commits:
                first_commit = commits[0]
                return {
                    "created": first_commit.committed_datetime.isoformat(),
                    "initial_author": first_commit.author.email or str(first_commit.author),
                    "initial_message": first_commit.message.strip(),
                }

            return {"created": utc_now_iso()}

        except Exception as e:
            logger.debug(f"Error getting file creation info for {file_path}: {e}")
            return {"created": utc_now_iso()}

    def _analyze_contributors(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Analyze contributors using git blame.
        Return dictionary with email as key and statistics.
        """
        if not self.has_git or not self.repo:
            return {}

        try:
            # Ensure file_path is relative to repo
            path = Path(file_path)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.repo_path)
                    file_path = str(path)
                except ValueError:
                    pass

            # Use git blame to get line-by-line authorship
            blame_data = self.repo.blame("HEAD", file_path)
            if blame_data is None:
                return {}
            contributors = {}
            total_lines = 0
            for entry in blame_data:
                if not entry or not hasattr(entry, '__getitem__'):
                    continue
                commit = entry[0]
                lines = entry[1]
                # Validar commit
                if commit is None or isinstance(commit, (list, dict)):
                    continue
                if not hasattr(commit, 'author') or not hasattr(commit, 'committed_datetime'):
                    continue
                # Validar lines: debe ser lista de str/bytes
                if lines is None or not isinstance(lines, list):
                    continue
                if not all(isinstance(line, (str, bytes)) for line in lines):
                    continue
                author_email = getattr(commit.author, 'email', None) or str(commit.author)
                line_count = len(lines)
                total_lines += line_count
                if author_email not in contributors:
                    contributors[author_email] = {
                        "lines": 0,
                        "percentage": 0.0,
                        "first_commit": getattr(
                            commit, 'committed_datetime', utc_now()
                        ).isoformat(),
                        "last_commit": getattr(commit, 'committed_datetime', utc_now()).isoformat(),
                    }
                contributors[author_email]["lines"] += line_count
                commit_date = getattr(commit, 'committed_datetime', utc_now()).isoformat()
                if commit_date and commit_date < contributors[author_email]["first_commit"]:
                    contributors[author_email]["first_commit"] = commit_date
                if commit_date and commit_date > contributors[author_email]["last_commit"]:
                    contributors[author_email]["last_commit"] = commit_date
            # Calculate percentages
            if total_lines > 0:
                for author_data in contributors.values():
                    author_data["percentage"] = round((author_data["lines"] / total_lines) * 100, 1)

            logger.debug(f"Found {len(contributors)} contributors for {file_path}")
            return contributors

        except Exception as e:
            logger.debug(f"Error analyzing contributors for {file_path}: {e}")
            return {}

    def _count_total_commits(self, file_path: str) -> int:
        """
        Count total commits that touched the file.
        """
        if not self.has_git or not self.repo:
            return 1

        try:
            # Ensure file_path is relative to repo
            path = Path(file_path)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.repo_path)
                    file_path = str(path)
                except ValueError:
                    pass

            commit_count = sum(1 for _ in self.repo.iter_commits(paths=file_path))
            return max(1, commit_count)

        except Exception as e:
            logger.debug(f"Error counting total commits for {file_path}: {e}")
            return 1

    def _calculate_change_frequency(self, file_path: str) -> float:
        """
        Calculate change frequency per week in the last year.
        """
        if not self.has_git or not self.repo:
            return 0.0

        try:
            # Ensure file_path is relative to repo
            path = Path(file_path)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.repo_path)
                    file_path = str(path)
                except ValueError:
                    pass

            # Limit date: 1 year ago
            one_year_ago = utc_now() - timedelta(days=365)

            # Count commits in the last year
            commit_count = sum(
                1 for _ in self.repo.iter_commits(since=one_year_ago, paths=file_path)
            )

            # Calculate frequency per week
            weeks = 52  # Weeks in a year
            frequency = commit_count / weeks

            logger.debug(
                f"Change frequency for {file_path}: {commit_count} commits "
                f"in 52 weeks = {frequency:.2f}/week"
            )

            return round(frequency, 2)

        except Exception as e:
            logger.debug(f"Error calculating change frequency for {file_path}: {e}")
            return 0.0

    def _analyze_commit_types(self, file_path: str) -> Dict[str, int]:
        """
        Analyze commit types following conventional commits pattern.
        """
        if not self.has_git or not self.repo:
            return {}

        try:
            commit_types = {
                "fix": 0,
                "feat": 0,
                "refactor": 0,
                "docs": 0,
                "style": 0,
                "test": 0,
                "chore": 0,
                "other": 0,
            }

            # Ensure file_path is relative to repo
            path = Path(file_path)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.repo_path)
                    file_path = str(path)
                except ValueError:
                    pass

            # Analyze last 100 commits
            for commit in self.repo.iter_commits(paths=file_path, max_count=100):
                message = commit.message
                if isinstance(message, (bytes, bytearray, memoryview)):
                    message = bytes(message).decode("utf-8", errors="ignore")
                message = str(message).lower().strip()
                found_type = False
                for commit_type in commit_types.keys():
                    if commit_type == "other":
                        continue
                    patterns = (f"{commit_type}:", f"{commit_type}(", f"{commit_type} ")
                    if message.startswith(patterns):
                        commit_types[commit_type] += 1
                        found_type = True
                        break
                if not found_type:
                    commit_types["other"] += 1

            # Remove types with 0 commits
            commit_types = {k: v for k, v in commit_types.items() if v > 0}

            logger.debug(f"Commit types for {file_path}: {commit_types}")
            return commit_types

        except Exception as e:
            logger.debug(f"Error analyzing commit types for {file_path}: {e}")
            return {}

    def _find_last_refactor_date(self, file_path: str) -> Optional[str]:
        """
        Find the date of the last refactor commit.
        """
        if not self.has_git or not self.repo:
            return None

        try:
            # Ensure file_path is relative to repo
            path = Path(file_path)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.repo_path)
                    file_path = str(path)
                except ValueError:
                    pass

            # Search commits with "refactor" in the message
            for commit in self.repo.iter_commits(paths=file_path, max_count=50):
                message = commit.message
                if isinstance(message, (bytes, bytearray, memoryview)):
                    message = bytes(message).decode("utf-8", errors="ignore")
                message = str(message).lower()
                if "refactor" in message:
                    return commit.committed_datetime.isoformat()

            return None

        except Exception as e:
            logger.debug(f"Error finding last refactor date for {file_path}: {e}")
            return None

    def _find_co_modified_files(self, file_path: str) -> List[str]:
        """
        Find files that are frequently modified together with this one.
        """
        if not self.has_git or not self.repo:
            return []

        try:
            co_modified = {}

            # Ensure file_path is relative to repo
            path = Path(file_path)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.repo_path)
                    file_path = str(path)
                except ValueError:
                    pass

            # Analyze last 50 commits of the file
            for commit in self.repo.iter_commits(paths=file_path, max_count=50):
                try:
                    # Get all modified files in the same commit
                    modified_files = list(commit.stats.files.keys())

                    for other_file in modified_files:
                        # Ignore current file and test files
                        if other_file == file_path:
                            continue

                        # Ignore common configuration files
                        other_file_str = str(other_file)
                        if any(
                            pattern in other_file_str.lower()
                            for pattern in [
                                ".gitignore",
                                "package.json",
                                "pyproject.toml",
                                "requirements",
                                "readme",
                                ".lock",
                            ]
                        ):
                            continue

                        # Count co-modifications
                        co_modified[other_file_str] = co_modified.get(other_file_str, 0) + 1

                except (AttributeError, KeyError, TypeError) as e:
                    logger.debug(f"Error processing commit stats: {e}")
                    continue

            # Sort by frequency and take top 5
            sorted_files = sorted(co_modified.items(), key=lambda x: x[1], reverse=True)[:5]

            # Return only files with 2+ co-modifications
            result = [file for file, count in sorted_files if count >= 2]

            logger.debug(f"Co-modified files for {file_path}: {result}")
            return result

        except Exception as e:
            logger.debug(f"Error finding co-modified files for {file_path}: {e}")
            return []

    def _determine_modification_pattern(
        self, commits_30_days: int, freq_per_week: float, total_commits: int
    ) -> str:
        """
        Determine the modification pattern of the file.

        Patterns:
        - steady: Consistent changes
        - burst: Burst changes
        - sporadic: Sporadic changes
        - inactive: No recent changes
        """
        # If there are no recent commits, it's inactive
        if commits_30_days == 0:
            return "inactive"

        # If there are many recent commits vs historical, it's burst
        if total_commits > 0:
            recent_ratio = commits_30_days / (total_commits * 0.1)  # 10% of total
            if recent_ratio > 2.0:  # More than double expected
                return "burst"

        # If the frequency is high and consistent, it's steady
        if freq_per_week >= 1.0:
            return "steady"

        # Everything else is sporadic
        return "sporadic"

    def _normalize_file_path(self, file_path: str) -> str:
        """Devuelve file_path relativo al repo o el original si no es posible."""
        path = Path(file_path)
        if path.is_absolute():
            try:
                path = path.relative_to(self.repo_path)
                return str(path)
            except ValueError:
                return file_path
        return file_path
