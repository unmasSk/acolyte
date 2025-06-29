"""
Dream Analyzer - Deep code analysis engine.

Performs comprehensive analysis of the codebase using extended context windows.
Implements sliding window strategy for 32k models and full context for 128k.
"""

from typing import Dict, Any, List, Optional, cast
import json
from pathlib import Path
from enum import Enum

from acolyte.core.logging import logger
from acolyte.core.ollama import OllamaClient
from acolyte.core.secure_config import Settings
from acolyte.core.exceptions import AcolyteError
from acolyte.rag.retrieval.hybrid_search import HybridSearch, SearchFilters
from acolyte.rag.graph import NeuralGraph
from acolyte.embeddings import get_embeddings
from acolyte.core.utils.file_types import FileTypeDetector


class AnalysisCapability(Enum):
    """Analysis capability levels based on available components."""

    FULL = "FULL"  # All components available (search, embeddings, graph)
    LIMITED = "LIMITED"  # Some components missing (e.g., no search)
    MINIMAL = "MINIMAL"  # Core functionality only


class DreamWindowManager:
    """
    Manages context windows for Dream analysis.

    Implements sliding window for 32k models and full context for 128k+.
    """

    def __init__(self, context_size: int, config: Optional[Settings] = None):
        """
        Initialize window manager.

        Args:
            context_size: Total model context size
            config: Optional configuration settings
        """
        self.context_size = context_size

        # Load configuration or use defaults
        if config is None:
            config = Settings()

        # Get usable context ratio from config
        usable_ratio = config.get("dream.analysis.usable_context_ratio", 0.9)
        self.usable_context = int(context_size * usable_ratio)

        # Determine model size category
        if context_size <= 32768:
            size_category = "32k"
        elif context_size <= 65536:
            size_category = "64k"
        else:
            size_category = "128k+"

        # Get window configuration for this model size
        window_config = config.get(f"dream.analysis.window_sizes.{size_category}", {})

        if window_config:
            self.strategy = window_config.get("strategy", "sliding_window")

            if size_category == "128k+":
                system_reserve = window_config.get("system_reserve", 5000)
                self.new_code_size = self.usable_context - system_reserve
                self.preserved_context_size = system_reserve
            else:
                self.new_code_size = window_config.get("new_code_size", 28000)
                self.preserved_context_size = window_config.get("preserved_context_size", 1500)
        else:
            # Fallback to hardcoded defaults if config missing
            logger.warning(
                "Dream window configuration not found, using defaults", size_category=size_category
            )
            if context_size <= 32768:
                self.strategy = "sliding_window"
                self.new_code_size = 27000  # Reduced from 28000 to fit within usable_context
                self.preserved_context_size = 1500
            elif context_size <= 65536:
                self.strategy = "sliding_window"
                self.new_code_size = 55000  # Reduced from 58000 to fit within usable_context
                self.preserved_context_size = 3000
            else:
                self.strategy = "single_pass"
                self.new_code_size = self.usable_context - 5000
                self.preserved_context_size = 5000

        self.critical_context: List[Any] = []  # Preserved between cycles

        logger.info(
            "Dream window manager initialized",
            strategy=self.strategy,
            context_size=context_size,
            new_code_size=self.new_code_size,
            usable_ratio=usable_ratio,
            size_category=size_category,
        )

    def prepare_next_cycle(self, previous_findings: Dict[str, Any]) -> str:
        """
        Prepare context for next analysis cycle.

        Args:
            previous_findings: Findings from previous cycle

        Returns:
            Compressed critical context
        """
        if self.strategy == "single_pass":
            return ""  # No need to preserve in single pass

        # Extract critical findings
        critical = self._extract_critical_findings(previous_findings)

        # Compress to token limit
        compressed_context = self._compress_to_tokens(critical, self.preserved_context_size)

        return compressed_context

    def _extract_critical_findings(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Extract most critical findings to preserve."""
        critical: Dict[str, Any] = {"must_fix": [], "patterns": [], "relations": {}}

        # Priority 1: Bugs and vulnerabilities
        if "bugs" in findings:
            critical["must_fix"].extend(findings["bugs"])
        if "security_issues" in findings:
            critical["must_fix"].extend(findings["security_issues"])

        # Priority 2: Architectural issues
        if "architectural_issues" in findings:
            critical["patterns"] = findings["architectural_issues"][:3]

        # Priority 3: Critical relationships
        if "file_dependencies" in findings:
            # Keep only strong dependencies
            critical["relations"] = {
                k: v for k, v in findings["file_dependencies"].items() if v.get("strength", 0) > 0.7
            }

        return critical

    def _compress_to_tokens(self, data: Dict[str, Any], max_tokens: int) -> str:
        """Compress data to fit in token limit."""
        # Simple JSON representation for now
        # Could use more sophisticated compression
        context = json.dumps(data, indent=2)

        # Get chars per token estimate from config
        config = Settings()
        chars_per_token = config.get("dream.analysis.chars_per_token", 4)

        # Truncate if too long (crude but effective)
        # In production, use proper token counting
        max_chars = max_tokens * chars_per_token
        if len(context) > max_chars:
            context = context[:max_chars] + "..."

        return context


class DreamAnalyzer:
    """
    Core analysis engine for Dream system.

    Performs deep analysis using extended context windows to find:
    - Hidden bugs and vulnerabilities
    - Performance bottlenecks
    - Architectural issues
    - Code patterns and anti-patterns
    """

    def __init__(self, weaviate_client: Optional[Any] = None) -> None:
        """
        Initialize analyzer with dependencies.

        Args:
            weaviate_client: Optional Weaviate client. If not provided,
                           search functionality will be limited.
        """
        self.config = Settings()
        self.ollama = OllamaClient()
        self.neural_graph = NeuralGraph()

        # Initialize search and embeddings
        if weaviate_client:
            self.search = HybridSearch(weaviate_client=weaviate_client)
        else:
            self.search = None
            logger.warning(
                "DreamAnalyzer initialized without Weaviate client. "
                "Search functionality will be limited."
            )

        # Initialize embeddings (doesn't require external dependencies)
        try:
            self.embeddings = get_embeddings()
        except Exception as e:
            logger.error("Failed to initialize embeddings", error=str(e))
            self.embeddings = None

        # Get context configuration
        self.context_size = self.config.get("model.context_size", 32768)
        self.window_manager = DreamWindowManager(self.context_size, self.config)

        # Analysis prompts
        self._load_analysis_prompts()

        # Determine capability level
        self.capability = self._determine_capability_level()

        logger.info(
            "Dream analyzer initialized",
            module="dream",
            has_search=self.search is not None,
            has_embeddings=self.embeddings is not None,
            capability_level=self.capability.value,
        )

    def _determine_capability_level(self) -> AnalysisCapability:
        """Determine the analysis capability level based on available components.

        Returns:
            AnalysisCapability enum value
        """
        if self.search and self.embeddings and self.neural_graph:
            return AnalysisCapability.FULL
        elif self.search or self.embeddings:
            return AnalysisCapability.LIMITED
        else:
            return AnalysisCapability.MINIMAL

    def get_capability_info(self) -> Dict[str, Any]:
        """Get detailed information about current capabilities.

        Returns:
            Dict with capability level and available features
        """
        return {
            "level": self.capability.value,
            "features": {
                "search": self.search is not None,
                "embeddings": self.embeddings is not None,
                "neural_graph": self.neural_graph is not None,
                "recent_files_tracking": self.search is not None,
                "semantic_analysis": self.embeddings is not None,
                "impact_prediction": self.neural_graph is not None,
            },
            "limitations": self._get_limitations(),
        }

    def _get_limitations(self) -> List[str]:
        """Get list of current limitations based on capability level.

        Returns:
            List of limitation descriptions
        """
        limitations = []

        if not self.search:
            limitations.append("Cannot track recently changed files - using default file selection")
            limitations.append("No semantic search - file selection based on basic patterns")

        if not self.embeddings:
            limitations.append("Cannot perform semantic similarity analysis")
            limitations.append("Code understanding limited to syntax parsing")

        if not self.neural_graph:
            limitations.append("Cannot predict impact of changes across codebase")
            limitations.append("No dependency tracking between files")

        if self.capability == AnalysisCapability.MINIMAL:
            limitations.append("Operating in minimal mode - basic analysis only")

        return limitations

    def _load_analysis_prompts(self) -> None:
        """Load specialized analysis prompts from files.

        Prompts can be loaded from:
        1. Default location: dream/prompts/*.md
        2. Custom directory specified in config
        3. Inline in config (override specific prompts)
        """
        self.prompts = {}

        # Get prompts directory from config or use default
        prompts_dir = self.config.get("dream.prompts_directory", None)
        if prompts_dir:
            prompts_path = Path(prompts_dir)
            if not prompts_path.is_absolute():
                # Relative to project root
                project_path = Path(self.config.get("project.path", "."))
                prompts_path = project_path / prompts_path
        else:
            # Use default location
            prompts_path = Path(__file__).parent / "prompts"

        # Define expected prompt files
        prompt_files = [
            "bug_detection",
            "security_analysis",
            "performance_analysis",
            "architecture_analysis",
            "pattern_detection",
        ]

        # Load prompts from files
        for prompt_name in prompt_files:
            prompt_file = prompts_path / f"{prompt_name}.md"

            try:
                if prompt_file.exists():
                    self.prompts[prompt_name] = prompt_file.read_text(encoding="utf-8")
                    logger.debug(
                        "Loaded prompt from file", prompt=prompt_name, file=str(prompt_file)
                    )
                else:
                    logger.warning(
                        "Prompt file not found", prompt=prompt_name, expected_path=str(prompt_file)
                    )
                    # Use empty prompt as fallback
                    self.prompts[prompt_name] = "Error: Prompt file not found"
            except Exception as e:
                logger.error("Failed to load prompt file", prompt=prompt_name, error=str(e))
                self.prompts[prompt_name] = "Error: Failed to load prompt"

        # Allow config overrides for specific prompts
        config_prompts = self.config.get("dream.prompts", {})
        for prompt_name, prompt_content in config_prompts.items():
            if prompt_name in self.prompts:
                self.prompts[prompt_name] = prompt_content
                logger.info("Prompt overridden from config", prompt=prompt_name)

        # Validate all prompts have required placeholders
        for prompt_name, prompt_content in self.prompts.items():
            if "{code}" not in prompt_content:
                logger.warning("Prompt missing {code} placeholder", prompt=prompt_name)

        logger.info("Analysis prompts loaded", total=len(self.prompts), source=str(prompts_path))

    async def explore_codebase(
        self, focus_areas: Optional[List[str]] = None, context_size: int = 32768
    ) -> Dict[str, Any]:
        """
        Initial exploration phase (DREAMING state).

        Args:
            focus_areas: Specific areas to focus on
            context_size: Available context window

        Returns:
            Initial findings and areas of interest
        """
        try:
            logger.info(
                "Starting codebase exploration", focus_areas=focus_areas, context_size=context_size
            )

            # Get relevant files based on focus areas
            files_to_analyze = await self._select_files_for_analysis(focus_areas)

            # For single pass strategy, analyze everything at once
            if self.window_manager.strategy == "single_pass":
                results = await self._analyze_all_files(files_to_analyze)
                results["capability_info"] = self.get_capability_info()
                return results

            # For sliding window, analyze in cycles
            findings: Dict[str, Any] = {
                "overview": {},
                "areas_of_concern": [],
                "files_analyzed": [],
                "patterns_detected": [],
                "requires_deep_analysis": [],
                "capability_info": self.get_capability_info(),
            }

            # First cycle: Get overview
            overview_files = files_to_analyze[:20]  # Start with top files
            code_context = await self._load_code_context(overview_files)

            # Run architectural analysis for overview
            arch_analysis = await self._run_analysis("architecture_analysis", code_context, "")

            if arch_analysis:
                findings["overview"] = arch_analysis
                # Ensure list type before extending
                if isinstance(findings["files_analyzed"], list):
                    findings["files_analyzed"].extend([f["path"] for f in overview_files])

                # Identify areas that need deep analysis
                if "architectural_issues" in arch_analysis and isinstance(
                    findings["requires_deep_analysis"], list
                ):
                    for issue in arch_analysis.get("architectural_issues", []):
                        # Check severity in the issue dict
                        severity = issue.get("severity", "MEDIUM").upper()
                        if severity in ["HIGH", "CRITICAL"]:
                            findings["areas_of_concern"].append(issue)
                            if "files" in issue:
                                findings["requires_deep_analysis"].extend(issue["files"])

            return findings

        except Exception as e:
            logger.error(f"Codebase exploration failed: {e}")
            raise AcolyteError(f"Exploration failed: {e}")

    async def analyze_deeply(
        self, initial_findings: Dict[str, Any], priorities: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Deep analysis phase (REM state).

        Args:
            initial_findings: Findings from exploration phase
            priorities: Analysis priorities (bugs, security, etc.)

        Returns:
            Deep analysis results
        """
        try:
            logger.info("Starting deep analysis", priorities=priorities)

            results: Dict[str, Any] = {
                "bugs": [],
                "security_issues": [],
                "performance_issues": [],
                "architectural_issues": [],
                "patterns": [],
                "recommendations": [],
            }

            # Get files that need deep analysis
            files_to_analyze = initial_findings.get("requires_deep_analysis", [])
            if not files_to_analyze:
                # Fall back to analyzing changed files
                file_paths, is_fallback = await self._get_analysis_candidates()
                if is_fallback:
                    logger.info(
                        "Using fallback file selection for deep analysis",
                        reason="no specific files identified",
                        capability_level="LIMITED",
                    )
                files_to_analyze = [{"path": fp} for fp in file_paths]

            # Prepare for sliding window if needed
            cycles_needed = 1
            if self.window_manager.strategy == "sliding_window":
                # Get average tokens per file from config
                avg_tokens_per_file = self.config.get("dream.analysis.avg_tokens_per_file", None)
                if avg_tokens_per_file is None:
                    avg_tokens_per_file = 1000
                    logger.debug(
                        "Using default avg_tokens_per_file",
                        value=avg_tokens_per_file,
                        hint="Configure dream.analysis.avg_tokens_per_file for better estimates",
                    )

                files_per_cycle = self.window_manager.new_code_size // avg_tokens_per_file
                cycles_needed = max(1, len(files_to_analyze) // files_per_cycle + 1)

            logger.info(
                "Deep analysis will run cycles",
                cycles_count=cycles_needed,
                total_files=len(files_to_analyze),
            )

            # Run analysis cycles
            preserved_context = ""
            all_cycle_results: List[Dict[str, Any]] = []

            for cycle in range(cycles_needed):
                # Select files for this cycle
                start_idx = cycle * (len(files_to_analyze) // cycles_needed)
                end_idx = (cycle + 1) * (len(files_to_analyze) // cycles_needed)

                # Handle list of strings or dicts
                cycle_items = files_to_analyze[start_idx:end_idx]
                cycle_files = []
                for item in cycle_items:
                    if isinstance(item, str):
                        cycle_files.append({"path": item})
                    elif isinstance(item, dict):
                        cycle_files.append(item)

                if not cycle_files:
                    continue

                logger.info(
                    "Analysis cycle",
                    cycle_number=cycle + 1,
                    total_cycles=cycles_needed,
                    files_count=len(cycle_files),
                )

                # Load code for this cycle
                code_chunks = await self._load_code_context(cycle_files)

                # Run prioritized analyses
                cycle_results = await self._run_prioritized_analyses(
                    code_chunks, preserved_context, priorities
                )
                all_cycle_results.append(cycle_results)

                # Prepare context for next cycle
                if cycle < cycles_needed - 1:
                    preserved_context = self.window_manager.prepare_next_cycle(cycle_results)

            # Merge all results after cycles are complete
            for cycle_res in all_cycle_results:
                for key, value in cycle_res.items():
                    if key in results and isinstance(results[key], list):
                        if isinstance(value, list):
                            results[key].extend(value)
                        else:
                            results[key].append(value)
                    elif key not in results:
                        results[key] = value

            # Post-process results
            results = self._deduplicate_findings(results)
            results = self._prioritize_findings(results)

            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results)

            # Add capability information
            results["capability_info"] = self.get_capability_info()

            return results

        except Exception as e:
            logger.error(f"Deep analysis failed: {e}")
            raise AcolyteError(f"Deep analysis failed: {e}")

    async def _select_files_for_analysis(
        self, focus_areas: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Select files to analyze based on focus areas."""
        files = []

        if focus_areas and self.search:
            # Search for files matching focus areas
            for area in focus_areas:
                # Create search filters
                filters = SearchFilters()

                # Get search results
                scored_chunks = await self.search.search(query=area, max_chunks=50, filters=filters)

                # Extract unique files from scored chunks
                seen_paths = set()
                for scored_chunk in scored_chunks:
                    chunk = scored_chunk.chunk
                    file_path = chunk.metadata.file_path

                    if file_path not in seen_paths:
                        seen_paths.add(file_path)
                        file_info = {
                            "path": file_path,
                            "relevance": scored_chunk.score,
                            "language": chunk.metadata.language,
                        }
                        files.append(file_info)
        else:
            # Fallback when search is not available
            if not self.search:
                logger.info("Using default project files due to missing search capability")
            file_paths, is_fallback = await self._get_analysis_candidates()
            if is_fallback:
                logger.info(
                    "Operating with LIMITED capabilities - using fallback file selection",
                    files_count=len(file_paths),
                )
            # Infer language from file extension
            for fp in file_paths:
                ext = Path(fp).suffix.lower()
                language = self._infer_language_from_extension(ext)
                files.append({"path": fp, "relevance": 0.5, "language": language})

        # Sort by relevance
        files.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        return files[:100]  # Limit to top 100 files

    async def _get_analysis_candidates(self) -> tuple[List[str], bool]:
        """Get candidate files for analysis.

        Returns:
            tuple: (list of file paths, is_fallback)
                - is_fallback=True means using default files without recency info
                - is_fallback=False means using actual recently changed files
        """
        if not self.search:
            logger.info(
                "Search capability unavailable - using fallback file selection",
                module="dream",
            )
            # Return default files with fallback indicator
            default_files = await self._get_default_project_files()
            return (default_files, True)

        # Get files with recent changes from search
        recent_files = await self._get_recent_files_from_search()
        return (recent_files, False)

    async def _get_recent_files_from_search(self) -> List[str]:
        """Get files with recent changes using search capability.

        Returns files that have been modified recently, prioritizing:
        - Files modified in the last 30 days
        - Files with high change frequency
        - Files with many recent commits
        """
        if not self.search:
            # This should not happen as caller should check first
            logger.error("Called _get_recent_files_from_search without search capability")
            return []

        try:
            # Create filters for recent activity
            filters = SearchFilters()

            # Query for chunks with recent modifications
            # Use a broad query to get diverse results
            query = "recent changes modified updated"

            # Get more chunks to ensure we find enough unique files
            scored_chunks = await self.search.search(
                query=query, max_chunks=200, filters=filters  # Get more chunks to find unique files
            )

            # Extract unique files with their metadata
            file_activity = {}  # file_path -> activity_score

            for scored_chunk in scored_chunks:
                chunk = scored_chunk.chunk
                file_path = chunk.metadata.file_path

                # Calculate activity score based on metadata
                if file_path not in file_activity:
                    # Initialize with base score from search relevance
                    git_metadata = (
                        chunk.metadata.language_specific.get('git_metadata', {})
                        if chunk.metadata.language_specific
                        else {}
                    )
                    file_activity[file_path] = {
                        'path': file_path,
                        'score': scored_chunk.score,
                        'last_modified': git_metadata.get('last_modified'),
                        'commits_count': git_metadata.get('commits_last_30_days', 0),
                        'lines_changed': git_metadata.get('lines_changed_last_30_days', 0),
                    }
                else:
                    # Update score if we found a better chunk from same file
                    if scored_chunk.score > file_activity[file_path]['score']:
                        file_activity[file_path]['score'] = scored_chunk.score

            # Calculate composite activity score for each file
            from acolyte.core.utils.datetime_utils import utc_now, parse_iso_datetime
            from datetime import timedelta

            now = utc_now()
            thirty_days_ago = now - timedelta(days=30)

            for file_data in file_activity.values():
                # Parse last modified date
                recency_score = 0.0
                if file_data['last_modified']:
                    try:
                        last_mod = parse_iso_datetime(file_data['last_modified'])
                        if last_mod > thirty_days_ago:
                            # Linear decay from 1.0 (today) to 0.0 (30 days ago)
                            days_old = (now - last_mod).days
                            recency_score = max(0.0, 1.0 - (days_old / 30.0))
                    except Exception as e:
                        logger.debug("Could not parse date", error=str(e))

                # Normalize commit count (assume 10+ commits in 30 days is very active)
                commit_score = min(1.0, file_data['commits_count'] / 10.0)

                # Normalize lines changed (assume 500+ lines changed is very active)
                change_score = min(1.0, file_data['lines_changed'] / 500.0)

                # Composite score with weights
                file_data['activity_score'] = (
                    0.4 * recency_score  # 40% weight on recency
                    + 0.3 * commit_score  # 30% weight on commit frequency
                    + 0.2 * change_score  # 20% weight on change volume
                    + 0.1 * file_data['score']  # 10% weight on search relevance
                )

            # Sort by activity score and get top files
            sorted_files = sorted(
                file_activity.values(), key=lambda x: x['activity_score'], reverse=True
            )

            # Return file paths of most active files
            active_files = [f['path'] for f in sorted_files[:50]]  # Top 50 files

            logger.info(
                "Found recently changed files using search",
                total_files=len(active_files),
                top_file=active_files[0] if active_files else None,
                capability_level="FULL",
            )

            return active_files

        except Exception as e:
            logger.error("Failed to get recently changed files", error=str(e), module="dream")
            return []

    async def _load_code_context(self, files: List[Dict[str, Any]]) -> str:
        """Load code from files into context string."""
        context_parts = []

        for file_info in files:
            try:
                file_path = Path(file_info["path"])
                if file_path.exists():
                    content = file_path.read_text(encoding="utf-8")

                    # Add file header
                    context_parts.append(f"\n\n=== File: {file_path} ===\n")
                    context_parts.append(content)

                    # Add graph relationships
                    try:
                        impact = await self.neural_graph.predict_impact(str(file_path))
                        if impact["total_affected"] > 0:
                            context_parts.append(
                                f"\n// Dependencies: {impact['total_affected']} files depend on this"
                            )
                    except Exception:
                        pass

            except Exception as e:
                logger.warning("Could not load file", file_path=file_info['path'], error=str(e))

        return "\n".join(context_parts)

    async def _run_analysis(self, analysis_type: str, code: str, context: str) -> Dict[str, Any]:
        """Run specific analysis using Ollama."""
        prompt = self.prompts[analysis_type].format(code=code, context=context)

        try:
            response = await self.ollama.generate(
                prompt=prompt,
                system="You are an expert code analyzer. Respond only with valid JSON.",
                temperature=0.1,  # Low temperature for consistent analysis
            )

            # --- Asegurar que response es un string ---
            import collections.abc

            if isinstance(response, collections.abc.AsyncIterator):
                response = "".join([chunk async for chunk in response])

            # Parse JSON response
            try:
                # Use cast to inform mypy about the expected type
                return cast(Dict[str, Any], json.loads(response))
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re

                # First try to find JSON in markdown code block
                code_block_match = re.search(
                    r'```(?:json)?\s*\n?({.*?})\s*\n?```', response, re.DOTALL
                )
                if code_block_match:
                    try:
                        return cast(Dict[str, Any], json.loads(code_block_match.group(1)))
                    except json.JSONDecodeError:
                        pass

                # If not in code block, try to find raw JSON
                json_match = re.search(r'{.*}', response, re.DOTALL)
                if json_match:
                    try:
                        logger.info(
                            "[UNTESTED PATH] Parsing JSON from raw response without code block"
                        )
                        return cast(Dict[str, Any], json.loads(json_match.group()))
                    except json.JSONDecodeError:
                        logger.warning("[UNTESTED PATH] Failed to parse JSON even from raw match")
                        pass

                logger.warning("Could not parse JSON from analysis", analysis_type=analysis_type)
                return {}

        except Exception as e:
            logger.error("Analysis failed", analysis_type=analysis_type, error=str(e))
            return {}

    async def _run_prioritized_analyses(
        self, code: str, context: str, priorities: Dict[str, float]
    ) -> Dict[str, Any]:
        """Run analyses based on priorities."""
        results: Dict[str, Any] = {}

        # Map priorities to analysis types
        analysis_map = {
            "bugs": "bug_detection",
            "security": "security_analysis",
            "performance": "performance_analysis",
            "architecture": "architecture_analysis",
            "patterns": "pattern_detection",
        }

        # Sort by priority
        sorted_priorities = sorted(priorities.items(), key=lambda x: x[1], reverse=True)

        # Run analyses
        for priority_type, weight in sorted_priorities:
            if weight > 0 and priority_type in analysis_map:
                analysis_type = analysis_map[priority_type]

                logger.info("Running analysis", analysis_type=analysis_type, weight=weight)

                result = await self._run_analysis(analysis_type, code, context)

                # Merge results
                results.update(result)

        return results

    async def _analyze_all_files(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze all files in single pass (128k+ models)."""
        # Load all code at once
        code_context = await self._load_code_context(files)

        # Run all analyses
        results: Dict[str, Any] = {}

        for analysis_type in self.prompts:
            logger.info("Running analysis on full context", analysis_type=analysis_type)
            result = await self._run_analysis(analysis_type, code_context, "")
            results.update(result)

        return results

    def _deduplicate_findings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate findings across cycles."""
        for key in results:
            if isinstance(results[key], list) and len(results[key]) > 0:
                # Simple deduplication by converting to unique strings
                seen = set()
                unique = []

                for item in results[key]:
                    # Create unique key for item
                    if isinstance(item, dict):
                        key_str = json.dumps(item, sort_keys=True)
                        if key_str not in seen:
                            seen.add(key_str)
                            unique.append(item)
                    else:
                        if item not in seen:
                            seen.add(item)
                            unique.append(item)

                results[key] = unique

        return results

    def _prioritize_findings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Sort findings by severity/importance."""
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

        # Sort bugs by severity
        if "bugs" in results:
            results["bugs"].sort(key=lambda x: severity_order.get(x.get("severity", "LOW"), 4))

        # Sort security issues
        if "security_issues" in results:
            results["security_issues"].sort(
                key=lambda x: severity_order.get(x.get("severity", "LOW"), 4)
            )

        return results

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations from findings."""
        recommendations = []

        # Critical bugs
        critical_bugs = [
            b for b in results.get("bugs", []) if b.get("severity") in ["CRITICAL", "HIGH"]
        ]
        if critical_bugs:
            recommendations.append(f"Fix {len(critical_bugs)} critical bugs immediately")

        # Security issues
        security_issues = results.get("security_issues", [])
        if security_issues:
            recommendations.append(f"Address {len(security_issues)} security vulnerabilities")

        # Performance
        perf_issues = results.get("performance_issues", [])
        if perf_issues:
            top_impact = sorted(perf_issues, key=lambda x: x.get("impact", ""), reverse=True)[:3]
            for issue in top_impact:
                recommendations.append(
                    f"Optimize {issue.get('file', 'code')}: {issue.get('issue', '')}"
                )

        # Architecture
        arch_issues = results.get("architectural_issues", [])
        if arch_issues:
            recommendations.append("Consider architectural refactoring for better maintainability")

        return recommendations

    async def _get_default_project_files(self) -> List[str]:
        """Get default project files when search is not available.

        Returns a basic set of important files from the project root.
        This is a fallback when Weaviate search is not available.
        """
        try:
            project_path = Path(self.config.get("project.path", "."))
            important_patterns = [
                "*.py",  # Python files
                "*.js",  # JavaScript files
                "*.ts",  # TypeScript files
                "*.java",  # Java files
                "*.go",  # Go files
                "*.rs",  # Rust files
                "*.cpp",  # C++ files
                "*.c",  # C files
                "*.h",  # Header files
                "*.cs",  # C# files
                "README*",  # README files
                "*.md",  # Markdown files
            ]

            all_files: List[Path] = []
            # Get files from project root and immediate subdirectories
            for pattern in important_patterns:
                # Root level
                all_files.extend(project_path.glob(pattern))
                # One level deep
                all_files.extend(project_path.glob(f"*/{pattern}"))

            # Convert to relative paths and filter
            relative_files = []
            for file_path in all_files:
                if file_path.is_file():
                    try:
                        rel_path = file_path.relative_to(project_path)
                        # Skip common ignore patterns
                        skip_patterns = [".git", "__pycache__", "node_modules", "venv", ".venv"]
                        if not any(skip in str(rel_path) for skip in skip_patterns):
                            relative_files.append(str(rel_path))
                    except ValueError:
                        continue

            # Limit to reasonable number
            if len(relative_files) > 50:
                logger.info("Found files, limiting to 50", found_count=len(relative_files))
                relative_files = relative_files[:50]

            logger.info("Using default project files for analysis", file_count=len(relative_files))
            return relative_files

        except Exception as e:
            logger.error("Failed to get default project files", error=str(e))
            return []

    def _infer_language_from_extension(self, extension: str) -> str:
        """Infer programming language from file extension.

        Args:
            extension: File extension (e.g., '.py', '.js')

        Returns:
            Language name or 'unknown'
        """
        # Create a temporary Path with the extension to use FileTypeDetector
        temp_path = Path(f"temp{extension}")
        return FileTypeDetector.get_language(temp_path)
