"""
Dream Insight Writer - Generates and stores analysis insights.

Writes insights to both database and filesystem in human-readable format.
Creates actionable documents that developers can use immediately.
"""

from typing import Dict, Any, List, Optional
from datetime import timedelta
from pathlib import Path
import json
import re
import unicodedata

from acolyte.core.logging import logger
from acolyte.core.database import get_db_manager
from acolyte.core.id_generator import generate_id
from acolyte.core.secure_config import Settings
from acolyte.core.exceptions import DatabaseError
from acolyte.core.utils.datetime_utils import utc_now, utc_now_iso


class InsightWriter:
    """
    Writes Dream analysis insights to storage.

    Handles:
    - Database persistence (dream_insights table)
    - Filesystem documents (.acolyte-dreams/)
    - Formatting and organization
    - Summary generation
    """

    def __init__(self) -> None:
        """Initialize insight writer."""
        self.db = get_db_manager()
        self.config = Settings()

        # Get dream folder configuration
        self.dream_folder = self.config.get("dream.dream_folder_name", "dreams")
        self.keep_days = self.config.get("dream.keep_sessions_days", 90)

        # Ensure dream folder exists in correct location
        self.dream_path = self._get_dream_path()
        self.dream_path.mkdir(parents=True, exist_ok=True)

        logger.info("Dream insight writer initialized", module="dream")

    def _get_dream_path(self) -> Path:
        """Get the correct path for dream insights.

        CLEAN PROJECT ARCHITECTURE:
        - If .acolyte.project exists: use ~/.acolyte/projects/{id}/data/dreams/
        - Otherwise (during development): use ./data/dreams/
        """
        # Check if we're in a configured project
        project_file = Path.cwd() / ".acolyte.project"

        if project_file.exists():
            try:
                import json

                with open(project_file) as f:
                    project_data = json.load(f)
                    project_id = project_data.get("project_id")

                if project_id:
                    # Use global project directory
                    global_dream_dir = (
                        Path.home()
                        / ".acolyte"
                        / "projects"
                        / project_id
                        / "data"
                        / self.dream_folder
                    )
                    return global_dream_dir
            except Exception as e:
                logger.warning("Failed to read project file, using local data", error=str(e))

        # Fallback for development
        return Path("data") / self.dream_folder

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize filename to prevent security issues.

        Args:
            name: Raw filename component

        Returns:
            Safe filename string
        """
        # Remove any path components to prevent traversal
        name = Path(name).name

        # Normalize unicode and convert to ASCII
        name = unicodedata.normalize('NFKD', name)
        name = name.encode('ascii', 'ignore').decode('ascii')

        # Replace spaces with underscores
        name = name.replace(' ', '_')

        # Keep only alphanumeric, underscore, hyphen, and dot
        # This is a strict whitelist approach
        name = re.sub(r'[^a-zA-Z0-9_\-\.]', '', name)

        # Remove multiple consecutive dots to prevent ../ patterns
        name = re.sub(r'\.{2,}', '_', name)

        # Remove leading/trailing dots and underscores
        name = name.strip('._')

        # Ensure not empty
        if not name:
            name = "unknown"

        # Limit length to prevent filesystem issues
        max_length = 100  # Safe for most filesystems
        if len(name) > max_length:
            # Keep extension if present
            if '.' in name:
                base, ext = name.rsplit('.', 1)
                base = base[: max_length - len(ext) - 1]
                name = f"{base}.{ext}"
            else:
                name = name[:max_length]

        return name.lower()

    async def write_insights(
        self,
        session_id: str,
        insights: List[Dict[str, Any]],
        focus_areas: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Write insights to storage.

        Args:
            session_id: Dream session ID
            insights: List of insights from analysis
            focus_areas: Original focus areas for context

        Returns:
            Summary of written insights
        """
        try:
            logger.info("Writing insights", session_id=session_id, count=len(insights))

            # Categorize insights
            categorized = self._categorize_insights(insights)

            # Write to database
            db_count = await self._write_to_database(session_id, insights)

            # Generate document
            doc_path = await self._write_document(session_id, categorized, focus_areas)

            # Update summary
            await self._update_summary(categorized)

            # Clean old insights if configured
            if self.config.get("dream.auto_cleanup", True):
                await self._cleanup_old_insights()

            return {
                "session_id": session_id,
                "insights_written": db_count,
                "document_path": str(doc_path),
                "summary": self._generate_summary_stats(categorized),
            }

        except Exception as e:
            logger.error("Failed to write insights", error=str(e))
            raise DatabaseError(f"Insight writing failed: {e}")

    def _categorize_insights(
        self, insights: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize insights by type.

        Args:
            insights: Raw insights list

        Returns:
            Dict with categorized insights
        """
        categorized: Dict[str, List[Dict[str, Any]]] = {
            "bugs": [],
            "security": [],
            "performance": [],
            "architecture": [],
            "patterns": [],
            "other": [],
        }

        for insight in insights:
            # Determine category
            if "bugs" in insight:
                categorized["bugs"].extend(insight["bugs"])
            elif "security_issues" in insight:
                categorized["security"].extend(insight["security_issues"])
            elif "performance_issues" in insight:
                categorized["performance"].extend(insight["performance_issues"])
            elif "architectural_issues" in insight:
                categorized["architecture"].extend(insight["architectural_issues"])
            elif "patterns" in insight:
                categorized["patterns"].extend(insight["patterns"])
            else:
                # Single insight format
                insight_type = insight.get("type", "").lower()
                if "bug" in insight_type:
                    categorized["bugs"].append(insight)
                elif "security" in insight_type:
                    categorized["security"].append(insight)
                elif "performance" in insight_type:
                    categorized["performance"].append(insight)
                elif "architect" in insight_type:
                    categorized["architecture"].append(insight)
                elif "pattern" in insight_type:
                    categorized["patterns"].append(insight)
                else:
                    categorized["other"].append(insight)

        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}

    async def _write_to_database(self, session_id: str, insights: List[Dict[str, Any]]) -> int:
        """
        Write insights to database.

        Args:
            session_id: Dream session ID
            insights: Insights to store

        Returns:
            Number of insights written
        """
        count = 0

        for insight in insights:
            # Handle both nested and flat formats
            if "bugs" in insight:
                for bug in insight["bugs"]:
                    await self._insert_insight(session_id, "BUG_RISK", bug)
                    count += 1
            elif "security_issues" in insight:
                for issue in insight["security_issues"]:
                    await self._insert_insight(session_id, "PATTERN", issue)
                    count += 1
            elif "performance_issues" in insight:
                for issue in insight["performance_issues"]:
                    await self._insert_insight(session_id, "OPTIMIZATION", issue)
                    count += 1
            elif "architectural_issues" in insight:
                for issue in insight["architectural_issues"]:
                    await self._insert_insight(session_id, "ARCHITECTURE", issue)
                    count += 1
            elif "patterns" in insight:
                for pattern in insight["patterns"]:
                    await self._insert_insight(session_id, "PATTERN", pattern)
                    count += 1
            else:
                # Single insight
                insight_type = self._map_insight_type(insight.get("type", ""))
                await self._insert_insight(session_id, insight_type, insight)
                count += 1

        return count

    async def _insert_insight(
        self, session_id: str, insight_type: str, data: Dict[str, Any]
    ) -> None:
        """Insert single insight into database."""
        insight_id = generate_id()

        # Extract common fields
        title = data.get("title") or data.get("issue") or data.get("description", "")[:100]

        description = data.get("description") or data.get("issue") or data.get("message", "")

        # Determine confidence and impact
        confidence = self._calculate_confidence(data)
        impact = self._determine_impact(data)

        # Extract entities and references
        entities = self._extract_entities(data)
        references = self._extract_references(data)

        try:
            await self.db.execute_async(
                """
                INSERT INTO dream_insights (
                    id, session_id, insight_type, title, description,
                    confidence, impact, entities_involved, code_references
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    insight_id,
                    session_id,
                    insight_type,
                    title,
                    description,
                    confidence,
                    impact,
                    json.dumps(entities),
                    json.dumps(references),
                ),
            )
        except Exception as e:
            logger.error("Failed to insert insight", error=str(e))

    def _map_insight_type(self, type_str: str) -> str:
        """Map insight type to database enum."""
        type_lower = type_str.lower()

        if "bug" in type_lower:
            return "BUG_RISK"
        elif "pattern" in type_lower:
            return "PATTERN"
        elif "connect" in type_lower:
            return "CONNECTION"
        elif "optim" in type_lower:
            return "OPTIMIZATION"
        elif "architect" in type_lower:
            return "ARCHITECTURE"
        else:
            return "PATTERN"  # Default

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score for insight."""
        # If confidence is provided, use it
        if "confidence" in data:
            return float(data["confidence"])

        # Otherwise, derive from severity/impact
        severity = data.get("severity", "").upper()
        if severity in ["CRITICAL", "HIGH"]:
            return 0.9
        elif severity == "MEDIUM":
            return 0.7
        else:
            return 0.5

    def _determine_impact(self, data: Dict[str, Any]) -> str:
        """Determine impact level."""
        # Check for explicit impact
        if "impact" in data:
            impact_str = str(data["impact"]).upper()
            if impact_str in ["HIGH", "MEDIUM", "LOW"]:
                return impact_str

        # Derive from severity
        severity = data.get("severity", "").upper()
        if severity in ["CRITICAL", "HIGH"]:
            return "HIGH"
        elif severity == "MEDIUM":
            return "MEDIUM"
        else:
            return "LOW"

    def _extract_entities(self, data: Dict[str, Any]) -> List[str]:
        """Extract entities (files, classes, functions) from insight."""
        entities = []

        # Common fields that contain entities
        for field in ["file", "files", "file_path", "path", "class", "function"]:
            if field in data:
                value = data[field]
                if isinstance(value, list):
                    entities.extend(value)
                elif value:
                    entities.append(str(value))

        return list(set(entities))  # Unique

    def _extract_references(self, data: Dict[str, Any]) -> List[str]:
        """Extract code references from insight."""
        references = []

        # Line numbers
        if "line" in data and "file" in data:
            references.append(f"{data['file']}:{data['line']}")

        # Code references field
        if "code_references" in data:
            refs = data["code_references"]
            if isinstance(refs, list):
                references.extend(refs)

        # Occurrences
        if "occurrences" in data:
            for occ in data.get("occurrences", []):
                if isinstance(occ, dict) and "file" in occ:
                    ref = occ["file"]
                    if "line" in occ:
                        ref += f":{occ['line']}"
                    references.append(ref)

        return list(set(references))

    async def _write_document(
        self,
        session_id: str,
        categorized: Dict[str, List[Dict[str, Any]]],
        focus_areas: Optional[List[str]],
    ) -> Path:
        """
        Write markdown document with insights.

        Args:
            session_id: Dream session ID
            categorized: Categorized insights
            focus_areas: Original focus areas

        Returns:
            Path to written document
        """
        # Generate filename
        timestamp = utc_now().strftime("%Y-%m-%d_%H-%M")

        # Determine document type with proper sanitization
        if focus_areas:
            # Sanitize the focus area to create safe filename
            doc_type = self._sanitize_filename(focus_areas[0])
        else:
            doc_type = "general_analysis"

        # Construct filename with sanitized components
        filename = f"{timestamp}_{doc_type}.md"
        doc_path = self.dream_path / filename

        # Generate content
        content = self._generate_markdown(session_id, categorized, focus_areas)

        # Write file
        doc_path.write_text(content, encoding="utf-8")

        logger.info("Dream document written", path=str(doc_path))

        return doc_path

    def _generate_markdown(
        self,
        session_id: str,
        categorized: Dict[str, List[Dict[str, Any]]],
        focus_areas: Optional[List[str]],
    ) -> str:
        """Generate markdown document content."""
        lines = []

        # Header
        lines.append("# DeepDream Analysis Report")
        lines.append("")
        lines.append(f"**Date**: {utc_now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Session**: {session_id}")
        lines.append(
            f"**Model**: acolyte:latest ({self.config.get('model.context_size', 32768)} tokens)"
        )

        if focus_areas:
            lines.append(f"**Focus Areas**: {', '.join(focus_areas)}")

        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.extend(self._generate_executive_summary(categorized))
        lines.append("")

        # Critical Findings
        if categorized.get("bugs") or categorized.get("security"):
            lines.append("## 🚨 Critical Findings")
            lines.append("")

            # Bugs
            critical_bugs = [
                b
                for b in categorized.get("bugs", [])
                if b.get("severity", "").upper() in ["CRITICAL", "HIGH"]
            ]
            if critical_bugs:
                lines.append("### Critical Bugs")
                lines.append("")
                for bug in critical_bugs[:5]:  # Top 5
                    lines.extend(self._format_bug(bug))
                    lines.append("")

            # Security
            security_issues = categorized.get("security", [])
            if security_issues:
                lines.append("### Security Issues")
                lines.append("")
                for issue in security_issues[:5]:
                    lines.extend(self._format_security_issue(issue))
                    lines.append("")

        # Performance Issues
        if categorized.get("performance"):
            lines.append("## ⚡ Performance Issues")
            lines.append("")
            for issue in categorized["performance"][:10]:
                lines.extend(self._format_performance_issue(issue))
                lines.append("")

        # Architecture
        if categorized.get("architecture"):
            lines.append("## 🏗️ Architectural Insights")
            lines.append("")
            for issue in categorized["architecture"][:10]:
                lines.extend(self._format_architectural_issue(issue))
                lines.append("")

        # Patterns
        if categorized.get("patterns"):
            lines.append("## 🔍 Patterns Detected")
            lines.append("")
            for pattern in categorized["patterns"][:10]:
                lines.extend(self._format_pattern(pattern))
                lines.append("")

        # Recommendations
        lines.append("## 📋 Recommendations")
        lines.append("")
        lines.extend(self._generate_recommendations(categorized))
        lines.append("")

        # Next Steps
        lines.append("## 🚀 Next Steps")
        lines.append("")
        lines.extend(self._generate_next_steps(categorized))

        return "\n".join(lines)

    def _generate_executive_summary(
        self, categorized: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Generate executive summary."""
        lines = []

        # Count issues
        total_issues = sum(len(v) for v in categorized.values())
        critical_count = sum(
            1
            for cat in ["bugs", "security"]
            for item in categorized.get(cat, [])
            if item.get("severity", "").upper() in ["CRITICAL", "HIGH"]
        )

        lines.append(f"Found **{total_issues}** issues during deep analysis:")
        lines.append("")

        # Summary by category
        if categorized.get("bugs"):
            lines.append(f"- 🐛 **{len(categorized['bugs'])}** bugs")
        if categorized.get("security"):
            lines.append(f"- 🔒 **{len(categorized['security'])}** security issues")
        if categorized.get("performance"):
            lines.append(f"- ⚡ **{len(categorized['performance'])}** performance issues")
        if categorized.get("architecture"):
            lines.append(f"- 🏗️ **{len(categorized['architecture'])}** architectural concerns")
        if categorized.get("patterns"):
            lines.append(f"- 🔍 **{len(categorized['patterns'])}** patterns detected")

        lines.append("")

        if critical_count > 0:
            lines.append(f"**⚠️ {critical_count} critical issues require immediate attention**")

        return lines

    def _format_bug(self, bug: Dict[str, Any]) -> List[str]:
        """Format bug for markdown."""
        lines = []

        title = bug.get("issue", "Bug")
        file_path = bug.get("file", "Unknown")
        line = bug.get("line", "")
        severity = bug.get("severity", "MEDIUM")

        lines.append(f"#### {title}")
        lines.append(f"**File**: `{file_path}`" + (f":{line}" if line else ""))
        lines.append(f"**Severity**: {severity}")

        if "description" in bug:
            lines.append(f"**Description**: {bug['description']}")

        if "fix" in bug:
            lines.append(f"**Fix**: {bug['fix']}")

        return lines

    def _format_security_issue(self, issue: Dict[str, Any]) -> List[str]:
        """Format security issue."""
        lines = []

        title = issue.get("type", "Security Issue")
        severity = issue.get("severity", "MEDIUM")

        lines.append(f"#### {title}")
        lines.append(f"**Severity**: {severity}")

        if "description" in issue:
            lines.append(f"**Description**: {issue['description']}")

        if "mitigation" in issue:
            lines.append(f"**Mitigation**: {issue['mitigation']}")

        if "file" in issue:
            lines.append(f"**Location**: `{issue['file']}`")

        return lines

    def _format_performance_issue(self, issue: Dict[str, Any]) -> List[str]:
        """Format performance issue."""
        lines = []

        title = issue.get("issue", "Performance Issue")

        lines.append(f"#### {title}")

        if "file" in issue:
            lines.append(f"**File**: `{issue['file']}`")

        if "impact" in issue:
            lines.append(f"**Impact**: {issue['impact']}")

        if "suggestion" in issue:
            lines.append(f"**Suggestion**: {issue['suggestion']}")

        return lines

    def _format_architectural_issue(self, issue: Dict[str, Any]) -> List[str]:
        """Format architectural issue."""
        lines = []

        title = issue.get("type", "Architectural Issue")

        lines.append(f"#### {title}")

        if "description" in issue:
            lines.append(f"{issue['description']}")

        if "files" in issue and isinstance(issue["files"], list):
            lines.append(f"**Affected files**: {len(issue['files'])} files")

        if "recommendation" in issue:
            lines.append(f"**Recommendation**: {issue['recommendation']}")

        return lines

    def _format_pattern(self, pattern: Dict[str, Any]) -> List[str]:
        """Format detected pattern."""
        lines = []

        name = pattern.get("name", "Pattern")
        pattern_type = pattern.get("type", "pattern")

        lines.append(f"#### {name} ({pattern_type})")

        if "description" in pattern:
            lines.append(pattern["description"])

        if "occurrences" in pattern:
            lines.append(f"**Occurrences**: {pattern['occurrences']}")

        if "files" in pattern and isinstance(pattern["files"], list):
            files_str = ", ".join(f"`{f}`" for f in pattern["files"][:3])
            if len(pattern["files"]) > 3:
                files_str += f" and {len(pattern['files']) - 3} more"
            lines.append(f"**Files**: {files_str}")

        return lines

    def _generate_recommendations(self, categorized: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate prioritized recommendations."""
        recs = []

        # Critical bugs first
        critical_bugs = [
            b
            for b in categorized.get("bugs", [])
            if b.get("severity", "").upper() in ["CRITICAL", "HIGH"]
        ]
        if critical_bugs:
            recs.append(f"1. **Fix {len(critical_bugs)} critical bugs immediately**")
            for i, bug in enumerate(critical_bugs[:3], 1):
                file_ref = bug.get("file", "code")
                issue = bug.get("issue", "bug")
                recs.append(f"   - {file_ref}: {issue}")

        # Security
        if categorized.get("security"):
            count = len(categorized["security"])
            recs.append(f"2. **Address {count} security vulnerabilities**")

        # Performance
        perf_issues = categorized.get("performance", [])
        if perf_issues:
            high_impact = [p for p in perf_issues if "high" in str(p.get("impact", "")).lower()]
            if high_impact:
                recs.append(f"3. **Optimize {len(high_impact)} high-impact performance issues**")

        # Architecture
        if categorized.get("architecture"):
            recs.append("4. **Consider architectural refactoring for maintainability**")

        return recs

    def _generate_next_steps(self, categorized: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate actionable next steps."""
        steps = []

        if categorized.get("bugs") or categorized.get("security"):
            steps.append("1. Review and fix all critical issues")
            steps.append("2. Add tests to prevent regression")

        if categorized.get("performance"):
            steps.append("3. Profile and optimize identified bottlenecks")

        steps.append("4. Update documentation with findings")
        steps.append("5. Schedule follow-up analysis in 30 days")

        return steps

    def _generate_summary_stats(
        self, categorized: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, int]:
        """Generate summary statistics."""
        return {
            "total": sum(len(v) for v in categorized.values()),
            "bugs": len(categorized.get("bugs", [])),
            "security": len(categorized.get("security", [])),
            "performance": len(categorized.get("performance", [])),
            "architecture": len(categorized.get("architecture", [])),
            "patterns": len(categorized.get("patterns", [])),
        }

    async def _update_summary(self, categorized: Dict[str, List[Dict[str, Any]]]) -> None:
        """Update latest summary file."""
        summary_path = self.dream_path / "summaries"
        summary_path.mkdir(exist_ok=True)

        latest_file = summary_path / "latest.json"

        summary = {
            "last_analysis": utc_now_iso(),
            "stats": self._generate_summary_stats(categorized),
            "critical_count": sum(
                1
                for cat in ["bugs", "security"]
                for item in categorized.get(cat, [])
                if item.get("severity", "").upper() in ["CRITICAL", "HIGH"]
            ),
            "top_issues": self._get_top_issues(categorized, 5),
        }

        latest_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _get_top_issues(
        self, categorized: Dict[str, List[Dict[str, Any]]], limit: int
    ) -> List[Dict[str, str]]:
        """Get top issues across all categories."""
        all_issues = []

        # Collect all issues with normalized format
        for category, items in categorized.items():
            for item in items:
                issue = {
                    "category": category,
                    "title": (item.get("title") or item.get("issue") or item.get("type", "Issue")),
                    "severity": item.get("severity", "MEDIUM"),
                    "file": item.get("file", ""),
                }
                all_issues.append(issue)

        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        all_issues.sort(key=lambda x: severity_order.get(x["severity"], 4))

        return all_issues[:limit]

    async def _cleanup_old_insights(self) -> None:
        """Clean up old insight documents."""
        try:
            cutoff_date = utc_now() - timedelta(days=self.keep_days)

            # Clean filesystem
            for file_path in self.dream_path.glob("*.md"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    logger.info("Deleted old insight", filename=file_path.name)

            # Clean database
            await self.db.execute_async(
                """
                DELETE FROM dream_insights
                WHERE datetime(created_at) < datetime('now', '-' || ? || ' days')
                """,
                (self.keep_days,),
            )

        except Exception as e:
            logger.warning("Cleanup failed", error=str(e))
