"""
Endpoints for API indexing from Dashboard and Git Hooks.
NOT for direct user use.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
from fnmatch import fnmatch

# Core imports
from acolyte.core.logging import logger
from acolyte.core.id_generator import generate_id
from acolyte.core.secure_config import Settings
from acolyte.core.utils.file_types import FileTypeDetector
from acolyte.core.exceptions import (
    ValidationError,
    ConfigurationError,
    from_exception,
    internal_error,
)

# Services imports
from acolyte.services import IndexingService

# NOTE: The progress is notified automatically via EventBus
# IndexingService publishes ProgressEvent → WebSocket listens to it
# No manual notify_progress() is required

router = APIRouter()

# Configuration
config = Settings()
logger.info("Indexing API initializing...", module="index")


# ============================================================================
# MODELS FOR REQUEST/RESPONSE
# ============================================================================


class ProjectIndexRequest(BaseModel):
    """Request for initial indexing from dashboard."""

    patterns: List[str] = Field(
        default_factory=lambda: _get_default_patterns(),
        description="File patterns to index",
    )
    exclude_patterns: List[str] = Field(
        default=["**/node_modules/**", "**/__pycache__/**", "**/dist/**", "**/.git/**"],
        description="File patterns to exclude",
    )
    respect_gitignore: bool = Field(default=True, description="Respect .gitignore rules")
    respect_acolyteignore: bool = Field(default=True, description="Respect .acolyteignore rules")
    force_reindex: bool = Field(default=False, description="Force re-indexing of existing files")

    @field_validator("patterns")
    @classmethod
    def validate_patterns(cls, v):
        if not v:
            raise ValueError("At least one pattern required")
        if len(v) > 50:
            raise ValueError("Too many patterns (max 50)")
        return v

    @field_validator("exclude_patterns")
    @classmethod
    def validate_exclude_patterns(cls, v):
        if len(v) > 100:
            raise ValueError("Too many exclude patterns (max 100)")
        return v


class GitChangeFile(BaseModel):
    """Information about a modified file in Git."""

    path: str = Field(..., description="Relative path of the file")
    action: str = Field(..., description="Action: added, modified, deleted, renamed")
    old_path: Optional[str] = Field(None, description="Previous path (only for renamed)")
    diff: Optional[str] = Field(None, description="Diff of the file (optional)")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        valid_actions = {"added", "modified", "deleted", "renamed"}
        if v not in valid_actions:
            raise ValueError(f"Invalid action: {v}. Must be one of: {valid_actions}")
        return v

    @field_validator("path")
    @classmethod
    def validate_path(cls, v):
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")

        # Improved security validation with pathlib
        from pathlib import Path

        try:
            # Normalize and resolve the path (without base yet)
            path_str = v.strip()

            # Reject absolute paths or paths with dangerous characters
            if path_str.startswith(("/", "\\")) or ".." in path_str:
                logger.info(
                    "[UNTESTED PATH] GitChangeFile path validation: absolute or parent refs"
                )
                raise ValueError("Path cannot be absolute or contain parent directory references")

            # Reject paths with problematic Windows characters
            if any(char in path_str for char in [":", "*", "?", '"', "<", ">", "|"]) and not (
                len(path_str) > 1 and path_str[1] == ":"
            ):
                logger.info("[UNTESTED PATH] GitChangeFile path validation: invalid characters")
                raise ValueError("Path contains invalid characters")

            # Try to create a Path to validate format
            test_path = Path(path_str)

            # Reject if it has absolute components or parent
            if test_path.is_absolute() or any(part == ".." for part in test_path.parts):
                logger.info("[UNTESTED PATH] GitChangeFile path validation: absolute components")
                raise ValueError("Path must be relative and cannot navigate to parent directories")

            return path_str

        except (ValueError, OSError) as e:
            # Re-throw ValueError with clearer message
            if isinstance(e, ValueError):
                raise e
            logger.info("[UNTESTED PATH] GitChangeFile path validation: OSError")
            raise ValueError(f"Invalid path format: {str(e)}")
        except Exception:
            logger.info("[UNTESTED PATH] GitChangeFile path validation: general exception")
            raise ValueError("Invalid path format")


class GitChangesRequest(BaseModel):
    """Request from git hooks after commit."""

    trigger: str = Field(..., description="Trigger type: commit, pull, checkout, fetch")
    files: List[GitChangeFile] = Field(..., description="List of modified files")

    # Metadata of the commit (optional)
    commit_hash: Optional[str] = Field(None, description="Hash of the commit")
    branch: Optional[str] = Field(None, description="Current branch")
    author: Optional[str] = Field(None, description="Author of the commit")
    message: Optional[str] = Field(None, description="Message of the commit")
    timestamp: Optional[int] = Field(None, description="Timestamp of the commit")

    # Metadata specific to the trigger
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("trigger")
    @classmethod
    def validate_trigger(cls, v):
        valid_triggers = {"commit", "pull", "checkout", "fetch"}
        if v not in valid_triggers:
            raise ValueError(f"Invalid trigger: {v}. Must be one of: {valid_triggers}")
        return v

    @field_validator("files")
    @classmethod
    def validate_files(cls, v):
        if not v:
            logger.info("[UNTESTED PATH] GitChangesRequest.validate_files: empty list")
            raise ValueError("At least one file change required")
        if len(v) > 1000:
            logger.info("[UNTESTED PATH] GitChangesRequest.validate_files: too many files")
            raise ValueError("Too many file changes (max 1000)")
        return v


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("/project")
async def index_project(
    request: ProjectIndexRequest, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Initial indexing of the entire project.

    Called by:
    - Dashboard web during initial setup
    - CLI command `acolyte index` in emergencies

    IMPORTANT: It can take several minutes to index large projects.
    Use WebSocket /api/ws/progress/{task_id} to see progress.
    """
    start_time = time.time()
    task_id = f"idx_{int(time.time())}_{generate_id()[:8]}"

    logger.info(
        "Project index request",
        patterns_count=len(request.patterns),
        force_reindex=request.force_reindex,
        task_id=task_id,
    )

    try:
        # IndexingService handles concurrency internally with _indexing_lock
        # If there is indexing in progress, index_files() will automatically throw an exception

        # Get the root of the project safely
        try:
            project_root = Path(config.get("project.path", ".")).resolve()
            if not project_root.exists():
                raise ConfigurationError(
                    message=f"Project root does not exist: {project_root}",
                    context={"configured_path": config.get("project.path")},
                )
        except Exception as e:
            raise ConfigurationError(
                message="Invalid project root configuration", context={"error": str(e)}
            )

        # Collect all files in the project in a single pass
        all_files = [str(f) for f in project_root.rglob("*") if f.is_file()]

        # Unify patterns in glob format
        patterns = request.patterns

        # Filter files that match at least one pattern
        files_to_index = [
            f
            for f in all_files
            if any(
                fnmatch(f, pattern.replace("*.", "*.")) or fnmatch(f, pattern)
                for pattern in patterns
            )
        ]

        # Remove duplicates (just in case)
        files_to_index = list(set(files_to_index))

        # Estimate files to index
        try:
            estimated_files = len(files_to_index)
        except Exception as e:
            logger.warning("Failed to estimate files", error=str(e))
            estimated_files = 0

        # Calculate estimated time (approx 0.1s per file)
        estimated_seconds = max(estimated_files * 0.1, 5)  # Minimum 5 seconds

        # Start asynchronous indexing
        background_tasks.add_task(
            _run_project_indexing,
            task_id=task_id,
            project_root=project_root,
            request=request,
            estimated_files=estimated_files,
        )

        processing_time = int((time.time() - start_time) * 1000)

        logger.info(
            "Project index started",
            task_id=task_id,
            estimated_files=estimated_files,
            processing_time_ms=processing_time,
        )

        return {
            "task_id": task_id,
            "status": "started",
            "estimated_files": estimated_files,
            "estimated_seconds": int(estimated_seconds),
            "websocket_url": f"/api/ws/progress/{task_id}",
            "project_root": str(project_root),
            "patterns": request.patterns,
            "message": "Project indexing started. Connect to WebSocket for real-time progress.",
        }

    except (ValidationError, ConfigurationError) as e:
        logger.warning(
            "Project index validation failed", validation_message=e.message, task_id=task_id
        )
        logger.info("[UNTESTED PATH] index_project validation/config error")
        raise HTTPException(status_code=400, detail=from_exception(e).model_dump())

    except Exception as e:
        logger.error("Project index failed", error=str(e), task_id=task_id, exc_info=True)
        error_response = internal_error(
            message="Failed to start project indexing",
            error_id=task_id,
            context={"error_type": type(e).__name__},
        )
        raise HTTPException(status_code=500, detail=error_response.model_dump())


@router.post("/git-changes")
async def index_git_changes(
    request: GitChangesRequest, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Index changes after a Git commit.

    Called automatically by git hooks (post-commit, post-merge, etc.).
    Processes only modified files, not the entire project.

    IMPORTANT: This endpoint is fast (only processes diffs).
    """
    start_time = time.time()
    request_id = generate_id()[:8]

    logger.info(
        "Git changes request",
        trigger=request.trigger,
        files_count=len(request.files),
        request_id=request_id,
    )

    try:
        # Get the root of the project
        project_root = Path(config.get("project.path", ".")).resolve()

        processed_files = []
        skipped_files = []
        error_files = []

        # Process each file
        for file_change in request.files:
            try:
                result = await _process_file_change(
                    project_root=project_root,
                    file_change=file_change,
                    trigger=request.trigger,
                    commit_metadata={
                        "hash": request.commit_hash,
                        "author": request.author,
                        "message": request.message,
                        "timestamp": request.timestamp,
                        "branch": request.branch,
                    },
                )

                if result["status"] == "processed":
                    processed_files.append(result)
                elif result["status"] == "skipped":
                    skipped_files.append(result)

            except Exception as e:
                logger.error("Error processing file", path=file_change.path, error=str(e))
                error_files.append(
                    {
                        "file": file_change.path,
                        "action": file_change.action,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )

        # Apply cache invalidation if necessary
        if request.trigger in ["pull", "checkout"] and processed_files:
            try:
                # The cache invalidation is handled automatically by the EventBus
                # The services subscribe to CacheInvalidateEvent as needed
                logger.info("Cache invalidation triggered", files_count=len(processed_files))
            except Exception as e:
                logger.warning("Cache invalidation failed", error=str(e))

        processing_time = int((time.time() - start_time) * 1000)

        # Determine the general state
        total_files = len(request.files)
        success_rate = len(processed_files) / total_files if total_files > 0 else 0

        status = "success"
        if error_files:
            status = "partial_success" if processed_files else "failed"

        result = {
            "status": status,
            "trigger": request.trigger,
            "processing_time_ms": processing_time,
            "summary": {
                "total_files": total_files,
                "processed": len(processed_files),
                "skipped": len(skipped_files),
                "errors": len(error_files),
                "success_rate": round(success_rate, 2),
            },
            "details": {
                "processed_files": processed_files[:20],  # First 20
                "skipped_files": skipped_files[:10],  # First 10
                "error_files": error_files[:10],  # First 10
            },
        }

        # Add commit metadata if available
        if request.commit_hash:
            result["commit"] = {
                "hash": request.commit_hash[:8],
                "branch": request.branch,
                "author": request.author,
                "message": request.message[:100] if request.message else None,
            }

        logger.info(
            "Git changes processed",
            status=status,
            processed_count=len(processed_files),
            total_files=total_files,
            processing_time_ms=processing_time,
            request_id=request_id,
        )

        return result

    except Exception as e:
        logger.error("Git changes failed", error=str(e), request_id=request_id, exc_info=True)
        error_response = internal_error(
            message="Failed to process git changes",
            error_id=request_id,
            context={
                "error_type": type(e).__name__,
                "trigger": request.trigger,
                "files_count": len(request.files),
            },
        )
        raise HTTPException(status_code=500, detail=error_response.model_dump())


# NOTE: Endpoint /cache removed - over-engineering unnecessary
# Orphaned embeddings are a theoretical problem that doesn't happen in practice


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _get_default_patterns() -> List[str]:
    """Get default file patterns from FileTypeDetector."""
    # Get all supported extensions
    extensions = FileTypeDetector.get_all_supported_extensions()
    # Convert to glob patterns (add * prefix)
    patterns = [f"*{ext}" for ext in sorted(extensions)]
    # Return most common ones first
    priority_patterns = ["*.py", "*.js", "*.ts", "*.tsx", "*.md", "*.yml", "*.yaml"]
    other_patterns = [p for p in patterns if p not in priority_patterns]
    return priority_patterns + other_patterns[:10]  # Limit to avoid too many patterns


async def _estimate_files_to_index(
    root: Path,
    patterns: List[str],
    exclude_patterns: List[str],
    respect_gitignore: bool,
    respect_acolyteignore: bool,
) -> int:
    """
    Estimate how many files will be indexed using IndexingService.
    """
    try:
        indexing_service = IndexingService()
        return await indexing_service.estimate_files(
            root=root,
            patterns=patterns,
            exclude_patterns=exclude_patterns,
            respect_gitignore=respect_gitignore,
            respect_acolyteignore=respect_acolyteignore,
        )

    except Exception as e:
        logger.warning("File estimation failed", error=str(e))
        logger.info("[UNTESTED PATH] _estimate_files_to_index failed")
        return 100  # Conservative estimation


async def _run_project_indexing(
    task_id: str, project_root: Path, request: ProjectIndexRequest, estimated_files: int
) -> None:
    """
    Executes the project indexing in background.

    PROGRESS FLOW:
    1. IndexingService calls _notify_progress() internally
    2. _notify_progress() publishes ProgressEvent to the EventBus
    3. WebSocket handler listens to events where task_id appears in the message
    4. WebSocket sends updates to the client automatically

    No manual notification is required - the system is reactive via EventBus.
    """
    try:
        logger.info("Starting project indexing", task_id=task_id)

        # Collect files in the project using patterns
        files_to_index = []

        for pattern in request.patterns:
            if pattern.startswith("*."):
                ext = pattern[2:]
                matches = list(project_root.rglob(f"*.{ext}"))
                files_to_index.extend([str(f) for f in matches if f.is_file()])
            else:
                matches = list(project_root.rglob(pattern))
                files_to_index.extend([str(f) for f in matches if f.is_file()])

        # Remove duplicates
        files_to_index = list(set(files_to_index))

        # Use real IndexingService to index
        indexing_service = IndexingService()

        # The progress is notified automatically when IndexingService processes files
        # The WebSocket will detect events with "Task: {task_id}" in the message

        # Index using the real service
        # IndexingService will include "Task: {task_id}" in the progress messages
        # so the WebSocket can filter events for this specific task
        await indexing_service.index_files(
            files=files_to_index,
            trigger="manual",
            task_id=task_id,  # Now pass the task_id for precise filtering
        )

        logger.info("Project indexing completed", task_id=task_id)

    except Exception as e:
        logger.error("Project indexing failed", task_id=task_id, error=str(e), exc_info=True)
        logger.info("[UNTESTED PATH] _run_project_indexing failed")
        # The error will naturally propagate to the client when no more events are received
        # or we could publish an ErrorEvent to the EventBus (TODO)


async def _process_file_change(
    project_root: Path, file_change: GitChangeFile, trigger: str, commit_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Processes an individual file change.
    """
    try:
        # Validate path safely with pathlib
        try:
            # Use resolve() with strict=False to handle files that don't exist yet
            file_path = (project_root / file_change.path).resolve(strict=False)

            # Verify that the resolved path is inside the project
            try:
                file_path.relative_to(project_root)
            except ValueError:
                # The path is outside the project
                return {
                    "file": file_change.path,
                    "action": file_change.action,
                    "status": "skipped",
                    "reason": "outside_project",
                }

            # Verify against malicious symlinks
            if file_path.exists() and file_path.is_symlink():
                # Resolve the symlink and verify that it points inside the project
                real_path = file_path.resolve(strict=True)
                try:
                    real_path.relative_to(project_root)
                except ValueError:
                    logger.info("[UNTESTED PATH] _process_file_change: symlink outside project")
                    return {
                        "file": file_change.path,
                        "action": file_change.action,
                        "status": "skipped",
                        "reason": "symlink_outside_project",
                    }

        except (ValueError, OSError) as e:
            logger.warning("Invalid path", path=file_change.path, error=str(e))
            return {
                "file": file_change.path,
                "action": file_change.action,
                "status": "skipped",
                "reason": "invalid_path",
                "error": str(e),
            }

        # Verify if the file should be indexed (using IndexingService logic)
        indexing_service = IndexingService()
        if not indexing_service.is_supported_file(file_path):
            return {
                "file": file_change.path,
                "action": file_change.action,
                "status": "skipped",
                "reason": "unsupported_file_type",
            }

        # Process according to the action
        if file_change.action == "deleted":
            # Remove from the index using IndexingService
            success = await indexing_service.remove_file(str(file_path))
            return {
                "file": file_change.path,
                "action": "removed",
                "status": "processed" if success else "error",
                "success": success,
            }

        elif file_change.action in ["added", "modified"]:
            # Re-index the file using IndexingService
            await indexing_service.index_files(
                files=[str(file_path)],
                trigger=trigger,
                task_id=None,  # Git hooks don't have a specific task_id
            )

            return {
                "file": file_change.path,
                "action": "indexed",
                "status": "processed",
                "chunks_created": 0,
                "embeddings_created": 0,
            }

        elif file_change.action == "renamed":
            # Update references in the index
            if file_change.old_path:
                success = await indexing_service.rename_file(
                    old_path=file_change.old_path, new_path=str(file_path)
                )
                return {
                    "file": file_change.path,
                    "action": "renamed",
                    "status": "processed" if success else "error",
                    "old_path": file_change.old_path,
                    "success": success,
                }
            else:
                return {
                    "file": file_change.path,
                    "action": "renamed",
                    "status": "error",
                    "error": "old_path required for rename operation",
                }

        return {
            "file": file_change.path,
            "action": file_change.action,
            "status": "skipped",
            "reason": "unknown_action",
        }

    except Exception as e:
        logger.error("Failed to process file change", path=file_change.path, error=str(e))
        return {
            "file": file_change.path,
            "action": file_change.action,
            "status": "error",
            "error": str(e),
        }
