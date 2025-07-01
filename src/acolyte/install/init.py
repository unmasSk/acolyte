"""
ACOLYTE Project Initialization Module

Provides classes and functions for initializing ACOLYTE in a project.
No subprocess calls - everything is Python functions!
"""

import json
import os
import shutil
import stat
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from acolyte.core.logging import logger
from acolyte.core.utils.datetime_utils import utc_now_iso
from acolyte._version import __version__
from acolyte.install.common import (
    Colors,
    print_error,
    print_header,
    print_info,
    print_step,
    print_success,
    print_warning,
    show_spinner,
)
from acolyte.install.resources_manager import copy_resource_to_path


class DependencyChecker:
    """Check and install missing dependencies"""

    REQUIRED_TOOLS: Dict[str, Dict[str, Any]] = {
        "git": {
            "command": ["git", "--version"],
            "min_version": "2.0.0",
            "install": {
                "windows": "winget install --id Git.Git -e --source winget",
                "linux": "sudo apt-get install git || sudo yum install git",
                "darwin": "brew install git",
            },
            "install_hint": "Install from: https://git-scm.com",
        },
        "docker": {
            "command": ["docker", "--version"],
            "min_version": "20.0.0",
            "install": {
                "windows": "winget install Docker.DockerDesktop",
                "linux": "curl -fsSL https://get.docker.com | sh",
                "darwin": "brew install --cask docker",
            },
            "install_hint": "Install from: https://docs.docker.com/get-docker/",
        },
        "python": {
            "command": (["python3", "--version"] if os.name != "nt" else ["python", "--version"]),
            "min_version": "3.11.0",
            "install": {
                "windows": "winget install Python.Python.3.11",
                "linux": "sudo apt-get install python3.11 || sudo yum install python3.11",
                "darwin": "brew install python@3.11",
            },
            "install_hint": "Python 3.11+ required",
        },
    }

    @classmethod
    def check_tool(cls, tool_name: str, tool_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if a tool is installed and meets version requirements"""
        try:
            command = tool_info["command"].copy()
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                return False, "Not installed"

            # Extract version if needed
            output = result.stdout.strip()
            if tool_info.get("min_version"):
                import re
                from packaging.version import Version

                version_match = re.search(r"(\d+\.\d+\.\d+)", output)
                if version_match:
                    version_str = version_match.group(1)
                    try:
                        version = Version(version_str)
                        min_version = Version(tool_info["min_version"])
                        if version < min_version:
                            return False, f"Version {version_str} < {tool_info['min_version']}"
                        return True, f"Version {version_str}"
                    except Exception as e:
                        logger.warning(f"Failed to parse version '{version_str}': {e}")
                        return False, f"Invalid version format: {version_str}"

            return True, "Installed"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "Not found"
        except Exception as e:
            logger.error("Error checking tool", tool_name=tool_name, error=str(e))
            return False, f"Error: {str(e)}"

    @classmethod
    def check_all(cls) -> Tuple[bool, Dict[str, Tuple[bool, str]]]:
        """Check all required tools"""
        results = {}
        all_ok = True

        print_info("Checking system requirements...")

        for tool_name, tool_info in cls.REQUIRED_TOOLS.items():
            show_spinner(f"Checking {tool_name}...", 0.5)
            success, message = cls.check_tool(tool_name, tool_info)

            if success:
                print_success(f"{tool_name}: {message}")
            else:
                print_error(f"{tool_name}: {message}")
                print_info(f"  {tool_info['install_hint']}")
                all_ok = False

            results[tool_name] = (success, message)

        return all_ok, results


class ProjectValidator:
    """Validate project directory and structure"""

    VALID_PROJECT_MARKERS = [
        ".git",
        "package.json",
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "Cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "composer.json",
        "Gemfile",
        "CMakeLists.txt",
        "Makefile",
        ".gitignore",
        "README.md",
        "readme.md",
        "README.rst",
        "index.html",
        "main.py",
        "app.py",
        "src/",
        "lib/",
    ]

    @classmethod
    def is_valid_project_directory(cls, path: Path) -> Tuple[bool, str]:
        """Check if directory is a valid project"""
        if not path.exists():
            return False, "Directory does not exist"

        if not path.is_dir():
            return False, "Not a directory"

        # Check for any project markers
        found_markers = []
        for marker in cls.VALID_PROJECT_MARKERS:
            marker_path = path / marker
            if marker_path.exists():
                found_markers.append(marker)

        if not found_markers:
            return False, "No project files found (no package.json, pyproject.toml, .git, etc.)"

        # Detect project type
        project_type = cls._detect_project_type(path, found_markers)
        return True, project_type

    @classmethod
    def _detect_project_type(cls, path: Path, markers: List[str]) -> str:
        """Detect the type of project"""
        if "package.json" in markers:
            try:
                with open(path / "package.json") as f:
                    pkg = json.load(f)
                    deps = pkg.get("dependencies", {})

                    if any(k in deps for k in ["react", "vue", "angular", "@angular/core"]):
                        return "Frontend JavaScript/TypeScript project"
                    elif any(k in deps for k in ["express", "fastify", "koa", "nestjs"]):
                        return "Backend Node.js project"
                    else:
                        return "Node.js project"
            except Exception:
                return "Node.js project"

        elif "pyproject.toml" in markers or "setup.py" in markers or "requirements.txt" in markers:
            return "Python project"
        elif "Cargo.toml" in markers:
            return "Rust project"
        elif "go.mod" in markers:
            return "Go project"
        elif "pom.xml" in markers or "build.gradle" in markers:
            return "Java project"
        elif ".git" in markers:
            return "Git repository"
        else:
            return "Generic project"

    @classmethod
    def ensure_git_initialized(cls, path: Path) -> bool:
        """Ensure Git is initialized in the project"""
        try:
            git_dir = path / ".git"
            if not git_dir.exists():
                result = subprocess.run(["git", "init"], cwd=path, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error("Failed to initialize git repo", error=result.stderr)
                    return False
            return True
        except Exception as e:
            logger.error("Error initializing git repo", error=str(e))
            return False


class GitHooksManager:
    """Manages Git hooks installation"""

    HOOK_NAMES = ["post-commit", "post-merge", "post-checkout", "post-fetch"]

    @classmethod
    def install_hooks(cls, project_path: Path) -> bool:
        """Install Git hooks for automatic indexing"""
        try:
            git_dir = project_path / ".git"
            if not git_dir.exists():
                logger.error("No .git directory found")
                return False

            hooks_dir = git_dir / "hooks"
            hooks_dir.mkdir(exist_ok=True)

            installed = []
            for hook_name in cls.HOOK_NAMES:
                if cls._install_single_hook(hooks_dir, hook_name):
                    installed.append(hook_name)

            if not installed:
                logger.error("No Git hooks were installed")
                return False

            print_info(f"Installed {len(installed)} Git hooks: {', '.join(installed)}")
            print_info("Git hooks will notify ACOLYTE of code changes automatically")
            return True

        except Exception as e:
            logger.error("Error installing Git hooks", error=str(e))
            return False

    @classmethod
    def _install_single_hook(cls, hooks_dir: Path, hook_name: str) -> bool:
        """Install a single Git hook"""
        try:
            dest = hooks_dir / hook_name

            # Backup existing hook if present
            if dest.exists():
                backup = dest.with_suffix(".backup")
                shutil.copy2(dest, backup)
                logger.info(f"Backed up existing {hook_name} hook")

            # Copy hook from resources
            if not copy_resource_to_path(f"hooks/{hook_name}", dest):
                logger.warning(f"Hook {hook_name} not found in resources")
                return False

            # Make executable
            st = os.stat(dest)
            os.chmod(dest, st.st_mode | stat.S_IEXEC)

            logger.info(f"Installed {hook_name} hook")
            return True

        except Exception as e:
            logger.error(f"Error installing hook {hook_name}", error=str(e))
            return False


class ProjectInitializer:
    """Main initialization class for ACOLYTE projects"""

    def __init__(self, project_path: Path, global_dir: Optional[Path] = None):
        self.project_path = project_path.resolve()
        self.global_dir = global_dir or Path.home() / ".acolyte"
        self.project_id = self._generate_project_id()

    def _generate_project_id(self) -> str:
        """Generate unique project ID from path and git remote"""
        git_remote = ""
        git_dir = self.project_path / ".git"
        if git_dir.exists():
            try:
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    git_remote = result.stdout.strip()
            except Exception:
                pass

        # Generate hash from absolute path + git remote
        abs_path = str(self.project_path)
        unique_string = f"{git_remote}:{abs_path}"
        project_id = hashlib.sha256(unique_string.encode()).hexdigest()[:12]
        return project_id

    def check_already_initialized(self) -> Tuple[bool, Optional[str]]:
        """Check if project is already initialized"""
        project_file = self.project_path / ".acolyte.project"

        if project_file.exists():
            try:
                with open(project_file) as f:
                    data = json.load(f)
                    return True, data.get("project_id")
            except Exception:
                return False, None

        return False, None

    def create_project_structure(
        self, project_name: Optional[str] = None, user_name: Optional[str] = None
    ) -> bool:
        """Create basic project structure"""
        try:
            # Create global directory structure
            project_global_dir = self.global_dir / "projects" / self.project_id
            project_global_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            (project_global_dir / "data").mkdir(exist_ok=True)
            (project_global_dir / "data" / "logs").mkdir(exist_ok=True)
            (project_global_dir / "infra").mkdir(exist_ok=True)

            # Create .acolyte.project in user's project
            project_link = self.project_path / ".acolyte.project"
            project_data = {
                "project_id": self.project_id,
                "name": project_name or self.project_path.name,
                "user": user_name or "developer",
                "initialized": utc_now_iso(),
                "acolyte_version": __version__,
                "project_path": str(self.project_path),
            }

            with open(project_link, "w", encoding="utf-8") as f:
                json.dump(project_data, f, indent=2)

            print_success(f"Project link file created at: {project_link}")
            logger.info("Project link file created", path=str(project_link))

            # Create initial log entry
            log_entry = {
                "event": "project_initialized",
                "timestamp": utc_now_iso(),
                "project_path": str(self.project_path),
                "project_id": self.project_id,
            }
            log_path = self.project_path / ".acolyte.init.log"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([log_entry], f, indent=2)

            print_success(f"Initial log file created at: {log_path}")
            logger.info("Initial log file created", path=str(log_path))

            return True

        except PermissionError:
            print_error("Permission denied while creating project files")
            logger.error("Permission error creating project structure")
            return False
        except Exception as e:
            print_error("Unexpected error creating project structure")
            logger.error("Error creating project structure", error=str(e))
            return False

    def run(
        self,
        project_name: Optional[str] = None,
        user_name: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """
        Main initialization entry point

        Args:
            project_name: Optional project name
            user_name: Optional user name
            force: Force re-initialization

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if already initialized
            already_init, existing_id = self.check_already_initialized()
            if already_init and not force:
                print_warning(f"Project already initialized with ID: {existing_id}")
                print_info("Use --force to re-initialize")
                return False

            # Step 1: Validate project directory
            print_step(1, 6, "Validating Project Directory")
            is_valid, project_type = ProjectValidator.is_valid_project_directory(self.project_path)

            if not is_valid:
                print_error(f"Invalid project directory: {project_type}")
                print_warning("This does not look like a valid project directory")
                return False

            print_success(f"Valid project detected: {project_type}")

            # Step 2: Check dependencies
            print_step(2, 6, "Checking System Dependencies")
            deps_ok, deps_results = DependencyChecker.check_all()

            if not deps_ok:
                print_error("Some required dependencies are missing")
                return False

            print_success("All dependencies satisfied")

            # Step 3: Ensure Git repository
            print_step(3, 6, "Checking Git Repository")
            if not ProjectValidator.ensure_git_initialized(self.project_path):
                print_error("Could not initialize git repository")
                return False

            print_success("Git repository detected")

            # Step 4: Create project structure
            print_step(4, 6, "Creating Project Structure")
            show_spinner("Setting up ACOLYTE structure", 1.5)

            if not self.create_project_structure(project_name, user_name):
                print_error("Could not create project structure")
                return False

            print_success("Project structure created")

            # Step 5: Install Git hooks
            print_step(5, 6, "Installing Git Hooks")
            if not GitHooksManager.install_hooks(self.project_path):
                print_error("Could not install Git hooks")
                return False

            print_success("Git hooks installed")

            # Step 6: Show completion
            print_step(6, 6, "Initialization Complete")
            self._show_next_steps()

            logger.info("ACOLYTE initialized successfully", project_path=str(self.project_path))
            return True

        except KeyboardInterrupt:
            print_warning("\nInitialization cancelled by user.")
            return False
        except Exception as e:
            print_error(f"Initialization failed: {e}")
            logger.error("Fatal error during initialization", error=str(e), include_trace=True)
            return False

    def _show_next_steps(self):
        """Show what to do next"""
        print_header("✅ Initialization Complete!")

        print(f"\n{Colors.BOLD}Project initialized with ID:{Colors.ENDC} {self.project_id}")
        print(
            f"{Colors.BOLD}Configuration will be stored in:{Colors.ENDC} {self.global_dir / 'projects' / self.project_id}"
        )

        print(f"\n{Colors.CYAN}{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print(f"\n1. {Colors.GREEN}Configure and install ACOLYTE:{Colors.ENDC}")
        print(f"   {Colors.CYAN}acolyte install{Colors.ENDC}")
        print(f"\n2. {Colors.GREEN}Start ACOLYTE services:{Colors.ENDC}")
        print(f"   {Colors.CYAN}acolyte start{Colors.ENDC}")
        print(f"\n3. {Colors.GREEN}Check status:{Colors.ENDC}")
        print(f"   {Colors.CYAN}acolyte status{Colors.ENDC}")

        print(f"\n{Colors.YELLOW}💡 Tips:{Colors.ENDC}")
        if os.name == "nt":
            print("• Make sure Docker Desktop is running")
            print("• Run PowerShell as Administrator if you encounter permission issues")
        else:
            print("• Make sure Docker daemon is running")
            print("• You may need 'sudo' for Docker commands")

        print(f"\n{Colors.CYAN}For help:{Colors.ENDC} acolyte --help")
