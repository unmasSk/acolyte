#!/usr/bin/env python3
"""
🤖 ACOLYTE INIT - Quick Project Initialization
Like 'git init' - creates basic structure without interaction
"""

import json
import sys
import subprocess
import hashlib
import os
import stat
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Use centralized logging and datetime utilities
from acolyte.core.logging import logger
from acolyte.core.utils.datetime_utils import utc_now_iso

# Add common modules to path
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    ACOLYTE_LOGO,
    Colors,
    animate_text,
    print_error,
    print_header,
    print_info,
    print_step,
    print_success,
    print_warning,
    show_spinner,
)


class DependencyChecker:
    """Check and install missing dependencies"""

    # Type for tool info structure
    ToolInfo = Dict[str, Any]

    REQUIRED_TOOLS: Dict[str, ToolInfo] = {
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
            "command": (
                ["python3", "--version"] if sys.platform != "win32" else ["python", "--version"]
            ),
            "min_version": "3.11.0",
            "install": {
                "windows": "winget install Python.Python.3.11",
                "linux": "sudo apt-get install python3.11 || sudo yum install python3.11",
                "darwin": "brew install python@3.11",
            },
            "install_hint": "Python 3.11+ required",
        },
        "poetry": {
            "command": ["poetry", "--version"],
            "min_version": "1.0.0",
            "install": {
                "windows": "curl -sSL https://install.python-poetry.org | python -",
                "linux": "curl -sSL https://install.python-poetry.org | python3 -",
                "darwin": "brew install poetry",
            },
            "install_hint": "Install from: https://python-poetry.org/docs/",
            "optional": False,
        },
    }

    @classmethod
    def check_tool(cls, tool_name: str, tool_info: ToolInfo) -> Tuple[bool, str]:
        """Check if a tool is installed and meets version requirements"""
        try:
            command = tool_info["command"].copy()
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                return False, "Not installed"

            # Extract version if needed
            output = result.stdout.strip()
            if tool_info["min_version"]:
                import re

                version_match = re.search(r"(\d+\.\d+\.\d+)", output)
                if version_match:
                    version = version_match.group(1)
                    if version < tool_info["min_version"]:
                        return False, f"Version {version} < {tool_info['min_version']}"
                    return True, f"Version {version}"

            return True, "Installed"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "Not found"
        except Exception as e:
            logger.error("Error checking tool", tool_name=tool_name, error=str(e))
            return False, f"Error: {str(e)}"

    @classmethod
    def auto_install_tool(cls, tool_name: str, tool_info: ToolInfo) -> bool:
        """Attempt to automatically install a missing tool"""
        if tool_info.get("optional", False):
            return False  # Don't auto-install optional tools

        platform = sys.platform
        if platform.startswith("linux"):
            platform = "linux"
        elif platform == "darwin":
            platform = "darwin"
        elif platform == "win32":
            platform = "windows"
        else:
            return False

        install_cmd = tool_info["install"].get(platform)
        if not install_cmd:
            return False

        print_info(f"Attempting to install {tool_name}...")
        show_spinner(f"Installing {tool_name}", 2.0)

        try:
            # For Windows, handle special cases
            if platform == "windows" and "winget" in install_cmd:
                # Check if winget is available
                winget_check = subprocess.run(["winget", "--version"], capture_output=True)
                if winget_check.returncode != 0:
                    print_warning("winget not available, please install manually")
                    return False

            # Run installation command
            if platform == "windows":
                result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
            else:
                result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                print_success(f"{tool_name} installed successfully!")
                return True
            else:
                print_error(f"Failed to install {tool_name}: {result.stderr}")
                return False

        except Exception as e:
            print_error(f"Error installing {tool_name}: {e}")
            logger.error("Error installing tool", tool_name=tool_name, error=str(e))
            return False

    @classmethod
    def check_and_install_all(cls) -> Tuple[bool, Dict[str, Tuple[bool, str]]]:
        """Check all required tools and attempt to install missing ones"""
        results = {}
        all_ok = True

        print_info("Checking system requirements...")

        for tool_name, tool_info in cls.REQUIRED_TOOLS.items():
            show_spinner(f"Checking {tool_name}...", 0.5)
            success, message = cls.check_tool(tool_name, tool_info)

            if success:
                print_success(f"{tool_name}: {message}")
            else:
                if tool_info.get("optional", False):
                    print_warning(f"{tool_name}: {message} (optional)")
                else:
                    print_error(f"{tool_name}: {message}")

                    # Attempt auto-install for required tools
                    if cls.auto_install_tool(tool_name, tool_info):
                        # Re-check after installation
                        success, message = cls.check_tool(tool_name, tool_info)
                        if success:
                            print_success(f"{tool_name}: Now installed!")
                        else:
                            print_info(f"  {tool_info['install_hint']}")
                            print_warning(
                                f"Could not install {tool_name} automatically. Please install it manually and re-run 'acolyte init' in your project."
                            )
                            all_ok = False
                    else:
                        print_info(f"  {tool_info['install_hint']}")
                        print_warning(
                            f"Could not install {tool_name} automatically. Please install it manually and re-run 'acolyte init' in your project."
                        )
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
            print_error(
                "Directory does not exist. Please create your project directory first and re-run 'acolyte init'."
            )
            return False, "Directory does not exist"

        if not path.is_dir():
            print_error(
                "The specified path is not a directory. Please provide a valid project directory and re-run 'acolyte init'."
            )
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
            # Check if it's frontend or backend Node.js
            try:
                import json

                with open(path / "package.json") as f:
                    pkg = json.load(f)
                    deps = pkg.get("dependencies", {})

                    if any(k in deps for k in ["react", "vue", "angular", "@angular/core"]):
                        return "Frontend JavaScript/TypeScript project"
                    elif any(k in deps for k in ["express", "fastify", "koa", "nestjs"]):
                        return "Backend Node.js project"
                    else:
                        return "Node.js project"
            except json.JSONDecodeError as e:
                logger.error(
                    "Failed to parse package.json", path=str(path / "package.json"), error=str(e)
                )
                return "Node.js project"
            except IOError as e:
                logger.error(
                    "Failed to read package.json", path=str(path / "package.json"), error=str(e)
                )
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
        try:
            git_dir = path / ".git"
            if not git_dir.exists():
                result = subprocess.run(["git", "init"], cwd=path, capture_output=True, text=True)
                if result.returncode != 0:
                    print_error(
                        "Failed to initialize git repository. Please ensure you have git installed and write permissions, then re-run 'acolyte init'."
                    )
                    logger.error("Failed to initialize git repo", error=result.stderr)
                    return False
            return True
        except Exception as e:
            print_error(
                "Error initializing git repository. Please check your permissions and git installation, then re-run 'acolyte init'."
            )
            logger.error("Error initializing git repo", error=str(e))
            return False


class AcolyteInit:
    """Main initialization class - non-interactive"""

    def __init__(self, project_path: Path):
        self.project_path = project_path.resolve()
        self.global_dir = Path.home() / ".acolyte"
        self.project_id = self._generate_project_id()

    def _generate_project_id(self) -> str:
        """Generate unique project ID from path and git remote"""
        # Try to get git remote
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
            except subprocess.SubprocessError as e:
                logger.error("Failed to get git remote", error=str(e))
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
            except json.JSONDecodeError as e:
                logger.error(
                    "Failed to parse .acolyte.project", path=str(project_file), error=str(e)
                )
                return False, None
            except IOError as e:
                logger.error(
                    "Failed to read .acolyte.project", path=str(project_file), error=str(e)
                )
                return False, None

        return False, None

    def create_project_structure(self) -> bool:
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
                "initialized": utc_now_iso(),
                "acolyte_version": "1.0.0",
                "project_path": str(self.project_path),
            }

            with open(project_link, "w") as f:
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
            with open(log_path, "w") as f:
                json.dump([log_entry], f, indent=2)
            print_success(f"Initial log file created at: {log_path}")
            logger.info("Initial log file created", path=str(log_path))

            return True

        except PermissionError as e:
            print_error(
                "Permission denied while creating project files. Please ensure you have write permissions in this directory and re-run 'acolyte init'."
            )
            logger.error("Permission error creating project structure", error=str(e))
            return False
        except Exception as e:
            print_error(
                "Unexpected error creating project structure. Please check the logs and try again. If the problem persists, contact support."
            )
            logger.error("Error creating project structure", error=str(e))
            return False

    def install_git_hooks(self) -> bool:
        """Install Git hooks for automatic indexing."""
        try:
            # Verify we're in a git repo (should already be validated)
            git_dir = self.project_path / ".git"
            if not git_dir.exists():
                logger.error("No .git directory found")
                return False

            hooks_dir = git_dir / "hooks"
            hooks_dir.mkdir(exist_ok=True)

            # Path to our hook templates
            hooks_source_dir = Path(__file__).parent.parent / "git-hooks"
            if not hooks_source_dir.exists():
                logger.error("Git hooks source directory not found", path=str(hooks_source_dir))
                return False

            # Hooks to install
            hooks = ["post-commit", "post-merge", "post-checkout", "post-fetch"]
            installed = []

            for hook_name in hooks:
                source = hooks_source_dir / hook_name
                dest = hooks_dir / hook_name

                if not source.exists():
                    logger.warning(f"Hook {hook_name} not found in source", path=str(source))
                    continue

                # Backup existing hook if present
                if dest.exists():
                    backup = dest.with_suffix(".backup")
                    shutil.copy2(dest, backup)
                    logger.info(f"Backed up existing {hook_name} hook")

                # Copy hook
                shutil.copy2(source, dest)

                # Make executable
                st = os.stat(dest)
                os.chmod(dest, st.st_mode | stat.S_IEXEC)

                installed.append(hook_name)
                logger.info(f"Installed {hook_name} hook")

            if not installed:
                logger.error("No Git hooks were installed")
                return False

            print_info(f"Installed {len(installed)} Git hooks: {', '.join(installed)}")
            print_info("Git hooks will notify ACOLYTE of code changes automatically")

            return True

        except PermissionError as e:
            print_error("Permission denied while installing Git hooks")
            logger.error("Permission error installing hooks", error=str(e))
            return False
        except Exception as e:
            print_error(f"Unexpected error installing Git hooks: {e}")
            logger.error("Error installing Git hooks", error=str(e))
            return False

    def show_next_steps(self):
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

        # Show tips based on detected project type
        print(f"\n{Colors.YELLOW}💡 Tips:{Colors.ENDC}")

        if sys.platform == "win32":
            print("• Make sure Docker Desktop is running")
            print("• Run PowerShell as Administrator if you encounter permission issues")
        else:
            print("• Make sure Docker daemon is running")
            print("• You may need 'sudo' for Docker commands")

        print(f"\n{Colors.CYAN}For help:{Colors.ENDC} acolyte --help")


def main():
    """Main entry point"""
    try:
        # Show logo with animation
        print(ACOLYTE_LOGO)
        animate_text(
            f"{Colors.CYAN}{Colors.BOLD}ACOLYTE INIT - Quick Project Setup{Colors.ENDC}",
            duration=1.0,
        )
        print("\n")

        # Get project path (current directory)
        project_path = Path.cwd()

        print_info(f"Initializing ACOLYTE in: {project_path}")

        # Step 1: Validate project directory
        print_step(1, 6, "Validating Project Directory")
        is_valid, project_type = ProjectValidator.is_valid_project_directory(project_path)

        if not is_valid:
            print_error(f"Invalid project directory: {project_type}")
            print_warning(
                "This does not look like a valid project directory. Please create your project (e.g., with git, poetry, npm, etc.) and re-run 'acolyte init'."
            )
            sys.exit(1)

        print_success(f"Valid project detected: {project_type}")

        # Step 2: Check if already initialized
        print_step(2, 6, "Checking Initialization Status")
        init_instance = AcolyteInit(project_path)
        already_init, existing_id = init_instance.check_already_initialized()

        if already_init:
            print_warning(f"Project already initialized with ID: {existing_id}")
            print_info("Run 'acolyte install' to configure the project")
            sys.exit(0)

        print_success("Project not yet initialized")

        # Step 3: Check and install dependencies
        print_step(3, 6, "Checking System Dependencies")
        deps_ok, deps_results = DependencyChecker.check_and_install_all()

        if not deps_ok:
            print_error(
                "Some required dependencies are missing. Please install them and re-run 'acolyte init' in your project."
            )
            sys.exit(1)

        print_success("All dependencies satisfied")

        # Step 4: Check Git repository
        print_step(4, 6, "Checking Git Repository")
        if not ProjectValidator.ensure_git_initialized(project_path):
            print_error(
                "Could not initialize git repository. Please resolve the issue and re-run 'acolyte init'."
            )
            sys.exit(1)

        print_success("Git repository detected")

        # Step 5: Create project structure
        print_step(5, 6, "Creating Project Structure")
        show_spinner("Setting up ACOLYTE structure", 1.5)

        if not init_instance.create_project_structure():
            print_error(
                "Could not create project structure. Please resolve the issue above and re-run 'acolyte init'."
            )
            sys.exit(1)

        print_success("Project structure created")

        # Step 6: Install Git hooks
        print_step(6, 6, "Installing Git Hooks")
        if not init_instance.install_git_hooks():
            print_error(
                "Could not install Git hooks. Please resolve the issue above and re-run 'acolyte init'."
            )
            sys.exit(1)

        print_success("Git hooks installed")

        # Show completion and next steps
        init_instance.show_next_steps()

        # Log success
        logger.info("ACOLYTE initialized successfully", project_path=str(project_path))

    except KeyboardInterrupt:
        print_warning("\nInitialization cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print_error(f"Initialization failed: {e}")
        print_warning(
            "An unexpected error occurred. Please check the logs for more details and re-run 'acolyte init'."
        )
        logger.error("Fatal error during initialization", error=str(e), include_trace=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
