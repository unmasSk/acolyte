"""
ACOLYTE Project Installation Module

Provides classes and functions for installing and configuring ACOLYTE services.
Interactive installation with hardware detection, model selection, and Docker setup.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any

import yaml

from acolyte.core.logging import logger
from acolyte.core.exceptions import ConfigurationError
from acolyte.install.common import (
    Colors,
    DockerGenerator,
    ModelRecommender,
    PortManager,
    SystemDetector,
    animate_text,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
    show_spinner,
    validate_port,
)
from acolyte.install.common.config_template import get_complete_config
from acolyte.install.resources_manager import get_modelfile
from acolyte.install.database import DatabaseInitializer


def safe_input(prompt: str, default: str = "") -> str:
    """Safe input that handles terminal issues properly"""
    import sys

    # For Windows, flush input/output streams
    if sys.platform == "win32":
        sys.stdout.flush()
        sys.stdin.flush()

    # Try to get input multiple times if needed
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            result = input(prompt)
            return result.strip() if result else default
        except EOFError:
            # This is normal when piped input ends
            return default
        except KeyboardInterrupt:
            # User pressed Ctrl+C - re-raise this
            raise
        except Exception as e:
            # Log the actual error for debugging
            logger.warning(f"Input error (attempt {attempt + 1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                # Try again
                import time

                time.sleep(0.1)
                continue
            else:
                # Last attempt failed - ask user what to do
                print(f"\n{Colors.YELLOW}Warning: Input error occurred{Colors.ENDC}")
                print(f"Press Enter to use default value: {default}")
                try:
                    input()
                except EOFError:
                    pass
                return default
    return default


class InstallationCancelled(Exception):
    """Raised when user cancels installation"""


class ProjectInfoCollector:
    """Collect project information interactively"""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.project_name = project_path.name
        self._load_project_info()

    def _load_project_info(self):
        """Load project info from .acolyte.project file"""
        project_file = self.project_path / ".acolyte.project"
        if project_file.exists():
            try:
                with open(project_file) as f:
                    data = json.load(f)
                    self.project_name = data.get('name', self.project_name)
                    self.user_name = data.get('user', 'developer')
            except Exception as e:
                logger.error("Failed to load project info", error=str(e))
                self.user_name = 'developer'
        else:
            self.user_name = 'developer'

    def detect_and_confirm_stack(self) -> Dict[str, List[str]]:
        """Detect technology stack and allow user to confirm/modify"""
        print_header("🔍 Technology Stack Detection")

        print_info("Analyzing project files...")
        show_spinner("Detecting technologies", 1.5)

        # Auto-detect stack
        stack = self._auto_detect_stack()

        # Show detected stack
        print(f"\n{Colors.GREEN}Detected technologies:{Colors.ENDC}")

        if stack["backend"]:
            print(f"\n{Colors.CYAN}Backend:{Colors.ENDC}")
            for tech in stack["backend"]:
                print(f"  • {tech}")

        if stack["frontend"]:
            print(f"\n{Colors.CYAN}Frontend:{Colors.ENDC}")
            for tech in stack["frontend"]:
                print(f"  • {tech}")

        if stack["database"]:
            print(f"\n{Colors.CYAN}Database:{Colors.ENDC}")
            for tech in stack["database"]:
                print(f"  • {tech}")

        if stack["tools"]:
            print(f"\n{Colors.CYAN}Tools:{Colors.ENDC}")
            for tech in stack["tools"]:
                print(f"  • {tech}")

        # Ask if user wants to modify
        print(f"\n{Colors.CYAN}Is this correct?{Colors.ENDC}")
        modify = safe_input("Press Enter to accept, or 'e' to edit: ", default="").lower()

        if modify == 'e':
            stack = self._edit_stack(stack)

        return stack

    def _auto_detect_stack(self) -> Dict[str, List[str]]:
        """Auto-detect technology stack from project files"""
        stack: Dict[str, List[str]] = {"backend": [], "frontend": [], "database": [], "tools": []}

        # Backend detection
        if (self.project_path / "requirements.txt").exists() or (
            self.project_path / "pyproject.toml"
        ).exists():
            stack["backend"].append("Python")

            # Check for specific frameworks
            if (self.project_path / "manage.py").exists():
                stack["backend"].append("Django")
            elif any(
                (self.project_path / name).exists()
                for name in ["app.py", "application.py", "wsgi.py"]
            ):
                stack["backend"].append("Flask/FastAPI")

        if (self.project_path / "package.json").exists():
            try:
                with open(self.project_path / "package.json") as f:
                    pkg = json.load(f)
                    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

                    # Backend frameworks
                    if "express" in deps:
                        stack["backend"].append("Express.js")
                    elif "fastify" in deps:
                        stack["backend"].append("Fastify")
                    elif "@nestjs/core" in deps:
                        stack["backend"].append("NestJS")

                    # Frontend frameworks
                    if "react" in deps:
                        stack["frontend"].append("React")
                    if "vue" in deps:
                        stack["frontend"].append("Vue.js")
                    if "@angular/core" in deps:
                        stack["frontend"].append("Angular")
                    if "svelte" in deps:
                        stack["frontend"].append("Svelte")
                    if "next" in deps:
                        stack["frontend"].append("Next.js")

                    # Languages
                    if "typescript" in deps or "@types/node" in deps:
                        stack["frontend"].append("TypeScript")
                    elif stack["frontend"] or stack["backend"]:
                        stack["frontend"].append("JavaScript")

                    # Databases
                    if any(db in deps for db in ["pg", "postgres", "postgresql"]):
                        stack["database"].append("PostgreSQL")
                    if any(db in deps for db in ["mysql", "mysql2"]):
                        stack["database"].append("MySQL")
                    if "mongodb" in deps or "mongoose" in deps:
                        stack["database"].append("MongoDB")
                    if "redis" in deps:
                        stack["database"].append("Redis")
            except Exception as e:
                logger.error("Error reading package.json", error=str(e))

        # More backend languages
        if (self.project_path / "go.mod").exists():
            stack["backend"].append("Go")
        if (self.project_path / "Cargo.toml").exists():
            stack["backend"].append("Rust")
        if (self.project_path / "pom.xml").exists() or (
            self.project_path / "build.gradle"
        ).exists():
            stack["backend"].append("Java")
        if (self.project_path / "composer.json").exists():
            stack["backend"].append("PHP")
        if (self.project_path / "Gemfile").exists():
            stack["backend"].append("Ruby")

        # Database files
        if any(
            (self.project_path / f).exists() for f in ["docker-compose.yml", "docker-compose.yaml"]
        ):
            # Parse docker-compose for databases
            try:
                compose_file = None
                for fname in ["docker-compose.yml", "docker-compose.yaml"]:
                    if (self.project_path / fname).exists():
                        compose_file = self.project_path / fname
                        break

                if compose_file:
                    with open(compose_file) as f:
                        compose = yaml.safe_load(f)
                        services = compose.get("services", {})

                        if any("postgres" in s for s in services):
                            stack["database"].append("PostgreSQL")
                        if any("mysql" in s or "mariadb" in s for s in services):
                            stack["database"].append("MySQL/MariaDB")
                        if any("mongo" in s for s in services):
                            stack["database"].append("MongoDB")
                        if any("redis" in s for s in services):
                            stack["database"].append("Redis")
            except Exception as e:
                logger.error("Error reading docker-compose.yml", error=str(e))

        # Tools
        if (self.project_path / ".git").exists():
            stack["tools"].append("Git")
        if (self.project_path / "Dockerfile").exists() or (
            self.project_path / "docker-compose.yml"
        ).exists():
            stack["tools"].append("Docker")
        if (self.project_path / ".github" / "workflows").exists():
            stack["tools"].append("GitHub Actions")
        if (self.project_path / ".gitlab-ci.yml").exists():
            stack["tools"].append("GitLab CI")
        if (self.project_path / "Jenkinsfile").exists():
            stack["tools"].append("Jenkins")

        # Remove duplicates
        for key in stack:
            stack[key] = list(dict.fromkeys(stack[key]))

        return stack

    def _edit_stack(self, stack: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Allow user to edit detected stack"""
        print_header("✏️ Edit Technology Stack")

        # Common technologies
        common_tech = {
            "backend": ["Python", "Node.js", "Go", "Rust", "Java", "C#", "PHP", "Ruby", "Elixir"],
            "frontend": [
                "React",
                "Vue.js",
                "Angular",
                "Svelte",
                "Next.js",
                "Nuxt.js",
                "HTML/CSS",
                "TypeScript",
                "JavaScript",
            ],
            "database": [
                "PostgreSQL",
                "MySQL",
                "MongoDB",
                "Redis",
                "SQLite",
                "Cassandra",
                "Elasticsearch",
            ],
            "tools": [
                "Git",
                "Docker",
                "Kubernetes",
                "Jenkins",
                "GitHub Actions",
                "GitLab CI",
                "CircleCI",
            ],
        }

        for category in ["backend", "frontend", "database", "tools"]:
            print(f"\n{Colors.CYAN}{category.capitalize()} Technologies:{Colors.ENDC}")
            print(f"Current: {', '.join(stack[category]) if stack[category] else 'None'}")
            print("\nAvailable options:")

            for i, tech in enumerate(common_tech[category], 1):
                status = "✓" if tech in stack[category] else " "
                print(f"  {i}. [{status}] {tech}")

            print(
                f"\n{Colors.YELLOW}Enter numbers to toggle (comma-separated), or press Enter to continue:{Colors.ENDC}"
            )
            choices = safe_input("> ", default="")

            if choices:
                for choice in choices.split(','):
                    try:
                        idx = int(choice.strip()) - 1
                        if 0 <= idx < len(common_tech[category]):
                            tech = common_tech[category][idx]
                            if tech in stack[category]:
                                stack[category].remove(tech)
                            else:
                                stack[category].append(tech)
                    except (ValueError, IndexError):
                        pass

        return stack


class AdvancedConfiguration:
    """Advanced configuration options"""

    def __init__(self, hardware: Dict[str, Any]):
        self.hardware = hardware

    def configure_model(self) -> Dict[str, Any]:
        """Configure AI model selection"""
        print_header("🤖 AI Model Configuration")

        ram_gb = self.hardware["ram_gb"]
        gpu_info = self.hardware.get("gpu")

        # Get recommendation
        recommended = ModelRecommender.recommend(ram_gb, gpu_info)

        print(f"\n{Colors.CYAN}Available models for your system:{Colors.ENDC}")
        print(f"System RAM: {ram_gb}GB")
        if gpu_info:
            print(f"GPU: {gpu_info['name']} ({gpu_info['vram_mb']}MB VRAM)")

        models = []
        default_choice = "1"

        print(f"\n{Colors.YELLOW}Models:{Colors.ENDC}")
        for i, (key, info) in enumerate(ModelRecommender.MODELS.items(), 1):
            is_recommended = key == recommended

            model_name = info.get("ollama_model", f"qwen2.5-coder:{key}")
            status = f" {Colors.GREEN}(recommended){Colors.ENDC}" if is_recommended else ""
            compatibility = self._check_model_compatibility(info, ram_gb, gpu_info)

            print(f"\n  {i}. {Colors.BOLD}{model_name}{Colors.ENDC}{status}")
            print(f"     Size: {info['size']} parameters")
            print(f"     RAM required: {info['ram_min']}GB minimum")
            context_val = info.get('context')
            if isinstance(context_val, (int, float)):
                print(f"     Context window: {int(context_val) // 1024}k tokens")
            else:
                print("     Context window: unknown")
            print(f"     {compatibility}")

            models.append((key, info))
            if is_recommended:
                default_choice = str(i)

        # Custom model option
        print(f"\n  {len(models) + 1}. {Colors.BOLD}Custom model{Colors.ENDC}")
        print("     Specify your own Ollama model")

        while True:
            choice = safe_input(
                f"\n{Colors.CYAN}Select model [{default_choice}]: {Colors.ENDC}",
                default=default_choice,
            )

            try:
                idx = int(choice) - 1

                if idx == len(models):  # Custom model
                    return self._configure_custom_model()
                elif 0 <= idx < len(models):
                    selected_key, selected_info = models[idx]

                    # Warn if system doesn't meet requirements
                    if ram_gb < selected_info["ram_min"]:
                        print_warning(
                            f"Your system has {ram_gb}GB RAM but this model requires {selected_info['ram_min']}GB"
                        )
                        confirm = safe_input("Continue anyway? [y/N]: ", default="n").lower()
                        if confirm != 'y':
                            continue

                    return {
                        "name": selected_info.get("ollama_model", f"qwen2.5-coder:{selected_key}"),
                        "size": selected_info["size"],
                        "context_size": selected_info["context"],
                        "ram_required": selected_info["ram_min"],
                    }
                else:
                    print_warning("Invalid selection")
            except ValueError:
                print_warning("Please enter a number")

    def _check_model_compatibility(
        self, model_info: Dict, ram_gb: int, gpu_info: Optional[Dict]
    ) -> str:
        """Check model compatibility with system"""
        ram_req = model_info["ram_min"]

        if ram_gb >= ram_req * 1.5:
            return f"{Colors.GREEN}✓ Excellent performance expected{Colors.ENDC}"
        elif ram_gb >= ram_req:
            return f"{Colors.YELLOW}✓ Should run well{Colors.ENDC}"
        elif ram_gb >= ram_req * 0.8:
            return f"{Colors.YELLOW}⚠ May run with reduced performance{Colors.ENDC}"
        else:
            return f"{Colors.RED}✗ Insufficient RAM - not recommended{Colors.ENDC}"

    def _configure_custom_model(self) -> Dict[str, Any]:
        """Configure a custom Ollama model"""
        print_header("🔧 Custom Model Configuration")

        print(f"\n{Colors.CYAN}Enter custom Ollama model name:{Colors.ENDC}")
        print(f"{Colors.YELLOW}Examples: llama2:13b, mixtral:8x7b, codellama:34b{Colors.ENDC}")

        model_name = safe_input("Model name: ", default="")
        if not model_name:
            print_warning("Model name cannot be empty")
            return self.configure_model()

        # Get context size
        print(f"\n{Colors.CYAN}Context size (in tokens):{Colors.ENDC}")
        print(f"{Colors.YELLOW}Common values: 4096, 8192, 16384, 32768{Colors.ENDC}")

        while True:
            try:
                context_size = int(safe_input("Context size [32768]: ", default="32768"))
                if context_size < 512:
                    print_warning("Context size should be at least 512")
                    continue
                break
            except ValueError:
                print_warning("Please enter a number")

        # Estimate RAM requirement
        print(f"\n{Colors.CYAN}Minimum RAM requirement (GB):{Colors.ENDC}")

        while True:
            try:
                ram_required = int(safe_input("RAM required [8]: ", default="8"))
                if ram_required < 1:
                    print_warning("RAM requirement should be at least 1GB")
                    continue
                break
            except ValueError:
                print_warning("Please enter a number")

        return {
            "name": model_name,
            "size": "custom",
            "context_size": context_size,
            "ram_required": ram_required,
        }

    def configure_ports(self) -> Dict[str, int]:
        """Configure service ports with smart defaults"""
        print_header("🔌 Port Configuration")

        print(f"\n{Colors.CYAN}ACOLYTE uses dedicated port ranges to avoid conflicts:{Colors.ENDC}")
        print("• Weaviate (vector DB): 42080-42099")
        print("• Ollama (AI model): 42434-42453")
        print("• Backend API: 42000-42019")

        port_manager = PortManager()

        # Try to find available ports automatically
        try:
            auto_weaviate, auto_ollama, auto_backend = port_manager.find_available_ports()
            print_success("\nFound available ports automatically:")
            print(f"  Weaviate: {auto_weaviate}")
            print(f"  Ollama: {auto_ollama}")
            print(f"  Backend: {auto_backend}")

            use_auto = safe_input(
                f"\n{Colors.CYAN}Use these ports? [Y/n]: {Colors.ENDC}", default="y"
            ).lower()
            if use_auto != 'n':
                return {"weaviate": auto_weaviate, "ollama": auto_ollama, "backend": auto_backend}
        except RuntimeError:
            print_warning("Could not find all available ports automatically")
            auto_weaviate, auto_ollama, auto_backend = 42080, 42434, 42000

        # Manual configuration
        print(f"\n{Colors.CYAN}Configure ports manually:{Colors.ENDC}")
        ports = {}

        # Weaviate
        ports["weaviate"] = self._configure_single_port(
            "Weaviate", auto_weaviate, port_manager, "weaviate"
        )

        # Ollama
        ports["ollama"] = self._configure_single_port("Ollama", auto_ollama, port_manager, "ollama")

        # Backend
        ports["backend"] = self._configure_single_port(
            "Backend API", auto_backend, port_manager, "backend"
        )

        return ports

    def _configure_single_port(
        self, service: str, default: int, port_manager: PortManager, service_key: str
    ) -> int:
        """Configure a single port with validation"""
        while True:
            port_input = safe_input(f"{service} port [{default}]: ", default=str(default))

            if not port_input:
                port = default
            else:
                try:
                    port = int(port_input)
                except ValueError:
                    print_warning("Please enter a valid port number")
                    continue

            # Validate port
            if not validate_port(port):
                # Suggest next available port
                suggested = PortManager.find_next_available(port)
                if suggested:
                    print_warning(f"Port {port} is not available. Suggested: {suggested}")
                    use_suggested = safe_input("Use suggested port? [Y/n]: ", default="y").lower()
                    if use_suggested != 'n':
                        return suggested
                else:
                    print_error(f"Port {port} is not available and no alternatives found")
                continue

            return port

    def configure_resources(self, hardware: Dict[str, Any]) -> Dict[str, Any]:
        """Configure Docker resource limits"""
        print_header("🐳 Resource Configuration")

        ram_gb = hardware["ram_gb"]
        cpu_threads = hardware["cpu_threads"]

        print(f"\n{Colors.CYAN}System resources:{Colors.ENDC}")
        print(f"• Total RAM: {ram_gb}GB")
        print(f"• CPU threads: {cpu_threads}")

        # Docker memory
        default_memory = max(4, ram_gb // 2)
        print(f"\n{Colors.CYAN}Docker memory limit (GB):{Colors.ENDC}")
        print(f"{Colors.YELLOW}Recommended: 50% of system RAM{Colors.ENDC}")

        while True:
            memory_input = safe_input(
                f"Memory limit [{default_memory}]: ", default=str(default_memory)
            )
            if not memory_input:
                docker_memory = default_memory
                break

            try:
                docker_memory = int(memory_input)
                if docker_memory < 2:
                    print_warning("Minimum 2GB required for ACOLYTE")
                    continue
                if docker_memory > ram_gb:
                    print_warning(f"Cannot exceed system RAM ({ram_gb}GB)")
                    continue
                break
            except ValueError:
                print_warning("Please enter a number")

        # Docker CPUs
        default_cpus = max(2, cpu_threads // 2)
        print(f"\n{Colors.CYAN}Docker CPU limit:{Colors.ENDC}")
        print(f"{Colors.YELLOW}Recommended: 50% of CPU threads{Colors.ENDC}")

        while True:
            cpu_input = safe_input(f"CPU limit [{default_cpus}]: ", default=str(default_cpus))
            if not cpu_input:
                docker_cpus = default_cpus
                break

            try:
                docker_cpus = int(cpu_input)
                if docker_cpus < 1:
                    print_warning("Minimum 1 CPU required")
                    continue
                if docker_cpus > cpu_threads:
                    print_warning(f"Cannot exceed system threads ({cpu_threads})")
                    continue
                break
            except ValueError:
                print_warning("Please enter a number")

        return {
            "memory_limit": f"{docker_memory}G",
            "cpu_limit": str(docker_cpus),
            "gpu_enabled": bool(hardware.get("gpu")),
        }


class LanguageConfiguration:
    """Configure language-specific settings"""

    LINTERS = {
        "python": [
            {"linter": "ruff", "formatter": "black", "name": "Ruff + Black (fast & modern)"},
            {
                "linter": "flake8",
                "formatter": "autopep8",
                "name": "Flake8 + autopep8 (traditional)",
            },
            {"linter": "pylint", "formatter": "yapf", "name": "Pylint + YAPF (comprehensive)"},
            {"linter": "mypy", "formatter": "isort", "name": "MyPy + isort (type checking)"},
        ],
        "javascript": [
            {"linter": "eslint", "formatter": "prettier", "name": "ESLint + Prettier (standard)"},
            {"linter": "standard", "formatter": "standard", "name": "StandardJS (opinionated)"},
            {"linter": "jshint", "formatter": "js-beautify", "name": "JSHint + Beautify (classic)"},
        ],
        "typescript": [
            {
                "linter": "eslint",
                "formatter": "prettier",
                "name": "ESLint + Prettier (recommended)",
            },
            {"linter": "tslint", "formatter": "prettier", "name": "TSLint + Prettier (legacy)"},
            {"linter": "deno", "formatter": "deno", "name": "Deno lint + fmt (modern)"},
        ],
        "go": [
            {
                "linter": "golangci-lint",
                "formatter": "gofmt",
                "name": "golangci-lint + gofmt (standard)",
            },
            {"linter": "go-vet", "formatter": "goimports", "name": "go vet + goimports"},
            {
                "linter": "staticcheck",
                "formatter": "gofumpt",
                "name": "staticcheck + gofumpt (strict)",
            },
        ],
        "rust": [
            {"linter": "clippy", "formatter": "rustfmt", "name": "Clippy + rustfmt (official)"},
            {"linter": "rust-analyzer", "formatter": "rustfmt", "name": "rust-analyzer + rustfmt"},
        ],
    }

    def __init__(self, project_path: Path):
        self.project_path = project_path

    def detect_languages(self) -> List[str]:
        """Detect programming languages in project"""
        languages = set()
        extensions_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".h": "c",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
        }

        try:
            for ext, lang in extensions_map.items():
                if list(self.project_path.rglob(f"*{ext}")):
                    languages.add(lang)
        except Exception as e:
            logger.error("Error detecting languages", error=str(e))

        return sorted(list(languages))

    def configure_linters(self, languages: List[str]) -> Dict[str, Dict[str, str]]:
        """Configure linters for detected languages"""
        print_header("🔧 Code Quality Tools Configuration")

        if not languages:
            print_info("No programming languages detected")
            return {}

        print(f"\n{Colors.CYAN}Detected languages:{Colors.ENDC} {', '.join(languages)}")

        configure = safe_input(
            f"\n{Colors.CYAN}Configure linters and formatters? [Y/n]: {Colors.ENDC}", default="y"
        ).lower()
        if configure == 'n':
            return {}

        linter_config = {}

        for lang in languages:
            if lang not in self.LINTERS:
                continue

            print(f"\n{Colors.BOLD}Configuration for {lang.upper()}:{Colors.ENDC}")
            options = self.LINTERS[lang]

            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt['name']}")

            print(f"  {len(options) + 1}. Skip (no linting for {lang})")

            while True:
                choice = safe_input("\nSelect option [1]: ", default="1")

                try:
                    idx = int(choice) - 1
                    if idx == len(options):  # Skip
                        break
                    elif 0 <= idx < len(options):
                        selected = options[idx]
                        linter_config[lang] = {
                            "linter": selected["linter"],
                            "formatter": selected["formatter"],
                        }
                        print_success(f"Selected: {selected['name']}")
                        break
                    else:
                        print_warning("Invalid option")
                except ValueError:
                    print_warning("Please enter a number")

        return linter_config

    def configure_ignore_patterns(self) -> List[str]:
        """Configure additional ignore patterns"""
        print_header("📁 File Exclusion Configuration")

        print(f"\n{Colors.CYAN}ACOLYTE automatically ignores common patterns:{Colors.ENDC}")
        print("• Version control: .git/, .svn/")
        print("• Dependencies: node_modules/, venv/, vendor/")
        print("• Build outputs: dist/, build/, target/")
        print("• IDE files: .vscode/, .idea/")
        print("• Cache: __pycache__/, .cache/")

        print(f"\n{Colors.CYAN}Add custom folders/patterns to ignore?{Colors.ENDC}")
        print(f"{Colors.YELLOW}Examples: logs/, *.log, temp/, secrets/{Colors.ENDC}")

        custom_patterns = []
        print("\nEnter patterns (one per line, empty line to finish):")

        while True:
            pattern = safe_input("> ", default="")
            if not pattern:
                break
            custom_patterns.append(pattern)
            print_success(f"Added: {pattern}")

        return custom_patterns


class ProjectInstaller:
    """Main installer class that orchestrates the installation process"""

    def __init__(self, project_path: Path, global_dir: Optional[Path] = None):
        self.project_path = project_path
        self.project_file = project_path / ".acolyte.project"
        self.global_dir = global_dir or Path.home() / ".acolyte"

        # Load project info
        if not self.project_file.exists():
            raise FileNotFoundError("Project not initialized. Run 'acolyte init' first.")

        with open(self.project_file) as f:
            project_data = json.load(f)

        self.project_id = project_data["project_id"]
        self.project_name = project_data.get("name", self.project_path.name)
        self.user_name = project_data.get("user", "developer")
        self.project_global_dir = self.global_dir / "projects" / self.project_id
        self.config_path = self.project_global_dir / ".acolyte"

    async def run(self) -> bool:
        """
        Run the complete installation process

        Returns:
            True if successful, False otherwise
        """
        try:
            # Show header (logo already shown in cli.py)
            animate_text(
                f"{Colors.CYAN}{Colors.BOLD}Interactive Configuration{Colors.ENDC}",
                duration=1.0,
            )
            print("\n")

            # Check if already configured
            if self.config_path.exists():
                print_warning("ACOLYTE is already configured for this project")

                reconfigure = safe_input("Reconfigure? [y/N]: ", default="n").lower()

                if reconfigure != 'y':
                    print_info("Installation cancelled - keeping existing configuration")
                    return False  # Not an error, user chose to keep existing config

            # Collect all configuration
            config = await self._collect_configuration()

            # Show summary
            self._show_configuration_summary(config)

            # Confirm
            confirm = safe_input(
                f"\n{Colors.YELLOW}Proceed with installation? [Y/n]: {Colors.ENDC}", default="y"
            ).lower()

            if confirm == 'n':
                print_info("Installation cancelled by user")
                return False  # Not an error, user chose to cancel

            # Save configuration
            print_header("💾 Saving Configuration")
            self._save_configuration(config)

            # Generate Docker files
            print_header("🐳 Generating Docker Infrastructure")
            self._generate_docker_files(config)

            # Generate Modelfile
            print_header("🤖 Creating Model Configuration")
            self._generate_modelfile(config)

            # Initialize database
            print_header("🗄️ Initializing Database")
            self._initialize_database(config)

            # Show completion
            self._show_completion(config)

            return True

        except Exception as e:
            print_error(f"Installation failed: {e}")
            logger.error("Installation error", error=str(e), include_trace=True)
            return False

    async def _collect_configuration(self) -> Dict[str, Any]:
        """Collect all configuration from user"""
        # Hardware detection
        print_header("🖥️ Hardware Detection")
        print_info("Detecting system capabilities...")
        show_spinner("Analyzing hardware", 1.5)

        detector = SystemDetector()
        os_name, os_version = detector.detect_os()
        cpu_info = detector.detect_cpu()
        ram_gb = detector.detect_memory()
        gpu_info = detector.detect_gpu()
        disk_gb = detector.detect_disk_space()

        hardware = {
            "os": os_name,
            "os_version": os_version,
            "cpu_cores": cpu_info["cores"],
            "cpu_threads": cpu_info["threads"],
            "cpu_model": cpu_info["model"],
            "ram_gb": ram_gb,
            "disk_free_gb": disk_gb,
        }

        if gpu_info:
            hardware["gpu"] = gpu_info

        # Show detected hardware
        print_success("Hardware detected:")
        print(f"  OS: {os_name} {os_version}")
        print(
            f"  CPU: {cpu_info['model']} ({cpu_info['cores']} cores, {cpu_info['threads']} threads)"
        )
        print(f"  RAM: {ram_gb}GB")
        if gpu_info:
            print(f"  GPU: {gpu_info['name']} ({gpu_info['vram_mb']}MB VRAM)")
        print(f"  Free disk: {disk_gb}GB")

        # Use project info already loaded from .acolyte.project
        user_name = self.user_name
        project_name = self.project_name
        project_description = f"ACOLYTE-powered {project_name} project"

        print_info(f"Project: {project_name}")
        print_info(f"User: {user_name}")

        # Detect and confirm stack
        info_collector = ProjectInfoCollector(self.project_path)
        detected_stack = info_collector.detect_and_confirm_stack()

        # Advanced configuration
        advanced = AdvancedConfiguration(hardware)

        # Model selection
        model_config = advanced.configure_model()

        # Port configuration
        ports = advanced.configure_ports()

        # Resource limits
        docker_config = advanced.configure_resources(hardware)

        # Language configuration
        lang_config = LanguageConfiguration(self.project_path)
        languages = lang_config.detect_languages()
        linting_config = lang_config.configure_linters(languages)
        custom_ignore = lang_config.configure_ignore_patterns()

        # Detect code style
        code_style = self._detect_code_style()

        # Build complete configuration
        return get_complete_config(
            project_id=self.project_id,
            project_name=project_name,
            project_path=str(self.project_path),
            project_user=user_name,
            project_description=project_description,
            ports=ports,
            hardware=hardware,
            model=model_config,
            linting=linting_config,
            ignore_custom=custom_ignore,
            docker=docker_config,
            detected_stack=detected_stack,
            code_style=code_style,
        )

    def _detect_code_style(self) -> Dict[str, Any]:
        """Detect code style preferences"""
        code_style = {}

        # Python
        if (self.project_path / "pyproject.toml").exists():
            code_style["python"] = {
                "formatter": "black",
                "linter": "ruff",
                "line_length": 100,
                "quotes": "double",
                "docstring_style": "google",
                "type_checking": "strict",
            }

        # JavaScript/TypeScript
        if (self.project_path / ".eslintrc.json").exists() or (
            self.project_path / ".eslintrc.js"
        ).exists():
            code_style["javascript"] = {
                "formatter": "prettier",
                "linter": "eslint",
                "semicolons": False,
                "quotes": "single",
                "indent": 2,
                "typescript": (self.project_path / "tsconfig.json").exists(),
            }

        # General
        code_style["general"] = {
            "indent_style": "spaces",
            "trim_trailing_whitespace": True,
            "insert_final_newline": True,
            "charset": "utf-8",
        }

        return code_style

    def _show_configuration_summary(self, config: Dict[str, Any]):
        """Show configuration summary"""
        print_header("📋 Configuration Summary")

        print(f"\n{Colors.BOLD}Project:{Colors.ENDC}")
        print(f"  Name: {config['project']['name']}")
        print(f"  User: {config['project']['user']}")
        print(f"  Path: {self.project_path}")

        print(f"\n{Colors.BOLD}Model:{Colors.ENDC}")
        print(f"  Name: {config['model']['name']}")
        print(f"  Context: {config['model']['context_size']} tokens")

        print(f"\n{Colors.BOLD}Services:{Colors.ENDC}")
        print(f"  Weaviate: localhost:{config['ports']['weaviate']}")
        print(f"  Ollama: localhost:{config['ports']['ollama']}")
        print(f"  Backend: localhost:{config['ports']['backend']}")

        print(f"\n{Colors.BOLD}Resources:{Colors.ENDC}")
        print(f"  Docker memory: {config['docker']['memory_limit']}")
        print(f"  Docker CPUs: {config['docker']['cpu_limit']}")

        if config.get('linting'):
            print(f"\n{Colors.BOLD}Linters:{Colors.ENDC}")
            for lang, tools in config['linting'].items():
                print(f"  {lang}: {tools['linter']} + {tools['formatter']}")

    def _save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            # Ensure directory exists
            self.project_global_dir.mkdir(parents=True, exist_ok=True)

            # Backup if exists
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix('.acolyte.backup')
                shutil.copy2(self.config_path, backup_path)
                print_info(f"Backed up existing config to: {backup_path}")

            # Save configuration
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            print_success(f"Configuration saved to: {self.config_path}")

        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def _generate_docker_files(self, config: Dict[str, Any]):
        """Generate Docker infrastructure files"""
        try:
            generator = DockerGenerator(config, self.project_global_dir)

            # Generate docker-compose.yml
            compose = generator.generate_compose()
            if generator.save_compose(compose):
                print_success("Docker Compose file generated")

            # Generate Dockerfile
            if generator.generate_global_dockerfile():
                print_success("Dockerfile generated")

        except Exception as e:
            raise ConfigurationError(f"Failed to generate Docker files: {e}")

    def _generate_modelfile(self, config: Dict[str, Any]):
        """Generate Ollama Modelfile"""
        try:
            # Get system prompt template from resources
            system_prompt_template = get_modelfile()
            if not system_prompt_template:
                raise FileNotFoundError("System prompt template not found in resources")

            # Replace variables in the template
            system_prompt = system_prompt_template.replace(
                "{project_name}", config['project']['name']
            ).replace("{project_user}", config['project']['user'])

            # Build the complete Modelfile
            modelfile_content = f"""FROM {config['model']['name']}

# System prompt for ACOLYTE
SYSTEM \"\"\"
{system_prompt}
\"\"\"

# Model parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx {config['model']['context_size']}
PARAMETER repeat_penalty 1.1
PARAMETER seed 42
"""

            modelfile_path = self.project_global_dir / "infra" / "Modelfile"
            modelfile_path.parent.mkdir(exist_ok=True)
            modelfile_path.write_text(modelfile_content, encoding='utf-8')

            print_success("Modelfile created")

        except Exception as e:
            raise ConfigurationError(f"Failed to create Modelfile: {e}")

    def _initialize_database(self, config: Dict[str, Any]):
        """Initialize SQLite and Weaviate databases"""
        try:
            # Create data directories first
            dreams_dir = self.project_global_dir / "data" / "dreams"
            dreams_dir.mkdir(parents=True, exist_ok=True)

            embeddings_dir = self.project_global_dir / "data" / "embeddings_cache"
            embeddings_dir.mkdir(parents=True, exist_ok=True)

            # Initialize databases using the migrated module
            db_initializer = DatabaseInitializer(
                project_path=self.project_path,
                project_id=self.project_id,
                global_dir=self.global_dir,
            )

            # Run database initialization
            # Skip Weaviate if Docker is not running yet
            success = db_initializer.run(skip_weaviate=True)

            if not success:
                raise ConfigurationError("Database initialization failed")

            print_info("Weaviate will be initialized when services start")
            logger.info("Database initialization complete")

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize database: {e}")

    def _show_completion(self, config: Dict[str, Any]):
        """Show installation completion message"""
        print_header("✨ Configuration Complete!")

        print(
            f"\n{Colors.GREEN}{Colors.BOLD}ACOLYTE has been configured successfully!{Colors.ENDC}"
        )

        backend_port = config['ports']['backend']

        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print(f"\n1. {Colors.GREEN}Start ACOLYTE services:{Colors.ENDC}")
        print(f"   {Colors.CYAN}acolyte start{Colors.ENDC}")
        print(
            f"   {Colors.YELLOW}This will start Docker containers and initialize Weaviate{Colors.ENDC}"
        )

        print(f"\n2. {Colors.GREEN}Wait for services to be ready:{Colors.ENDC}")
        print(f"   {Colors.CYAN}acolyte status{Colors.ENDC}")
        print(f"   {Colors.YELLOW}All services should show 'healthy'{Colors.ENDC}")

        print(f"\n3. {Colors.GREEN}Index your project (IMPORTANT):{Colors.ENDC}")
        print(f"   {Colors.CYAN}acolyte index{Colors.ENDC}")
        print(
            f"   {Colors.YELLOW}This may take several minutes depending on project size{Colors.ENDC}"
        )
        print(f"   {Colors.YELLOW}ACOLYTE needs to analyze all your code files first{Colors.ENDC}")

        print(f"\n4. {Colors.GREEN}Start coding with ACOLYTE:{Colors.ENDC}")
        print(f"   {Colors.YELLOW}Use the API at http://localhost:{backend_port}{Colors.ENDC}")
        print(f"   {Colors.YELLOW}Or integrate with your IDE{Colors.ENDC}")

        print(f"\n{Colors.BOLD}Important Notes:{Colors.ENDC}")
        print(f"  • {Colors.YELLOW}Docker must be running before 'acolyte start'{Colors.ENDC}")
        print(
            f"  • {Colors.YELLOW}First indexing is crucial - ACOLYTE won't work without it{Colors.ENDC}"
        )
        print(
            f"  • {Colors.YELLOW}Git hooks are installed - commits will auto-update the index{Colors.ENDC}"
        )

        print(f"\n{Colors.BOLD}Ready to code with ACOLYTE!{Colors.ENDC} 🚀")
