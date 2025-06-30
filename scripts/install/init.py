#!/usr/bin/env python3
"""
🤖 ACOLYTE INIT - Project Configuration Setup
Creates configuration in ~/.acolyte/projects/{project_id}/
Only creates .acolyte.project in the user's project
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

# Add common modules to path
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    ACOLYTE_LOGO,
    Colors,
    DockerGenerator,
    ModelRecommender,
    PortManager,
    SystemDetector,
    animate_text,
    print_error,
    print_header,
    print_info,
    print_step,
    print_success,
    print_warning,
    sanitize_yaml_string,
    show_spinner,
    validate_port,
    validate_project_name,
)

# Get environment variables from CLI
PROJECT_ID = os.environ.get('ACOLYTE_PROJECT_ID', '')
PROJECT_PATH = os.environ.get('ACOLYTE_PROJECT_PATH', '.')
GLOBAL_DIR = os.environ.get('ACOLYTE_GLOBAL_DIR', str(Path.home() / '.acolyte'))
PROJECT_NAME = os.environ.get('ACOLYTE_PROJECT_NAME', 'new-project')

# Configure logging
project_dir = Path(GLOBAL_DIR) / "projects" / PROJECT_ID
project_dir.mkdir(parents=True, exist_ok=True)

log_dir = project_dir / "data" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"init_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class LinterConfigurator:
    """Configures linters and formatters by language"""

    LINTERS = {
        "python": {
            "options": [
                {
                    "linter": "ruff",
                    "formatter": "black",
                    "name": "ruff + black (recommended)",
                },
                {
                    "linter": "flake8",
                    "formatter": "autopep8",
                    "name": "flake8 + autopep8",
                },
                {"linter": "pylint", "formatter": "yapf", "name": "pylint + yapf"},
                {"linter": "mypy", "formatter": "isort", "name": "mypy + isort"},
            ]
        },
        "javascript": {
            "options": [
                {
                    "linter": "eslint",
                    "formatter": "prettier",
                    "name": "eslint + prettier (recommended)",
                },
                {"linter": "standard", "formatter": "standard", "name": "standard"},
                {
                    "linter": "jshint",
                    "formatter": "js-beautify",
                    "name": "jshint + beautify",
                },
            ]
        },
        "typescript": {
            "options": [
                {
                    "linter": "eslint",
                    "formatter": "prettier",
                    "name": "eslint + prettier (recommended)",
                },
                {
                    "linter": "tslint",
                    "formatter": "prettier",
                    "name": "tslint + prettier",
                },
                {"linter": "deno", "formatter": "deno", "name": "deno lint + fmt"},
            ]
        },
        "go": {
            "options": [
                {
                    "linter": "golangci-lint",
                    "formatter": "gofmt",
                    "name": "golangci-lint + gofmt (recommended)",
                },
                {
                    "linter": "go-vet",
                    "formatter": "goimports",
                    "name": "go vet + goimports",
                },
                {
                    "linter": "staticcheck",
                    "formatter": "gofumpt",
                    "name": "staticcheck + gofumpt",
                },
            ]
        },
        "rust": {
            "options": [
                {
                    "linter": "clippy",
                    "formatter": "rustfmt",
                    "name": "clippy + rustfmt (recommended)",
                },
                {
                    "linter": "rust-analyzer",
                    "formatter": "rustfmt",
                    "name": "rust-analyzer + rustfmt",
                },
            ]
        },
    }

    @classmethod
    def detect_languages(cls, project_path: Path) -> List[str]:
        """Detect languages in the project"""
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
            ".c": "c_cpp",
            ".cpp": "c_cpp",
            ".cc": "c_cpp",
            ".h": "c_cpp",
            ".hpp": "c_cpp",
        }

        logger.info(f"Detecting programming languages in {project_path}")
        try:
            for ext, lang in extensions_map.items():
                files = list(project_path.rglob(f"*{ext}"))
                if files:
                    languages.add(lang)
                    logger.info(f"Detected {lang} ({len(files)} files with {ext} extension)")
        except Exception as e:
            logger.error(f"Error detecting languages: {e}")

        return sorted(list(languages))

    @classmethod
    def configure_linters(cls, languages: List[str]) -> Dict:
        """Configure linters for each detected language"""
        config = {}

        for lang in languages:
            if lang not in cls.LINTERS:
                logger.warning(f"No linter configuration available for {lang}")
                continue

            print(f"\n{Colors.CYAN}Configuration for {lang.upper()}:{Colors.ENDC}")
            options = cls.LINTERS[lang]["options"]

            for i, opt in enumerate(options, 1):
                print(f"  {i}) {opt['name']}")
            print("  4) skip - Omit this language")

            while True:
                try:
                    choice = input("Selection [1]: ").strip() or "1"

                    # Handle skip option
                    if choice == "4" or choice.lower() == "skip":
                        logger.info(f"Skipped linter configuration for {lang}")
                        break

                    idx = int(choice) - 1
                    if 0 <= idx < len(options):
                        selected = options[idx]
                        config[lang] = {
                            "linter": selected["linter"],
                            "formatter": selected["formatter"],
                        }
                        logger.info(
                            f"Selected {selected['linter']} and {selected['formatter']} for {lang}"
                        )
                        break
                    else:
                        print_warning("Invalid option")
                except (ValueError, KeyboardInterrupt):
                    print_warning("Using default option")
                    config[lang] = {
                        "linter": options[0]["linter"],
                        "formatter": options[0]["formatter"],
                    }
                    logger.info(
                        f"Using default {options[0]['linter']} and {options[0]['formatter']} for {lang}"
                    )
                    break

        return config


class AcolyteConfig:
    """Generates and validates ACOLYTE configuration"""

    def __init__(self, project_path: Path, project_dir: Path):
        self.project_path = Path(project_path)  # User's project
        self.project_dir = project_dir  # ~/.acolyte/projects/{id}/
        self.config = {
            "version": "1.0",
            "project": {
                "id": PROJECT_ID,
                "name": PROJECT_NAME,
                "path": str(self.project_path.resolve()),
            },
            "hardware": {},
            "model": {},
            "linting": {},
            "ignore": {},
            "docker": {},
            "ports": {},
        }

    def generate(self) -> Dict:
        """Generate complete configuration"""
        print_header("🤖 ACOLYTE INIT - Project Configuration")

        # 1. Port configuration
        print_step(1, 6, "Port Configuration")
        self._configure_ports()

        # 2. Detect hardware
        print_step(2, 6, "Hardware Detection")
        self._detect_hardware()

        # 3. Select model
        print_step(3, 6, "Model Selection")
        self._select_model()

        # 4. Configure linters
        print_step(4, 6, "Language Detection")
        self._configure_linters()

        # 5. Configure ignore patterns
        print_step(5, 6, "Exclusion Configuration")
        self._configure_ignore_patterns()

        # 6. Configure Docker
        print_step(6, 6, "Docker Configuration")
        self._configure_docker()

        return self.config

    def _configure_ports(self):
        """Configure service ports with automatic conflict resolution"""
        print(
            f"\n{Colors.CYAN}Configure service ports (press Enter for auto-selection):{Colors.ENDC}"
        )

        # Find available ports automatically
        port_manager = PortManager()

        # Try to find available ports in ACOLYTE range
        try:
            auto_weaviate, auto_ollama, auto_backend = port_manager.find_available_ports()
            print_info(
                f"Found available ports: Weaviate={auto_weaviate}, Ollama={auto_ollama}, Backend={auto_backend}"
            )
        except RuntimeError as e:
            print_warning(f"Could not find all available ports: {e}")
            # Use defaults anyway
            auto_weaviate, auto_ollama, auto_backend = 42080, 42434, 42000

        # Weaviate port
        port_input = input(f"Weaviate port [{auto_weaviate}]: ").strip()
        if port_input:
            try:
                weaviate_port = int(port_input)
                if not validate_port(weaviate_port):
                    # Find next available
                    suggested = port_manager.find_next_available(weaviate_port)
                    if suggested:
                        print_warning(f"Port {weaviate_port} is not available, using {suggested}")
                        weaviate_port = suggested
                    else:
                        print_warning(
                            f"Port {weaviate_port} is not available, using {auto_weaviate}"
                        )
                        weaviate_port = auto_weaviate
            except ValueError:
                print_warning(f"Invalid port, using {auto_weaviate}")
                weaviate_port = auto_weaviate
        else:
            weaviate_port = auto_weaviate

        # Ollama port
        port_input = input(f"Ollama port [{auto_ollama}]: ").strip()
        if port_input:
            try:
                ollama_port = int(port_input)
                if not validate_port(ollama_port):
                    # Find next available
                    suggested = port_manager.find_next_available(ollama_port)
                    if suggested:
                        print_warning(f"Port {ollama_port} is not available, using {suggested}")
                        ollama_port = suggested
                    else:
                        print_warning(f"Port {ollama_port} is not available, using {auto_ollama}")
                        ollama_port = auto_ollama
            except ValueError:
                print_warning(f"Invalid port, using {auto_ollama}")
                ollama_port = auto_ollama
        else:
            ollama_port = auto_ollama

        # Backend port
        port_input = input(f"Backend API port [{auto_backend}]: ").strip()
        if port_input:
            try:
                backend_port = int(port_input)
                if not validate_port(backend_port):
                    # Find next available
                    suggested = port_manager.find_next_available(backend_port)
                    if suggested:
                        print_warning(f"Port {backend_port} is not available, using {suggested}")
                        backend_port = suggested
                    else:
                        print_warning(f"Port {backend_port} is not available, using {auto_backend}")
                        backend_port = auto_backend
            except ValueError:
                print_warning(f"Invalid port, using {auto_backend}")
                backend_port = auto_backend
        else:
            backend_port = auto_backend

        self.config["ports"] = {
            "weaviate": weaviate_port,
            "ollama": ollama_port,
            "backend": backend_port,
        }

        logger.info(f"Port configuration: {self.config['ports']}")

        # Show final selection
        print_success(
            f"Selected ports - Weaviate: {weaviate_port}, Ollama: {ollama_port}, Backend: {backend_port}"
        )

    def _detect_hardware(self):
        """Detect and configure hardware"""
        print_info("Scanning hardware...")
        show_spinner("Analyzing system capabilities", 1.5)

        detector = SystemDetector()

        os_name, os_version = detector.detect_os()
        cpu_info = detector.detect_cpu()
        ram_gb = detector.detect_memory()
        gpu_info = detector.detect_gpu()
        disk_gb = detector.detect_disk_space()

        self.config["hardware"] = {
            "os": os_name,
            "os_version": os_version,
            "cpu_cores": cpu_info["cores"],
            "cpu_threads": cpu_info["threads"],
            "cpu_model": cpu_info["model"],
            "ram_gb": ram_gb,
            "disk_free_gb": disk_gb,
        }

        if gpu_info:
            self.config["hardware"]["gpu"] = {
                "type": gpu_info["type"],
                "name": gpu_info["name"],
                "vram_mb": gpu_info["vram_mb"],
            }

        # Show detected hardware
        print(f"\n{Colors.GREEN}Hardware detected:{Colors.ENDC}")
        print(f"📊 CPU: {cpu_info['model']} ({cpu_info['cores']} cores)")
        print(f"💾 RAM: {ram_gb} GB")
        if gpu_info:
            print(f"🎮 GPU: {gpu_info['name']} ({gpu_info['vram_mb']} MB VRAM)")
        else:
            print("🎮 GPU: Not detected")
        print(f"💿 Free space: {disk_gb} GB")

    def _select_model(self):
        """Select AI model based on hardware"""
        ram_gb = self.config["hardware"]["ram_gb"]
        gpu_info = self.config["hardware"].get("gpu")

        recommended_model = ModelRecommender.recommend(ram_gb, gpu_info)

        print(f"\n{Colors.CYAN}💡 Available models:{Colors.ENDC}")
        model_options = []
        default_choice = "1"

        for i, (key, info) in enumerate(ModelRecommender.MODELS.items(), 1):
            is_recommended = key == recommended_model
            rec_text = f" {Colors.GREEN}(recommended){Colors.ENDC}" if is_recommended else ""
            context_display = "32k"
            model_name = info.get("ollama_model", f"qwen2.5-coder:{key}")
            print(
                f"  {i}) {model_name} - {info['size']} model, "
                f"{info['ram_min']}GB RAM, {context_display} context{rec_text}"
            )
            model_options.append(key)
            if is_recommended:
                default_choice = str(i)

        while True:
            choice = (
                input(f"\n{Colors.CYAN}Select model [{default_choice}]: {Colors.ENDC}").strip()
                or default_choice
            )
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(model_options):
                    selected_model = model_options[idx]
                    model_info = ModelRecommender.MODELS[selected_model]
                    break
                else:
                    print_warning("Invalid option")
            except ValueError:
                print_warning("Please enter a number")

        self.config["model"] = {
            "name": model_info.get("ollama_model", f"qwen2.5-coder:{selected_model}"),
            "size": model_info["size"],
            "context_size": model_info["context"],
            "ram_required": model_info["ram_min"],
        }

        logger.info(f"Selected model: {self.config['model']['name']}")

    def _configure_linters(self):
        """Configure linters for detected languages"""
        print_info("\nDetecting languages in project...")
        show_spinner("Scanning project files", 1.0)

        languages = LinterConfigurator.detect_languages(self.project_path)

        if languages:
            print(f"Languages detected: {', '.join(languages)}")

            # Ask if user wants to configure linters
            configure = (
                input(f"\n{Colors.CYAN}Configure linters? [Y/n]: {Colors.ENDC}").strip().lower()
            )

            if configure != "n":
                self.config["linting"] = LinterConfigurator.configure_linters(languages)
            else:
                print_info("Skipping all linter configuration")
                self.config["linting"] = {}
        else:
            print_warning("No programming languages detected")

    def _configure_ignore_patterns(self):
        """Configure exclusion patterns"""
        print(f"\n{Colors.CYAN}Exclusion configuration:{Colors.ENDC}")

        # Base patterns by category
        self.config["ignore"] = {
            "services": ["ollama/", "weaviate/", ".git/"],
            "cache": [
                "__pycache__/",
                ".pytest_cache/",
                ".mypy_cache/",
                ".ruff_cache/",
                ".coverage",
                "htmlcov/",
                ".vscode/",
                ".idea/",
                ".cursor/",
            ],
            "dependencies": {
                "python": [
                    ".venv/",
                    "venv/",
                    "*.egg-info/",
                    "poetry.lock",
                    "Pipfile.lock",
                ],
                "javascript": [
                    "node_modules/",
                    "package-lock.json",
                    "yarn.lock",
                    "pnpm-lock.yaml",
                ],
                "go": ["vendor/", "go.sum"],
                "rust": ["target/", "Cargo.lock"],
                "java": ["target/", "*.jar", "*.war"],
                "c_cpp": ["build/", "*.o", "*.so", "*.a"],
            },
            "compiled": ["*.pyc", "*.pyo", "*.so", "*.dll", "*.class", "*.o"],
            "media": ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.mp4", "*.mp3"],
            "custom": [],
        }

        # Add custom exclusions
        custom_excludes = input(
            f"{Colors.CYAN}Additional folders to exclude? (comma separated): {Colors.ENDC}"
        ).strip()
        if custom_excludes:
            self.config["ignore"]["custom"] = [x.strip() for x in custom_excludes.split(",")]
            logger.info(f"Added custom exclusions: {self.config['ignore']['custom']}")

    def _configure_docker(self):
        """Configure Docker limits"""
        ram_gb = self.config["hardware"]["ram_gb"]
        cpu_count = self.config["hardware"]["cpu_threads"]
        gpu_info = self.config["hardware"].get("gpu")

        # Use 50% of RAM for Docker
        docker_ram = max(4, ram_gb // 2)
        # Use 50% of CPUs
        docker_cpus = max(2, cpu_count // 2)

        self.config["docker"] = {
            "memory_limit": f"{docker_ram}G",
            "cpu_limit": str(docker_cpus),
            "gpu_enabled": bool(gpu_info),
        }

        logger.info(f"Docker configuration: {self.config['docker']}")

    def save(self) -> bool:
        """Save configuration to project directory"""
        config_path = self.project_dir / "config.yaml"

        try:
            # Backup if exists
            if config_path.exists():
                backup_path = config_path.with_suffix(".yaml.backup")
                import shutil

                shutil.copy2(config_path, backup_path)
                print_warning(f"Backup created: {backup_path}")
                logger.info(f"Created backup of existing config file: {backup_path}")

            # Save configuration
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

            print_success(f"Configuration saved to: {config_path}")
            logger.info(f"Configuration saved to: {config_path}")
            return True

        except Exception as e:
            print_error(f"Error saving configuration: {e}")
            logger.error(f"Error saving config: {e}")
            return False

    def generate_docker_compose(self) -> bool:
        """Generate docker-compose.yml based on configuration"""
        try:
            logger.info("Generating docker-compose.yml")
            print_info("Generating Docker Compose configuration...")
            show_spinner("Creating service definitions", 1.0)

            generator = DockerGenerator(self.config, self.project_dir)
            compose = generator.generate_compose()

            if generator.save_compose(compose):
                print_success("Docker Compose file generated")

                # Also create global Dockerfile if not exists
                if generator.generate_global_dockerfile():
                    print_success("Global Dockerfile updated")
                else:
                    print_warning("Failed to update global Dockerfile")

                return True
            else:
                print_error("Failed to save docker-compose.yml")
                return False

        except Exception as e:
            print_error(f"Error generating docker-compose.yml: {e}")
            logger.error(f"Docker compose generation error: {e}", exc_info=True)
            return False

    def generate_modelfile(self) -> bool:
        """Generate Modelfile with custom configuration"""
        try:
            logger.info("Generating Modelfile")
            print_info("Creating Modelfile with consciousness features...")
            show_spinner("Configuring model parameters", 1.0)

            project_name = self.config["project"]["name"]
            user_name = PROJECT_NAME  # From environment
            description = f"ACOLYTE assistant for {project_name}"
            model_name = self.config["model"]["name"]
            configured_num_ctx = self.config["model"]["context_size"]

            # Create a personalized system prompt
            system_prompt = f"""You are ACOLYTE, an advanced AI Programming Assistant for {project_name}. You help with architecting, developing, and maintaining high-quality software, adhering to best practices across all programming languages and paradigms. Project: {project_name}. Your responses are clear, precise, and actionable. You understand not just the current state of code, but its entire evolutionary history through Git integration."""

            # Create Modelfile content
            modelfile_content = f"""FROM {model_name}

# System prompt with project context
SYSTEM \"\"\"
{system_prompt}
\"\"\"

# Model parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx {configured_num_ctx}
PARAMETER repeat_penalty 1.1
PARAMETER seed 42
"""

            # Write Modelfile to infra directory
            infra_dir = self.project_dir / "infra"
            infra_dir.mkdir(parents=True, exist_ok=True)

            modelfile_path = infra_dir / "Modelfile"
            with open(modelfile_path, "w") as f:
                f.write(modelfile_content)

            print_success("Modelfile created with project context")
            logger.info(f"Modelfile generated: {modelfile_path}")
            return True

        except Exception as e:
            print_error(f"Error creating Modelfile: {e}")
            logger.error(f"Modelfile generation error: {e}", exc_info=True)
            return False


def validate_environment() -> List[str]:
    """Validate environment before proceeding"""
    errors = []

    logger.info("Validating environment")

    # Check if we're in a project directory
    project_path = Path(PROJECT_PATH)
    if not any(
        (project_path / pattern).exists()
        for pattern in [
            ".git",
            "package.json",
            "pyproject.toml",
            "Cargo.toml",
            "go.mod",
        ]
    ):
        error_msg = "Doesn't look like a project directory (no .git or project file found)"
        errors.append(error_msg)
        logger.warning(error_msg)

    # Check write permissions in global directory
    try:
        test_file = Path(GLOBAL_DIR) / "test_write"
        test_file.touch()
        test_file.unlink()
        logger.info("Write permissions verified")
    except Exception as e:
        error_msg = f"No write permissions in {GLOBAL_DIR}"
        errors.append(error_msg)
        logger.error(f"{error_msg}: {e}")

    return errors


def main():
    """Main function"""
    try:
        # Show logo
        print(ACOLYTE_LOGO)
        animate_text(
            f"{Colors.CYAN}{Colors.BOLD}Initializing ACOLYTE Project...{Colors.ENDC}",
            1.0,
        )

        logger.info("Starting ACOLYTE Init")
        logger.info(f"Project ID: {PROJECT_ID}")
        logger.info(f"Project Path: {PROJECT_PATH}")
        logger.info(f"Global Dir: {GLOBAL_DIR}")

        # Validate environment
        errors = validate_environment()
        if errors:
            print_error("Validation errors:")
            for error in errors:
                print(f"  • {error}")
            sys.exit(1)

        # Generate configuration
        project_path = Path(PROJECT_PATH)
        project_dir = Path(GLOBAL_DIR) / "projects" / PROJECT_ID

        config_generator = AcolyteConfig(project_path, project_dir)
        config = config_generator.generate()

        # Show summary
        print(f"\n{Colors.BOLD}📋 CONFIGURATION SUMMARY:{Colors.ENDC}")
        print(f"  Project: {config['project']['name']}")
        print(f"  ID: {PROJECT_ID[:8]}...")
        print(f"  Model: {config['model']['name']}")
        context_display = "32k"
        print(f"  Context: {context_display} tokens")
        print(
            f"  Docker: {config['docker']['memory_limit']} RAM, {config['docker']['cpu_limit']} CPUs"
        )
        print(
            f"  Ports: Weaviate:{config['ports']['weaviate']}, "
            f"Ollama:{config['ports']['ollama']}, Backend:{config['ports']['backend']}"
        )

        # Confirm
        confirm = (
            input(f"\n{Colors.YELLOW}Save this configuration? [Y/n]: {Colors.ENDC}").strip().lower()
        )
        if confirm == "n":
            print_warning("Configuration cancelled")
            logger.info("Configuration cancelled by user")
            sys.exit(0)

        # Save configuration files
        print_header("🚀 Generating Configuration Files")

        success_count = 0
        total_files = 3

        # 1. Save config.yaml
        if config_generator.save():
            success_count += 1
            print_success("1/3: Configuration file created")
        else:
            print_error("Failed to create configuration file")

        # 2. Generate docker-compose.yml
        if config_generator.generate_docker_compose():
            success_count += 1
            print_success("2/3: Docker infrastructure created")
        else:
            print_error("Failed to create Docker infrastructure")

        # 3. Generate Modelfile
        if config_generator.generate_modelfile():
            success_count += 1
            print_success("3/3: Modelfile created")
        else:
            print_error("Failed to create Modelfile")

        # Final status
        if success_count == total_files:
            print(f"\n{Colors.GREEN}✅ Configuration completed successfully!{Colors.ENDC}")
            print(f"\n{Colors.BOLD}Configuration stored in:{Colors.ENDC}")
            print(f"  {project_dir}")
            print(f"\n{Colors.BOLD}Next step:{Colors.ENDC}")
            print(
                f"  {Colors.CYAN}acolyte install{Colors.ENDC} - Install ACOLYTE with this configuration"
            )
            logger.info("ACOLYTE Init completed successfully")
        else:
            print(
                f"\n{Colors.YELLOW}⚠️ Configuration partially completed ({success_count}/{total_files} files){Colors.ENDC}"
            )
            logger.warning(f"ACOLYTE Init partially completed: {success_count}/{total_files} files")

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Configuration cancelled by user{Colors.ENDC}")
        logger.info("Configuration cancelled by user (KeyboardInterrupt)")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        print_info(f"See log file for details: {log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
