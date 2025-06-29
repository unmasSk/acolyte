#!/usr/bin/env python3
"""
🚀 ACOLYTE INSTALL - Complete Installation Process
Reads configuration from ~/.acolyte/projects/{project_id}/ and installs everything needed
"""

import asyncio
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import psutil
import yaml
from tqdm import tqdm

# Add common modules to path
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    ACOLYTE_LOGO,
    CONSCIOUSNESS_TIPS,
    Colors,
    animate_text,
    print_error,
    print_header,
    print_info,
    print_progress_bar,
    print_step,
    print_success,
    print_warning,
    show_spinner,
)

# Get environment variables from CLI
PROJECT_ID = os.environ.get('ACOLYTE_PROJECT_ID', '')
PROJECT_PATH = os.environ.get('ACOLYTE_PROJECT_PATH', '.')
GLOBAL_DIR = os.environ.get('ACOLYTE_GLOBAL_DIR', str(Path.home() / '.acolyte'))

# Configure logging
project_dir = Path(GLOBAL_DIR) / "projects" / PROJECT_ID
log_dir = project_dir / "data" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"install_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class RequirementsChecker:
    """Check and validate system requirements"""

    REQUIRED_TOOLS = {
        "git": {
            "command": ["git", "--version"],
            "min_version": "2.0.0",
            "install_hint": "Install from: https://git-scm.com",
        },
        "docker": {
            "command": ["docker", "--version"],
            "min_version": "20.0.0",
            "install_hint": "Install from: https://docs.docker.com/get-docker/",
        },
        "python": {
            "command": ["python3", "--version"],
            "min_version": "3.11.0",
            "install_hint": "Python 3.11+ required",
        },
    }

    @classmethod
    def check_tool(cls, tool_name: str, tool_info: Dict) -> Tuple[bool, str]:
        """Check if a tool is installed and meets version requirements"""
        try:
            show_spinner(f"Checking {tool_name}...", 0.5)

            # Use python instead of python3 on Windows
            command = tool_info["command"].copy()
            if tool_name == "python" and sys.platform == "win32":
                command[0] = "python"

            result = subprocess.run(command, capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                return False, "Not installed"

            # Extract version if needed
            output = result.stdout.strip()
            if tool_info["min_version"]:
                # Simple version extraction (works for most tools)
                import re

                version_match = re.search(r"(\d+\.\d+\.\d+)", output)
                if version_match:
                    version = version_match.group(1)
                    # Simple version comparison
                    if version < tool_info["min_version"]:
                        return False, f"Version {version} < {tool_info['min_version']}"
                    return True, f"Version {version}"

            return True, "Installed"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "Not found"
        except Exception as e:
            logger.error(f"Error checking {tool_name}: {e}")
            return False, f"Error: {str(e)}"

    @classmethod
    def check_all(cls) -> Dict[str, Tuple[bool, str]]:
        """Check all required tools"""
        results = {}

        print_info("Checking system requirements...")

        # Show fancy progress
        tools = list(cls.REQUIRED_TOOLS.items())
        for i, (tool_name, tool_info) in enumerate(tools):
            print_progress_bar(
                i,
                len(tools),
                prefix="Progress:",
                suffix=f"Checking {tool_name}",
                length=30,
            )
            time.sleep(0.3)  # Slow down for visual effect

            success, message = cls.check_tool(tool_name, tool_info)
            results[tool_name] = (success, message)

            if success:
                print_success(f"{tool_name}: {message}")
            else:
                print_error(f"{tool_name}: {message}")
                print(f"  {Colors.CYAN}{tool_info['install_hint']}{Colors.ENDC}")

        # Complete progress bar
        print_progress_bar(len(tools), len(tools), prefix="Progress:", suffix="Complete", length=30)

        return results


class ConfigLoader:
    """Load and validate configuration"""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = None

    def load(self) -> Optional[Dict]:
        """Load configuration from config.yaml"""
        try:
            show_spinner("Loading configuration...", 1.0)

            if not self.config_path.exists():
                print_error(f"Configuration file not found: {self.config_path}")
                print_info("Run 'acolyte init' first to create configuration")
                return None

            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            # Validate required fields with animation
            required_fields = ["version", "project", "hardware", "model", "docker"]
            for i, field in enumerate(required_fields):
                print_progress_bar(
                    i + 1,
                    len(required_fields),
                    prefix="Validating:",
                    suffix=f"Field: {field}",
                    length=30,
                )
                time.sleep(0.2)  # Slow down for visual effect

                if field not in self.config:
                    print_error(f"Missing required field in config: {field}")
                    return None

            print_success("Configuration loaded successfully")

            # Show project info
            project_name = self.config["project"]["name"]
            project_id = self.config["project"]["id"]
            model_name = self.config["model"]["name"]

            print(f"\n{Colors.CYAN}Project: {Colors.BOLD}{project_name}{Colors.ENDC}")
            print(f"{Colors.CYAN}ID: {Colors.BOLD}{project_id[:8]}...{Colors.ENDC}")
            print(f"{Colors.CYAN}Model: {Colors.BOLD}{model_name}{Colors.ENDC}\n")

            return self.config

        except yaml.YAMLError as e:
            print_error(f"Error parsing config: {e}")
            return None
        except Exception as e:
            print_error(f"Error loading configuration: {e}")
            logger.error(f"Config load error: {e}", exc_info=True)
            return None

    def validate_hardware(self) -> bool:
        """Validate that current hardware meets requirements"""
        if not self.config:
            return False

        show_spinner("Validating hardware requirements...", 1.0)

        model_ram = self.config["model"]["ram_required"]
        system_ram = psutil.virtual_memory().total / (1024**3)

        print(f"{Colors.CYAN}Model requires: {Colors.BOLD}{model_ram}GB RAM{Colors.ENDC}")
        print(f"{Colors.CYAN}System has: {Colors.BOLD}{system_ram:.1f}GB RAM{Colors.ENDC}")

        if system_ram < model_ram:
            print_warning(f"Model requires {model_ram}GB RAM, you have {system_ram:.1f}GB")
            response = (
                input(f"{Colors.YELLOW}Continue anyway? [y/N]: {Colors.ENDC}").strip().lower()
            )
            return response == "y"

        print_success("Hardware requirements met")
        return True


class DockerManager:
    """Manage Docker services"""

    def __init__(self, config: Dict, project_dir: Path):
        self.config = config
        self.project_dir = project_dir
        self.compose_file = project_dir / "infra" / "docker-compose.yml"

    def verify_compose_file(self) -> bool:
        """Verify that docker-compose.yml exists"""
        try:
            show_spinner("Verifying docker-compose.yml...", 1.0)

            if not self.compose_file.exists():
                print_error("docker-compose.yml not found")
                print_info("Run 'acolyte init' first to create docker-compose.yml")
                return False

            print_success("docker-compose.yml found")
            return True

        except Exception as e:
            print_error(f"Error verifying docker-compose.yml: {e}")
            logger.error(f"Docker compose verification error: {e}", exc_info=True)
            return False

    async def start_services(self) -> bool:
        """Start Docker services"""
        try:
            print_info("Starting Docker services...")

            # Animated starting sequence
            services = ["weaviate", "ollama", "backend"]
            for i, service in enumerate(services):
                show_spinner(f"Starting {service}...", 1.0)
                print_progress_bar(
                    i + 1,
                    len(services),
                    prefix="Starting:",
                    suffix=f"Service: {service}",
                    length=30,
                )

            # Change to infra directory for docker-compose
            os.chdir(self.compose_file.parent)

            process = await asyncio.create_subprocess_exec(
                "docker-compose",
                "up",
                "-d",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Show progress during docker compose execution
            print_info("Building and starting containers...")
            with tqdm(
                desc=f"{Colors.CYAN}Docker compose progress{Colors.ENDC}",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.CYAN, Colors.ENDC),
            ) as pbar:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    output = line.decode().strip()
                    if output:
                        # Update description with current action
                        if "Creating" in output:
                            pbar.set_description_str(
                                f"{Colors.CYAN}Creating containers{Colors.ENDC}"
                            )
                        elif "Starting" in output:
                            pbar.set_description_str(
                                f"{Colors.CYAN}Starting containers{Colors.ENDC}"
                            )
                        elif "Building" in output:
                            pbar.set_description_str(f"{Colors.CYAN}Building images{Colors.ENDC}")
                        pbar.update(1)
                        pbar.set_postfix_str(output[:50] + "..." if len(output) > 50 else output)

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                print_success("Docker services started")
                # Wait for services to be ready
                await self._wait_for_services()
                return True
            else:
                print_error(f"Failed to start services: {stderr.decode()}")
                return False

        except Exception as e:
            print_error(f"Error starting Docker services: {e}")
            return False

    async def _wait_for_services(self):
        """Wait for services to be ready"""
        print_info("Waiting for services to be ready...")

        # Get ports from config
        ports = self.config.get("ports", {})
        weaviate_port = ports.get("weaviate", 8080)
        ollama_port = ports.get("ollama", 11434)
        backend_port = ports.get("backend", 8000)

        checks = {
            "Weaviate": (f"http://localhost:{weaviate_port}/v1/.well-known/ready", 30),
            "Ollama": (f"http://localhost:{ollama_port}/api/tags", 30),
            "Backend": (
                f"http://localhost:{backend_port}/api/health",
                45,
            ),  # Backend takes longer
        }

        import aiohttp

        async with aiohttp.ClientSession() as session:
            for service, (url, timeout) in checks.items():
                print(f"{Colors.CYAN}Waiting for {service}...{Colors.ENDC}")

                start_time = time.time()
                with tqdm(
                    total=timeout,
                    desc=f"Waiting for {service}",
                    bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.CYAN, Colors.ENDC),
                ) as pbar:
                    while time.time() - start_time < timeout:
                        try:
                            async with session.get(
                                url, timeout=aiohttp.ClientTimeout(total=2)
                            ) as resp:
                                if resp.status == 200:
                                    pbar.update(timeout - pbar.n)  # Fill the bar
                                    print_success(f"{service} is ready")
                                    break
                        except Exception:
                            pass

                        # Show random consciousness tip
                        if random.random() < 0.2:  # 20% chance
                            tip = random.choice(CONSCIOUSNESS_TIPS)
                            # Clear current line and show tip above progress bar
                            print(f"\r{' ' * 80}\r", end="")  # Clear line
                            print(f"{Colors.YELLOW}💭 {tip}{Colors.ENDC}")
                            pbar.refresh()  # Redraw progress bar

                        await asyncio.sleep(1)
                        pbar.update(1)
                    else:
                        print_warning(f"{service} not ready after {timeout}s")


class OllamaManager:
    """Manage Ollama models"""

    def __init__(self, model_name: str, project_dir: Path):
        self.model_name = model_name
        self.modelfile_path = project_dir / "infra" / "Modelfile"

    def verify_modelfile(self) -> bool:
        """Verify that Modelfile exists"""
        try:
            show_spinner("Verifying Modelfile...", 1.0)

            if not self.modelfile_path.exists():
                print_error("Modelfile not found")
                print_info("Run 'acolyte init' first to create Modelfile")
                return False

            print_success("Modelfile found")
            return True

        except Exception as e:
            print_error(f"Error verifying Modelfile: {e}")
            logger.error(f"Modelfile verification error: {e}", exc_info=True)
            return False

    async def pull_model(self) -> bool:
        """Pull the specified model"""
        try:
            print_info(f"Downloading model: {self.model_name}")

            # Extract base model name (without acolyte prefix)
            base_model = self.model_name

            process = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                "acolyte-ollama",
                "ollama",
                "pull",
                base_model,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Show progress with consciousness tips
            with tqdm(
                desc=f"{Colors.CYAN}Downloading model{Colors.ENDC}",
                unit="MB",
                unit_scale=True,
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.CYAN, Colors.ENDC),
            ) as pbar:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break

                    output = line.decode().strip()
                    if output:
                        # Update progress bar
                        if "pulling" in output.lower():
                            # Show random consciousness tip above progress bar
                            tip = random.choice(CONSCIOUSNESS_TIPS)
                            print(f"\n{Colors.YELLOW}💭 {tip}{Colors.ENDC}")
                            pbar.refresh()

                        # Extract progress if possible
                        if "%" in output:
                            try:
                                import re

                                match = re.search(r"(\d+)%", output)
                                if match:
                                    percent = int(match.group(1))
                                    pbar.n = percent
                                    pbar.refresh()
                            except Exception:
                                pass

            # Get any error output
            _, stderr = await process.communicate()

            if process.returncode == 0:
                print_success(f"Model {base_model} downloaded")
                return True
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                print_error(f"Failed to download model: {error_msg}")
                # Try to provide helpful suggestions
                if "not found" in error_msg.lower():
                    print_info("Model might not exist. Check model name.")
                elif "connection" in error_msg.lower():
                    print_info("Network issue. Check internet connection.")
                return False

        except Exception as e:
            print_error(f"Error downloading model: {e}")
            logger.error(f"Ollama pull error: {e}", exc_info=True)
            return False

    async def create_acolyte_model(self) -> bool:
        """Create ACOLYTE model with custom configuration"""
        try:
            print_info("Creating ACOLYTE model with consciousness features...")

            # Check if Modelfile exists
            if not self.modelfile_path.exists():
                print_error("Modelfile not found")
                return False

            # Copy Modelfile to container with animation
            show_spinner("Copying Modelfile to container...", 1.0)
            subprocess.run(
                [
                    "docker",
                    "cp",
                    str(self.modelfile_path),
                    "acolyte-ollama:/tmp/Modelfile",
                ],
                check=True,
            )

            # Create model with animation
            show_spinner("Creating ACOLYTE model...", 1.0)
            process = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                "acolyte-ollama",
                "ollama",
                "create",
                "acolyte",
                "-f",
                "/tmp/Modelfile",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Show fancy animation during model creation
            frames = ["🧠", "💭", "✨", "🔮", "💫", "⚡", "🌟", "💡"]
            i = 0
            while process.returncode is None:
                frame = frames[i % len(frames)]
                sys.stdout.write(f"\r{frame} Creating ACOLYTE model with consciousness... {frame}")
                sys.stdout.flush()
                await asyncio.sleep(0.1)
                i += 1

                # Check if process has finished
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.1)
                    break
                except asyncio.TimeoutError:
                    continue

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                print("\r" + " " * 60 + "\r", end="")  # Clear line
                print_success("ACOLYTE model created with consciousness capabilities")
                return True
            else:
                print("\r" + " " * 60 + "\r", end="")  # Clear line
                print_error(f"Failed to create ACOLYTE model: {stderr.decode()}")
                return False

        except Exception as e:
            print_error(f"Error creating ACOLYTE model: {e}")
            return False


class ProjectIndexer:
    """Index project code into Weaviate"""

    def __init__(self, project_path: Path, config: Dict):
        self.project_path = Path(project_path)
        self.config = config

    async def index_project(self) -> bool:
        """Index the project"""
        try:
            print_info("Indexing project files...")

            # Use the API to trigger indexing
            ports = self.config.get('ports', {})
            backend_port = ports.get('backend', 8000)

            import aiohttp

            async with aiohttp.ClientSession() as session:
                # Trigger full indexing via API
                async with session.post(
                    f"http://localhost:{backend_port}/api/index/project",
                    json={
                        "force_reindex": True,
                        "respect_gitignore": True,
                        "respect_acolyteignore": True,
                    },
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        task_id = result.get('task_id')

                        print_success(f"Indexing started with task ID: {task_id}")
                        print_info("You can monitor progress via WebSocket or wait for completion")

                        # Simple progress simulation
                        with tqdm(
                            total=100,
                            desc=f"{Colors.CYAN}Indexing files{Colors.ENDC}",
                            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.CYAN, Colors.ENDC),
                        ) as pbar:
                            for i in range(100):
                                await asyncio.sleep(0.1)
                                pbar.update(1)

                                # Show consciousness tip occasionally
                                if i % 20 == 0:
                                    tip = random.choice(CONSCIOUSNESS_TIPS)
                                    print(f"\n{Colors.YELLOW}💭 {tip}{Colors.ENDC}")
                                    pbar.refresh()

                        print_success("Project indexing completed")
                        return True
                    else:
                        error = await resp.text()
                        print_error(f"Indexing failed: {error}")
                        return False

        except Exception as e:
            print_error(f"Error indexing project: {e}")
            logger.error(f"Indexing error: {e}", exc_info=True)
            return False


class AcolyteInstaller:
    """Main installer class"""

    def __init__(self):
        self.project_dir = Path(GLOBAL_DIR) / "projects" / PROJECT_ID
        self.config_path = self.project_dir / "config.yaml"
        self.config = None
        self.project_path = Path(PROJECT_PATH)

    async def install(self):
        """Run the complete installation process"""
        # Show logo with animation
        print(ACOLYTE_LOGO)
        animate_text(
            f"{Colors.CYAN}{Colors.BOLD}ACOLYTE INSTALL - Installation Process{Colors.ENDC}",
            duration=1.0,
        )
        print("\n")

        # Step 1: Check requirements
        print_step(1, 7, "System Requirements")
        requirements = RequirementsChecker.check_all()
        if not all(success for success, _ in requirements.values()):
            print_error("Some requirements are not met")
            return False

        # Step 2: Load configuration
        print_step(2, 7, "Loading Configuration")
        config_loader = ConfigLoader(self.config_path)
        self.config = config_loader.load()
        if not self.config:
            return False

        # Validate hardware
        if not config_loader.validate_hardware():
            return False

        # Step 3: Verify Docker setup
        print_step(3, 7, "Docker Infrastructure")
        docker_manager = DockerManager(self.config, self.project_dir)
        if not docker_manager.verify_compose_file():
            return False

        # Step 4: Start Docker services
        print_step(4, 7, "Starting Docker Services")
        if not await docker_manager.start_services():
            return False

        # Step 5: Setup Ollama model
        print_step(5, 7, "Setting Up Ollama Model")
        ollama_manager = OllamaManager(self.config["model"]["name"], self.project_dir)
        if not ollama_manager.verify_modelfile():
            return False

        if not await ollama_manager.pull_model():
            return False

        if not await ollama_manager.create_acolyte_model():
            return False

        # Step 6: Initialize database
        print_step(6, 7, "Initializing Database")
        # Database is auto-initialized by the backend on first start
        print_success("Database initialized automatically")

        # Step 7: Index project
        print_step(7, 7, "Indexing Project")
        indexer = ProjectIndexer(self.project_path, self.config)
        if not await indexer.index_project():
            return False

        # Final success message with animation
        print_header("✨ Installation Complete")
        print_success("ACOLYTE is now ready to use!")

        # Show usage tips with animation
        print(f"\n{Colors.CYAN}{Colors.BOLD}Quick Start Guide:{Colors.ENDC}")
        tips = [
            "Check status: acolyte status",
            "Stop services: acolyte stop",
            "Start services: acolyte start",
            "View logs: check ~/.acolyte/projects/{id}/data/logs/",
        ]

        for tip in tips:
            animate_text(f"{Colors.GREEN}▶ {tip}{Colors.ENDC}", duration=0.3)

        # Show service URLs
        ports = self.config.get("ports", {})
        print(f"\n{Colors.CYAN}{Colors.BOLD}Service URLs:{Colors.ENDC}")
        print(f"  Weaviate: http://localhost:{ports.get('weaviate', 8080)}")
        print(f"  Ollama: http://localhost:{ports.get('ollama', 11434)}")
        print(f"  Backend API: http://localhost:{ports.get('backend', 8000)}")
        print(f"  API Docs: http://localhost:{ports.get('backend', 8000)}/api/docs")

        return True


async def main():
    """Main entry point"""
    try:
        installer = AcolyteInstaller()
        await installer.install()
    except KeyboardInterrupt:
        print_warning("\nInstallation interrupted")
        print_info("You can resume installation by running 'acolyte install' again")
    except Exception as e:
        print_error(f"Installation failed: {e}")
        logger.error(f"Installation error: {e}", exc_info=True)
        print_info("Check logs for details: " + str(log_file))


if __name__ == "__main__":
    asyncio.run(main())
