#!/usr/bin/env python3
"""
ACOLYTE CLI - Command Line Interface
Global tool for managing ACOLYTE in user projects
"""

import hashlib
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import click
import yaml
import requests
import subprocess

# Handle imports differently when installed via pip vs development
try:
    # When installed via pip, imports work directly
    from acolyte.core.logging import logger
    from acolyte.core.exceptions import AcolyteError
    # Determine if we're running from installed package
    PACKAGE_DIR = Path(__file__).parent.parent  # acolyte package root
    if (PACKAGE_DIR / 'scripts').exists():
        # Running from source
        PROJECT_ROOT = PACKAGE_DIR.parent
    else:
        # Installed via pip - use package location
        PROJECT_ROOT = PACKAGE_DIR
except ImportError:
    # Development mode - add project root to path
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    from acolyte.core.logging import logger
    from acolyte.core.exceptions import AcolyteError


class ProjectManager:
    """Manages ACOLYTE projects and their configurations"""

    def __init__(self):
        self.global_dir = self._get_global_dir()
        self.projects_dir = self.global_dir / "projects"
        
        # Initialize global directory structure if needed
        self._ensure_global_structure()
    
    def _ensure_global_structure(self):
        """Ensure ACOLYTE global directory structure exists"""
        # Create directories
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(exist_ok=True)
        
        # Create other necessary directories
        (self.global_dir / "models").mkdir(exist_ok=True)
        (self.global_dir / "logs").mkdir(exist_ok=True)
        
        # Copy essential files if this is first run
        if not (self.global_dir / ".initialized").exists():
            self._first_run_setup()
    
    def _first_run_setup(self):
        """Setup ACOLYTE on first run after pip install"""
        logger.info("First run detected, initializing ACOLYTE...")
        
        # Copy Docker templates if available
        docker_template = PROJECT_ROOT / "scripts" / "install" / "common" / "docker-compose.template.yml"
        if docker_template.exists():
            templates_dir = self.global_dir / "templates"
            templates_dir.mkdir(exist_ok=True)
            shutil.copy2(docker_template, templates_dir / "docker-compose.template.yml")
        
        # Copy example configurations
        examples_dir = PROJECT_ROOT / "examples"
        if examples_dir.exists():
            dest_examples = self.global_dir / "examples"
            if dest_examples.exists():
                shutil.rmtree(dest_examples)
            shutil.copytree(examples_dir, dest_examples)
        
        # Mark as initialized
        (self.global_dir / ".initialized").touch()
        logger.info(f"ACOLYTE initialized at {self.global_dir}")

    def _get_global_dir(self) -> Path:
        """Get the global ACOLYTE directory"""
        if os.name == 'nt':  # Windows
            return Path.home() / ".acolyte"
        else:  # Linux/Mac
            # Check if running from development or installed
            if 'ACOLYTE_DEV' in os.environ:
                return Path.home() / ".acolyte-dev"
            return Path.home() / ".acolyte"

    def get_project_id(self, project_path: Path) -> str:
        """Generate unique project ID from path and git remote"""
        # Try to get git remote
        git_remote = ""
        git_dir = project_path / ".git"
        if git_dir.exists():
            try:
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    git_remote = result.stdout.strip()
            except Exception:
                pass

        # Generate hash from absolute path + git remote
        abs_path = str(project_path.resolve())
        unique_string = f"{git_remote}:{abs_path}"
        project_id = hashlib.sha256(unique_string.encode()).hexdigest()[:12]

        return project_id

    def get_project_dir(self, project_id: str) -> Path:
        """Get the directory for a specific project"""
        return self.projects_dir / project_id

    def is_project_initialized(self, project_path: Path) -> bool:
        """Check if project is already initialized"""
        project_file = project_path / ".acolyte.project"
        return project_file.exists()

    def load_project_info(self, project_path: Path) -> Optional[Dict[str, Any]]:
        """Load project info from .acolyte.project"""
        project_file = project_path / ".acolyte.project"
        if not project_file.exists():
            return None

        try:
            with open(project_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load project info: {e}")
            return None

    def save_project_info(self, project_path: Path, info: Dict[str, Any]) -> bool:
        """Save project info to .acolyte.project"""
        project_file = project_path / ".acolyte.project"
        try:
            with open(project_file, 'w') as f:
                yaml.dump(info, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save project info: {e}")
            return False


def validate_project_directory(ctx, param, value):
    """Validate that we're in a valid project directory"""
    project_path = Path(value or ".")

    # Check if it's a git repository or has project files
    markers = [
        ".git",
        "package.json",
        "pyproject.toml",
        "setup.py",  # Python
        "Cargo.toml",  # Rust
        "go.mod",  # Go
        "pom.xml",
        "build.gradle",  # Java
        "composer.json",  # PHP
        "Gemfile",  # Ruby
    ]

    has_marker = any((project_path / marker).exists() for marker in markers)

    if not has_marker:
        raise click.BadParameter(
            "Not a valid project directory. Please run from a project with version control or project file."
        )

    return project_path


@click.group()
@click.version_option(version="1.0.0", prog_name="ACOLYTE")
def cli():
    """
    ACOLYTE - AI Programming Assistant

    Your local AI assistant with infinite memory for code projects.
    """
    pass


@cli.command()
@click.option(
    '--path',
    default=".",
    callback=validate_project_directory,
    help='Project path (default: current directory)',
)
@click.option('--name', help='Project name (default: directory name)')
@click.option('--force', is_flag=True, help='Force re-initialization')
def init(path: str, name: Optional[str], force: bool):
    """Initialize ACOLYTE in the current project"""
    project_path = Path(path)
    manager = ProjectManager()

    # Check if already initialized
    if manager.is_project_initialized(project_path) and not force:
        click.echo(click.style("✗ Project already initialized!", fg="red"))
        click.echo("Use --force to re-initialize")
        return

    click.echo(click.style("🤖 ACOLYTE Project Initialization", fg="cyan", bold=True))
    click.echo(f"Project path: {project_path.resolve()}")

    # Generate project ID
    project_id = manager.get_project_id(project_path)
    click.echo(f"Project ID: {project_id}")

    # Get project name
    if not name:
        name = click.prompt("Project name", default=project_path.name)

    # Run the initialization script
    init_script = PROJECT_ROOT / "scripts" / "install" / "init.py"
    if not init_script.exists():
        click.echo(click.style("✗ Initialization script not found!", fg="red"))
        return

    # Set environment variables for the init script
    os.environ['ACOLYTE_PROJECT_ID'] = project_id
    os.environ['ACOLYTE_PROJECT_PATH'] = str(project_path.resolve())
    os.environ['ACOLYTE_GLOBAL_DIR'] = str(manager.global_dir)
    os.environ['ACOLYTE_PROJECT_NAME'] = name or ""

    # Run init script
    result = subprocess.run([sys.executable, str(init_script)], cwd=project_path)

    if result.returncode == 0:
        # Save project info
        project_info = {
            'project_id': project_id,
            'name': name,
            'path': str(project_path.resolve()),
            'initialized': datetime.now().isoformat(),
            'acolyte_version': '1.0.0',
        }

        if manager.save_project_info(project_path, project_info):
            click.echo(click.style("✓ Project initialized successfully!", fg="green"))
            click.echo(f"Configuration stored in: {manager.get_project_dir(project_id)}")
        else:
            click.echo(click.style("✗ Failed to save project info!", fg="red"))
    else:
        click.echo(click.style("✗ Initialization failed!", fg="red"))


@cli.command()
@click.option('--path', default=".", help='Project path')
def install(path: str):
    """Install ACOLYTE services for the project"""
    project_path = Path(path)
    manager = ProjectManager()

    # Check if initialized
    project_info = manager.load_project_info(project_path)
    if not project_info:
        click.echo(click.style("✗ Project not initialized!", fg="red"))
        click.echo("Run 'acolyte init' first")
        return

    project_id = project_info['project_id']
    click.echo(click.style("🚀 ACOLYTE Installation", fg="cyan", bold=True))
    click.echo(f"Project: {project_info['name']} ({project_id})")

    # Run installation script
    install_script = PROJECT_ROOT / "scripts" / "install" / "install.py"
    if not install_script.exists():
        click.echo(click.style("✗ Installation script not found!", fg="red"))
        return

    # Set environment variables
    os.environ['ACOLYTE_PROJECT_ID'] = project_id
    os.environ['ACOLYTE_PROJECT_PATH'] = str(project_path.resolve())
    os.environ['ACOLYTE_GLOBAL_DIR'] = str(manager.global_dir)

    # Run install script
    result = subprocess.run([sys.executable, str(install_script)], cwd=project_path)

    if result.returncode == 0:
        click.echo(click.style("✓ Installation completed!", fg="green"))
    else:
        click.echo(click.style("✗ Installation failed!", fg="red"))


def detect_docker_compose_cmd() -> list[str]:
    """Detecta si está disponible 'docker compose' (v2) o 'docker-compose' (v1) y retorna el comando adecuado como lista."""
    try:
        # Probar 'docker compose version'
        result = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            return ["docker", "compose"]
    except Exception:
        pass
    try:
        # Probar 'docker-compose version'
        result = subprocess.run(["docker-compose", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            return ["docker-compose"]
    except Exception:
        pass
    raise RuntimeError("Neither 'docker compose' nor 'docker-compose' is available on this system.")


@cli.command()
@click.option('--path', default=".", help='Project path')
def start(path: str):
    """Start ACOLYTE services"""
    project_path = Path(path)
    manager = ProjectManager()

    # Check if initialized
    project_info = manager.load_project_info(project_path)
    if not project_info:
        click.echo(click.style("✗ Project not initialized!", fg="red"))
        return

    project_id = project_info['project_id']
    project_dir = manager.get_project_dir(project_id)
    docker_compose = project_dir / "infra" / "docker-compose.yml"

    if not docker_compose.exists():
        click.echo(click.style("✗ Services not installed!", fg="red"))
        click.echo("Run 'acolyte install' first")
        return

    click.echo(click.style("🚀 Starting ACOLYTE services...", fg="cyan"))

    compose_cmd = detect_docker_compose_cmd()
    result = subprocess.run(
        compose_cmd + ["-f", str(docker_compose), "up", "-d"], cwd=project_dir / "infra"
    )

    if result.returncode == 0:
        click.echo(click.style("✓ Services started!", fg="green"))

        # Load config to show URLs
        config_file = project_dir / ".acolyte"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                ports = config.get('ports', {})

            click.echo("\nService URLs:")
            click.echo(f"  Weaviate: http://localhost:{ports.get('weaviate', 8080)}")
            click.echo(f"  Ollama: http://localhost:{ports.get('ollama', 11434)}")
            click.echo(f"  API: http://localhost:{ports.get('backend', 8000)}")
    else:
        click.echo(click.style("✗ Failed to start services!", fg="red"))


@cli.command()
@click.option('--path', default=".", help='Project path')
def stop(path: str):
    """Stop ACOLYTE services"""
    project_path = Path(path)
    manager = ProjectManager()

    # Check if initialized
    project_info = manager.load_project_info(project_path)
    if not project_info:
        click.echo(click.style("✗ Project not initialized!", fg="red"))
        return

    project_id = project_info['project_id']
    project_dir = manager.get_project_dir(project_id)
    docker_compose = project_dir / "infra" / "docker-compose.yml"

    if not docker_compose.exists():
        click.echo(click.style("✗ Services not installed!", fg="red"))
        return

    click.echo(click.style("🛑 Stopping ACOLYTE services...", fg="cyan"))

    compose_cmd = detect_docker_compose_cmd()
    result = subprocess.run(
        compose_cmd + ["-f", str(docker_compose), "down"], cwd=project_dir / "infra"
    )

    if result.returncode == 0:
        click.echo(click.style("✓ Services stopped!", fg="green"))
    else:
        click.echo(click.style("✗ Failed to stop services!", fg="red"))


@cli.command()
@click.option('--path', default=".", help='Project path')
def status(path: str):
    """Check ACOLYTE status for the project"""
    project_path = Path(path)
    manager = ProjectManager()

    # Check if initialized
    project_info = manager.load_project_info(project_path)
    if not project_info:
        click.echo(click.style("✗ Project not initialized", fg="red"))
        return

    project_id = project_info['project_id']
    project_dir = manager.get_project_dir(project_id)

    click.echo(click.style("📊 ACOLYTE Status", fg="cyan", bold=True))
    click.echo(f"Project: {project_info['name']}")
    click.echo(f"ID: {project_id}")
    click.echo(f"Initialized: {project_info['initialized']}")
    click.echo(f"Storage: {project_dir}")

    # Check services
    docker_compose = project_dir / "infra" / "docker-compose.yml"
    if docker_compose.exists():
        click.echo("\nServices:")

        compose_cmd = detect_docker_compose_cmd()
        result = subprocess.run(
            compose_cmd + ["-f", str(docker_compose), "ps"],
            cwd=project_dir / "infra",
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            output = result.stdout
            if "acolyte-weaviate" in output and "Up" in output:
                click.echo(click.style("  ✓ Weaviate: Running", fg="green"))
            else:
                click.echo(click.style("  ✗ Weaviate: Stopped", fg="red"))

            if "acolyte-ollama" in output and "Up" in output:
                click.echo(click.style("  ✓ Ollama: Running", fg="green"))
            else:
                click.echo(click.style("  ✗ Ollama: Stopped", fg="red"))

            if "acolyte-backend" in output and "Up" in output:
                click.echo(click.style("  ✓ Backend: Running", fg="green"))
            else:
                click.echo(click.style("  ✗ Backend: Stopped", fg="red"))
        else:
            click.echo(click.style("  ✗ Docker not available", fg="red"))
    else:
        click.echo(click.style("\n✗ Services not installed", fg="red"))

    # Check data
    data_dir = project_dir / "data"
    if data_dir.exists():
        db_file = data_dir / "acolyte.db"
        if db_file.exists():
            size_mb = db_file.stat().st_size / (1024 * 1024)
            click.echo(f"\nDatabase: {size_mb:.1f} MB")


@cli.command()
@click.option('--path', default=".", help='Project path')
@click.option('--full', is_flag=True, help='Full project indexing')
def index(path: str, full: bool):
    """Index project files"""
    project_path = Path(path)
    manager = ProjectManager()

    # Check if initialized
    project_info = manager.load_project_info(project_path)
    if not project_info:
        click.echo(click.style("✗ Project not initialized!", fg="red"))
        return

    click.echo(click.style("🔍 Indexing project files...", fg="cyan"))

    # Call indexing service via API
    project_id = project_info['project_id']
    project_dir = manager.get_project_dir(project_id)
    config_file = project_dir / ".acolyte"

    if not config_file.exists():
        click.echo(click.style("✗ Configuration not found!", fg="red"))
        click.echo("Run 'acolyte install' first")
        return

    # Load config to get backend port
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        backend_port = config.get('ports', {}).get('backend', 8000)

    # Trigger indexing via API

    try:
        response = requests.post(
            f"http://localhost:{backend_port}/api/index/project",
            json={"force_reindex": full, "respect_gitignore": True, "respect_acolyteignore": True},
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            task_id = result.get('task_id')
            estimated_files = result.get('estimated_files', 0)

            click.echo(click.style("✓ Indexing started!", fg="green"))
            click.echo(f"Task ID: {task_id}")
            click.echo(f"Estimated files: {estimated_files}")
            click.echo(
                f"\nMonitor progress at: ws://localhost:{backend_port}{result.get('websocket_url')}"
            )
        else:
            click.echo(click.style(f"✗ Indexing failed: {response.text}", fg="red"))
    except requests.exceptions.ConnectionError:
        click.echo(click.style("✗ Backend not running!", fg="red"))
        click.echo("Start services with 'acolyte start'")
    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg="red"))


@cli.command()
def projects():
    """List all ACOLYTE projects"""
    manager = ProjectManager()

    click.echo(click.style("📂 ACOLYTE Projects", fg="cyan", bold=True))

    projects = []
    for project_dir in manager.projects_dir.iterdir():
        if project_dir.is_dir():
            config_file = project_dir / ".acolyte"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                        projects.append(
                            {
                                'id': project_dir.name,
                                'name': config.get('project', {}).get('name', 'Unknown'),
                                'path': config.get('project', {}).get('path', 'Unknown'),
                            }
                        )
                except Exception:
                    pass

    if projects:
        for proj in projects:
            click.echo(f"\n• {proj['name']} ({proj['id'][:8]}...)")
            click.echo(f"  Path: {proj['path']}")
    else:
        click.echo("No projects found")


@cli.command()
@click.option('--path', default=".", help='Project path')
def clean(path: str):
    """Clean ACOLYTE cache and temporary files"""
    project_path = Path(path)
    manager = ProjectManager()

    # Check if initialized
    project_info = manager.load_project_info(project_path)
    if not project_info:
        click.echo(click.style("✗ Project not initialized!", fg="red"))
        return

    if click.confirm("This will clean all cache and logs. Continue?"):
        project_id = project_info['project_id']
        project_dir = manager.get_project_dir(project_id)

        # Clean logs
        logs_dir = project_dir / "data" / "logs"
        if logs_dir.exists():
            import shutil

            shutil.rmtree(logs_dir)
            logs_dir.mkdir(parents=True)
            click.echo(click.style("✓ Logs cleaned", fg="green"))

        # TODO: Clean other cache
        click.echo(click.style("✓ Cleanup completed!", fg="green"))


@cli.command()
@click.option('--path', default=".", help='Project path')
@click.option('-f', '--follow', is_flag=True, help='Follow log output (like tail -f)')
@click.option('-n', '--lines', default=100, help='Number of lines to show (default: 100)')
@click.option('-s', '--service', type=click.Choice(['backend', 'weaviate', 'ollama', 'all']), default='all', help='Service to show logs for')
@click.option('--file', is_flag=True, help='Show debug.log file instead of Docker logs')
@click.option('-g', '--grep', help='Filter logs containing text')
@click.option('--level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), help='Filter by log level (only for --file)')
def logs(path: str, follow: bool, lines: int, service: str, file: bool, grep: Optional[str], level: Optional[str]):
    """View ACOLYTE service logs"""
    project_path = Path(path)
    manager = ProjectManager()

    # Check if initialized
    project_info = manager.load_project_info(project_path)
    if not project_info:
        click.echo(click.style("✗ Project not initialized!", fg="red"))
        return

    project_id = project_info['project_id']
    project_dir = manager.get_project_dir(project_id)

    if file:
        # Show debug.log file
        debug_log = project_dir / "data" / "debug.log"
        if not debug_log.exists():
            click.echo(click.style("✗ No debug.log file found!", fg="red"))
            click.echo("Services may not have been started yet.")
            return

        click.echo(click.style("📄 Showing debug.log:", fg="cyan", bold=True))
        click.echo(f"File: {debug_log}")
        click.echo()

        # Build command based on OS
        if os.name == 'nt':  # Windows
            if follow:
                # Use PowerShell Get-Content -Wait for Windows
                cmd = ['powershell', '-Command', f'Get-Content "{debug_log}" -Tail {lines} -Wait']
            else:
                # Use PowerShell Get-Content for Windows
                cmd = ['powershell', '-Command', f'Get-Content "{debug_log}" -Tail {lines}']
                
            # Add grep filter if specified
            if grep:
                cmd[-1] += f' | Select-String "{grep}"'
            
            # Add level filter if specified
            if level:
                cmd[-1] += f' | Select-String "\\| {level} \\|"'
        else:  # Unix-like
            if follow:
                cmd = ['tail', '-f', f'-n{lines}', str(debug_log)]
            else:
                cmd = ['tail', f'-n{lines}', str(debug_log)]
            
            # Add filters using grep
            if grep or level:
                cmd.extend(['|', 'grep'])
                if level:
                    cmd.extend(['-E', f'\\| {level} \\|'])
                if grep:
                    if level:
                        cmd.extend(['|', 'grep', grep])
                    else:
                        cmd.append(grep)

        # Execute command
        try:
            if os.name == 'nt':
                # Windows: run PowerShell directly
                subprocess.run(cmd)
            else:
                # Unix: use shell=True for pipe
                subprocess.run(' '.join(cmd), shell=True)
        except KeyboardInterrupt:
            click.echo("\n" + click.style("✓ Log viewing stopped", fg="green"))

    else:
        # Show Docker logs
        docker_compose = project_dir / "infra" / "docker-compose.yml"
        if not docker_compose.exists():
            click.echo(click.style("✗ Services not installed!", fg="red"))
            return

        # Determine which services to show
        services_to_show = []
        if service == 'all':
            services_to_show = ['backend', 'weaviate', 'ollama']
        else:
            services_to_show = [service]

        compose_cmd = detect_docker_compose_cmd()

        # Build docker logs command
        for svc in services_to_show:
            container_name = f"acolyte-{svc}"
            
            click.echo(click.style(f"\n📋 Logs for {svc.upper()}:", fg="cyan", bold=True))
            click.echo(f"Container: {container_name}")
            
            # Build command
            cmd = compose_cmd + ['logs']
            
            if follow and len(services_to_show) == 1:
                # Only follow if showing single service
                cmd.append('-f')
            
            cmd.extend(['--tail', str(lines)])
            cmd.append(container_name)
            
            # Execute command
            try:
                result = subprocess.run(
                    cmd,
                    cwd=project_dir / "infra",
                    text=True,
                    capture_output=not follow  # Don't capture if following
                )
                
                if not follow and result.returncode == 0:
                    output = result.stdout
                    
                    # Apply grep filter if specified
                    if grep:
                        filtered_lines = []
                        for line in output.splitlines():
                            if grep.lower() in line.lower():
                                filtered_lines.append(line)
                        output = '\n'.join(filtered_lines)
                    
                    if output.strip():
                        click.echo(output)
                    else:
                        click.echo(click.style("  (no logs matching criteria)", fg="yellow"))
                elif result.returncode != 0 and not follow:
                    click.echo(click.style(f"  ✗ Failed to get logs: {result.stderr}", fg="red"))
                    
            except KeyboardInterrupt:
                click.echo("\n" + click.style("✓ Log viewing stopped", fg="green"))
                break
            except Exception as e:
                click.echo(click.style(f"  ✗ Error: {e}", fg="red"))

        if follow and len(services_to_show) > 1:
            click.echo(click.style("\n⚠️ Follow mode only works with single service. Use -s to specify.", fg="yellow"))


def main():
    """Main entry point"""
    try:
        cli()
    except AcolyteError as e:
        click.echo(click.style(f"✗ {e.message}", fg="red"))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"✗ Unexpected error: {e}", fg="red"))
        if os.environ.get('ACOLYTE_DEBUG'):
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
