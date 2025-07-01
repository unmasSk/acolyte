#!/usr/bin/env python3
"""
Docker and GPU detection utilities for ACOLYTE
"""

from pathlib import Path
from typing import Dict, List, cast, Any

import yaml
from acolyte.core.logging import logger


class GPUDetector:
    """Detects GPU libraries for Docker volume mounting"""

    @staticmethod
    def find_nvidia_libraries() -> Dict[str, List[str]]:
        """Find NVIDIA libraries and devices"""
        libraries: Dict[str, List[str]] = {"volumes": [], "devices": []}

        # Common NVIDIA library paths
        lib_paths = [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/local/cuda/lib64",
            "/usr/lib",
            "/lib/x86_64-linux-gnu",
        ]

        # Libraries to find
        required_libs = ["libcuda.so", "libnvidia-ml.so"]

        for lib_path in lib_paths:
            lib_dir = Path(lib_path)
            if not lib_dir.exists():
                continue

            for lib_name in required_libs:
                # Find all versions of the library
                lib_files = list(lib_dir.glob(f"{lib_name}*"))
                if lib_files:
                    # Get the actual library file (follow symlinks)
                    for lib_file in lib_files:
                        if lib_file.is_file():
                            actual_file = lib_file.resolve()
                            # Mount both the actual file and symlinks
                            libraries["volumes"].append(f"{actual_file}:{lib_file}")
                            if lib_file != actual_file:
                                libraries["volumes"].append(f"{actual_file}:{actual_file}")
                            break

        # NVIDIA devices
        nvidia_devices = [
            "/dev/nvidia0",
            "/dev/nvidiactl",
            "/dev/nvidia-modeset",
            "/dev/nvidia-uvm",
        ]

        for device in nvidia_devices:
            if Path(device).exists():
                libraries["devices"].append(f"{device}:{device}")

        return libraries


class DockerGenerator:
    """Generates Docker Compose configuration for project infrastructure"""

    def __init__(self, config: Dict, project_dir: Path):
        self.config = config
        self.project_dir = project_dir  # This is ~/.acolyte/projects/{project_id}/
        self.user_project_path = Path(config['project']['path'])  # User's actual project

    def generate_compose(self) -> Dict:
        """Generate docker-compose.yml configuration"""
        docker_config = self.config["docker"]
        gpu_config = self.config["hardware"].get("gpu")

        # Get port configuration
        weaviate_port = self.config.get("ports", {}).get("weaviate", 42080)
        ollama_port = self.config.get("ports", {}).get("ollama", 42434)
        backend_port = self.config.get("ports", {}).get("backend", 42000)

        # Get ACOLYTE source directory (always ~/.acolyte)
        acolyte_src = str(Path.home() / ".acolyte")

        # Base compose configuration
        compose = {
            "version": "3.8",
            "services": {
                "weaviate": {
                    "image": "cr.weaviate.io/semitechnologies/weaviate:1.24.1",
                    "container_name": "acolyte-weaviate",
                    "restart": "unless-stopped",
                    "ports": [f"{weaviate_port}:8080", "50051:50051"],
                    "environment": [
                        "QUERY_DEFAULTS_LIMIT=25",
                        "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true",
                        "PERSISTENCE_DATA_PATH=/var/lib/weaviate",
                        "DEFAULT_VECTORIZER_MODULE=none",
                        "ENABLE_MODULES=none",
                        "ENABLE_API_BASED_MODULES=true",
                        "CLUSTER_HOSTNAME=node1",
                        "LOG_LEVEL=warn",
                    ],
                    "volumes": ["./weaviate:/var/lib/weaviate"],
                    "healthcheck": {
                        "test": [
                            "CMD",
                            "curl",
                            "-f",
                            "http://localhost:8080/v1/.well-known/ready",
                        ],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5,
                    },
                },
                "ollama": {
                    "image": "ollama/ollama:latest",
                    "container_name": "acolyte-ollama",
                    "restart": "unless-stopped",
                    "ports": [f"{ollama_port}:11434"],
                    "volumes": ["./ollama:/root/.ollama"],
                    "environment": [
                        "NVIDIA_VISIBLE_DEVICES=all",
                        "GODEBUG=x509ignoreCN=0",
                    ],
                    "deploy": {
                        "resources": {
                            "limits": {
                                "cpus": docker_config["cpu_limit"],
                                "memory": docker_config["memory_limit"],
                            }
                        }
                    },
                    "healthcheck": {
                        "test": [
                            "CMD",
                            "curl",
                            "-f",
                            "http://localhost:11434/api/tags",
                        ],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5,
                    },
                },
                "backend": {
                    "build": {
                        "context": acolyte_src,  # Build from ACOLYTE installation
                        "dockerfile": "./Dockerfile",
                    },
                    "container_name": "acolyte-backend",
                    "restart": "unless-stopped",
                    "ports": [f"{backend_port}:8000"],
                    "environment": [
                        "WEAVIATE_URL=http://weaviate:8080",
                        "OLLAMA_URL=http://ollama:11434",
                        "OLLAMA_MODEL=acolyte",
                        "PYTHONUNBUFFERED=1",
                        f"ACOLYTE_PROJECT_ROOT={self.user_project_path}",
                        "DATA_DIR=/data",
                    ],
                    "volumes": self._get_backend_volumes(acolyte_src),
                    "depends_on": ["weaviate", "ollama"],
                    "healthcheck": {
                        "test": [
                            "CMD",
                            "curl",
                            "-f",
                            "http://localhost:8000/api/health",
                        ],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5,
                    },
                },
            },
            "volumes": {
                "weaviate-data": {"driver": "local"},
                "ollama-models": {"driver": "local"},
            },
            "networks": {"acolyte-network": {"driver": "bridge"}},
        }

        # Add GPU support if available
        if docker_config.get("gpu_enabled") and gpu_config:
            if gpu_config["type"] == "nvidia":
                # Auto-detect GPU libraries
                gpu_libs = GPUDetector.find_nvidia_libraries()
                gpu_libs = cast(Dict[str, List[str]], gpu_libs)
                services = cast(Dict[str, Any], compose["services"])
                ollama_service = cast(Dict[str, Any], services["ollama"])

                if isinstance(gpu_libs, dict) and "volumes" in gpu_libs and gpu_libs["volumes"]:
                    ollama_service["volumes"].extend(gpu_libs["volumes"])
                    logger.info(f"Added {len(gpu_libs['volumes'])} GPU library volumes")

                if isinstance(gpu_libs, dict) and "devices" in gpu_libs and gpu_libs["devices"]:
                    ollama_service["devices"] = gpu_libs["devices"]
                    logger.info(f"Added {len(gpu_libs['devices'])} GPU devices")

        # Add network to all services
        services = cast(Dict[str, Any], compose["services"])
        for service in services.values():
            service["networks"] = ["acolyte-network"]

        compose = cast(Dict[str, Any], compose)
        return compose

    def _get_backend_volumes(self, acolyte_src: str) -> List[str]:
        """Get backend volumes based on installation mode."""
        volumes = [
            f"{self.project_dir}/.acolyte:/.acolyte:ro",  # Project config
            f"{self.project_dir}/data:/data",  # Project data
            f"{self.user_project_path}:/project:ro",  # User's project (read-only)
        ]

        # Only mount source code if it exists (development mode)
        src_path = Path(acolyte_src) / "src"
        if src_path.exists():
            volumes.insert(0, f"{acolyte_src}/src:/app/src:ro")  # ACOLYTE source code (read-only)

        return volumes

    def save_compose(self, compose: Dict) -> bool:
        """Save docker-compose.yml file"""
        try:
            # Create infra directory
            infra_dir = self.project_dir / "infra"
            infra_dir.mkdir(parents=True, exist_ok=True)

            compose_file = infra_dir / "docker-compose.yml"

            # Backup if exists
            if compose_file.exists():
                backup_file = compose_file.with_suffix(".yml.backup")
                import shutil

                shutil.copy2(compose_file, backup_file)
                logger.info(f"Created backup: {backup_file}")

            with open(compose_file, "w", encoding="utf-8") as f:
                yaml.dump(compose, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Docker compose file saved: {compose_file}")
            return True

        except (IOError, OSError) as e:
            logger.error(f"Error saving docker-compose.yml: {type(e).__name__}: {e}")
            return False
        except yaml.YAMLError as e:
            logger.error(f"Error serializing docker-compose.yml: {e}")
            return False

    def generate_global_dockerfile(self) -> bool:
        """Generate Dockerfile for backend service in ACOLYTE installation"""
        try:
            # This goes in the global ACOLYTE installation, not in project
            acolyte_src = str(Path.home() / ".acolyte")

            # Check if src directory exists (development mode)
            src_exists = (Path(acolyte_src) / "src").exists()

            if src_exists:
                # Development mode - build from local source
                dockerfile_content = """FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    gcc \\
    g++ \\
    make \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ /app/src/

# Copy pyproject.toml if exists
COPY pyproject.toml* poetry.lock* ./

# Install poetry and dependencies
RUN pip install --no-cache-dir poetry && \\
    poetry config virtualenvs.create false && \\
    poetry install --only main --no-interaction --no-ansi --no-root

# Create non-root user
RUN useradd -m -u 1000 acolyte && \\
    chown -R acolyte:acolyte /app

USER acolyte

# Add src to Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "-m", "uvicorn", "acolyte.api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
            else:
                # Production mode - install from pip
                dockerfile_content = """FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    gcc \\
    g++ \\
    make \\
    && rm -rf /var/lib/apt/lists/*

# Install acolyte from pip
RUN pip install --no-cache-dir acolyte

# Create non-root user
RUN useradd -m -u 1000 acolyte

USER acolyte

# Expose port
EXPOSE 8000

# Run the API using the installed acolyte module
CMD ["python", "-m", "uvicorn", "acolyte.api:app", "--host", "0.0.0.0", "--port", "8000"]
"""

            dockerfile_path = Path(acolyte_src) / "Dockerfile"

            # Backup if exists
            if dockerfile_path.exists():
                backup_path = dockerfile_path.with_suffix(".backup")
                import shutil

                shutil.copy2(dockerfile_path, backup_path)
                logger.info(f"Created backup: {backup_path}")

            with open(dockerfile_path, "w", encoding="utf-8") as f:
                f.write(dockerfile_content)

            logger.info(f"Dockerfile created: {dockerfile_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating Dockerfile: {e}")
            return False
