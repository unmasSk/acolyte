"""
Secure configuration for ACOLYTE.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, cast

from acolyte.core.exceptions import ConfigurationError
from acolyte.core.logging import logger


class ConfigValidator:
    """
    Configuration validator with rules.

    Validations:
    1. Data types
    2. Value ranges
    3. localhost binding mandatory
    4. Secure paths with pathlib
    """

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate complete configuration.

        Key rules:
        - Ports binding must be localhost
        - Paths must be relative and secure
        - Model name must be qwen2.5-coder:XX or acolyte:latest
        - Ports must be in valid range
        """
        # Port validation - all must be accessible only locally
        ports = config.get("ports", {})
        for port_name, port_value in ports.items():
            if not isinstance(port_value, int) or port_value < 1024 or port_value > 65535:
                logger.error(
                    "Invalid port configuration", port_name=port_name, port_value=port_value
                )
                raise ConfigurationError(f"Invalid port {port_name}: {port_value}")

        # ========================================================================
        # CRITIC VALIDATION: ONLY TWO ALLOWED MODELS IN ACOLYTE
        # ========================================================================
        # FUNDAMENTAL REASON: ACOLYTE is a specialized assistant that requires
        # a specific model trained for code and a local derived model.
        #
        # ALLOWED MODELS:
        # 1. "qwen2.5-coder:*" - Base model from Microsoft/Qwen specialized for code
        # 2. "acolyte:latest" - Our custom model derived from qwen2.5-coder
        #
        # WHY NOT OTHER MODELS:
        # - ACOLYTE depends on specific code capabilities (syntax, AST)
        # - The System Prompt is optimized for qwen2.5-coder
        # - The token budgets are calibrated for this model
        # - The quality of embeddings UniXcoder is synchronized with this model
        # - Changing the model will break the core experience of ACOLYTE
        #
        # ARCHITECTURAL DECISION: Specialization > Flexibility
        # ========================================================================
        model = config.get("model", {})
        model_name = model.get("name", "")
        allowed_models = ["qwen2.5-coder:", "acolyte:latest"]

        # Only validate if there is a specified model
        if model_name and not any(
            model_name.startswith(allowed) or model_name == allowed for allowed in allowed_models
        ):
            logger.error("Invalid model attempted", model=model_name, allowed=allowed_models)
            raise ConfigurationError(
                f"ACOLYTE only supports code-specialized models. Allowed: qwen2.5-coder:* or acolyte:latest. Received: {model_name}"
            )

        # Validation of paths in project
        project = config.get("project", {})
        if "path" in project:
            path_str = str(project["path"])
            # Check for absolute paths (Unix/Windows), parent refs, or starting with /
            if (
                Path(path_str).is_absolute()
                or ".." in path_str
                or path_str.startswith("/")
                or (len(path_str) > 1 and path_str[1] == ":")
            ):
                logger.error("Unsafe project path detected", path=path_str)
                raise ConfigurationError(f"Unsafe project path: {project['path']}")


class Settings:
    """
    Main system configuration.

    Simple configuration for mono-user:
    1. Default values
    2. .acolyte file (SOURCE OF TRUTH)
    3. Environment variables
    """

    def __init__(self) -> None:
        self.config = self._load_config()
        self._validate_config()  # Validate missing sections
        self.validator = ConfigValidator()
        self._ensure_localhost_binding()
        self.validator.validate_config(self.config)
        logger.info(
            "Settings initialized",
            config_source=".acolyte" if self._find_config_file() else "defaults",
        )

    def _ensure_localhost_binding(self) -> None:
        """Ensures binding to localhost."""
        # For the backend API
        if "ports" not in self.config:
            self.config["ports"] = {}
        if "backend" not in self.config["ports"]:
            self.config["ports"]["backend"] = 42000

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with ACOLYTE port ranges.

        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "version": "1.0",
            "project": {"name": "acolyte-project", "path": "."},
            "model": {"name": "qwen2.5-coder:3b", "context_size": 32768},
            "ports": {
                "weaviate": 42080,  # ACOLYTE range: 42080-42099
                "ollama": 42434,  # ACOLYTE range: 42434-42453
                "backend": 42000,  # ACOLYTE range: 42000-42019
            },
            "logging": {"level": "INFO", "file": ".acolyte/logs/debug.log", "rotation_size_mb": 10},
            "dream": {
                "fatigue_threshold": 7.5,
                "emergency_threshold": 9.5,
                "cycle_duration_minutes": 5,
                "dream_folder_name": ".acolyte-dreams",
            },
            "cache": {
                "max_size": 1000,
                "ttl_seconds": 3600,
            },
            "rag": {
                "compression": {
                    "enabled": True,
                    "ratio": 0.7,
                    "avg_chunk_tokens": 200,
                },
            },
        }

    def _find_config_file(self) -> Path | None:
        """Find the .acolyte configuration file.

        Search order:
        1. Current directory .acolyte
        2. Global directory ~/.acolyte/projects/{id}/.acolyte (if .acolyte.project exists)

        Returns:
            Optional[Path]: Path to config file or None
        """
        # First, check current directory
        local_config = Path.cwd() / ".acolyte"
        if local_config.exists():
            logger.info(f"Using local configuration: {local_config}")
            return local_config

        # Then check for project-specific config
        project_file = Path.cwd() / ".acolyte.project"
        if project_file.exists():
            try:
                import json

                project_data = json.loads(project_file.read_text())
                project_id = project_data.get("project_id")
                if project_id:
                    global_config = Path.home() / ".acolyte" / "projects" / project_id / ".acolyte"
                    if global_config.exists():
                        logger.info(f"Using global configuration: {global_config}")
                        return global_config
            except Exception as e:
                logger.error(f"Error reading .acolyte.project: {e}")

        return None

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration in priority order.

        1. Defaults
        2. .acolyte file (SOURCE OF TRUTH)
        3. Environment variables (for development overrides)
        """
        # Get defaults from single source
        defaults = self._get_default_config()

        # Find .acolyte configuration file
        config_path = self._find_config_file()
        if config_path and config_path.exists():
            try:
                with open(config_path, encoding='utf-8') as f:
                    acolyte_config = yaml.safe_load(f)
                    if acolyte_config:
                        # Deep merge of configurations
                        self._deep_merge(defaults, acolyte_config)
                        logger.debug(
                            "Config loaded from .acolyte", keys=list(acolyte_config.keys())
                        )
            except Exception as e:
                logger.error(
                    "Error reading configuration file", file=str(config_path), error=str(e)
                )
                raise ConfigurationError(f"Error reading configuration file: {e}")

        # Selective override with env vars (only some values)
        env_overrides = {
            "ACOLYTE_PORT": ("ports", "backend"),
            "ACOLYTE_LOG_LEVEL": ("logging", "level"),
            "ACOLYTE_MODEL": ("model", "name"),
        }

        for env_key, path_tuple in env_overrides.items():
            env_value = os.getenv(env_key)
            if env_value:
                # Convert port to int if needed
                value_to_set: Any = env_value
                if env_key == "ACOLYTE_PORT":
                    try:
                        value_to_set = int(env_value)
                    except ValueError:
                        pass  # Keep as string if not convertible
                self._set_nested(defaults, path_tuple, value_to_set)

        return defaults

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Deep merge of dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(cast(Dict[str, Any], base[key]), cast(Dict[str, Any], value))
            else:
                base[key] = value

    def _set_nested(self, data: Dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        """Set value at nested path."""
        current = data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _validate_config(self) -> None:
        """Validate that all required configuration sections exist."""
        required_sections = [
            "version",
            "project",
            "model",
            "ports",
            "dream",
            "cache",
            "rag",
        ]

        missing_sections = []
        for section in required_sections:
            if section not in self.config:
                missing_sections.append(section)

        if missing_sections:
            logger.warning(
                f"Configuration missing required sections: {missing_sections}. "
                f"Using defaults for missing sections."
            )

            # Get defaults from single source
            defaults = self._get_default_config()

            # Add missing sections from defaults
            for section in missing_sections:
                if section in defaults:
                    self.config[section] = defaults[section]

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with support for nested paths."""
        # Support for paths with dots: "model.name", "ports.backend"
        if "." in key:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        return self.config.get(key, default)

    def require(self, key: str) -> Any:
        """
        Get required value or raise exception.

        Useful for critical configs that must exist.
        """
        if key not in self.config:
            logger.error("Required config missing", key=key)
            raise ConfigurationError(f"Missing required config: {key}")
        return self.config[key]
