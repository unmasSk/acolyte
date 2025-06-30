"""
Secure configuration for ACOLYTE.
"""

from typing import Dict, Any

class ConfigValidator:
    """
    Configuration validator with rules.

    Validations:
    1. Data types
    2. Value ranges
    3. localhost binding mandatory
    4. Secure paths with pathlib
    """

    def validate_config(self, config: Dict[str, Any]) -> None: ...

class Settings:
    """
    Main system configuration.

    Simple configuration for mono-user:
    1. Default values
    2. .acolyte file (SOURCE OF TRUTH)
    3. Environment variables
    """

    config: Dict[str, Any]
    validator: ConfigValidator

    def __init__(self) -> None: ...
    def _ensure_localhost_binding(self) -> None: ...
    def _load_config(self) -> Dict[str, Any]: ...
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None: ...
    def _set_nested(self, data: Dict[str, Any], path: tuple[str, ...], value: Any) -> None: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def require(self, key: str) -> Any: ...
