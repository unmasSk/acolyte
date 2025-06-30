#!/usr/bin/env python3
"""
Input validation utilities for ACOLYTE
"""

import re
import socket


def validate_project_name(name: str) -> bool:
    """Validate project name format"""
    pattern = r"^[a-zA-Z0-9_\-\.]{1,64}$"
    return bool(re.match(pattern, name))


def validate_port(port: int) -> bool:
    """Validate port number and check if available"""
    if not 1024 <= port <= 65535:
        return False

    # Check if port is available
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))
            return True
    except OSError:
        return False


def sanitize_yaml_string(value: str) -> str:
    """Sanitize string for YAML output"""
    if not value:
        return ""

    # Escape special characters
    value = value.replace('"', '\\"')
    value = value.replace("\n", "\\n")
    value = value.replace("\t", "\\t")

    return value


def validate_path(path: str) -> bool:
    """Validate path doesn't contain dangerous patterns"""
    dangerous_patterns = ["..", "~", "$", "|", ">", "<", "&", ";"]
    return not any(pattern in path for pattern in dangerous_patterns)
