#!/usr/bin/env python3
"""
Input validation utilities for ACOLYTE
"""

import re
import socket
import yaml
import os


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
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def sanitize_yaml_string(value: str) -> str:
    """Sanitize string for YAML output"""
    if not value:
        return ""

    return yaml.dump(value).strip()


def validate_path(path: str) -> bool:
    """Validate path doesn't contain dangerous patterns"""
    try:
        # Normalize and resolve the path
        normalized = os.path.normpath(os.path.expanduser(path))

        # Check for path traversal attempts
        if ".." in normalized.split(os.sep):
            return False

        # Check for shell metacharacters (but allow them in filenames)
        dangerous_chars = ["$", "|", ">", "<", "&", ";"]
        # Only check at path boundaries to avoid false positives
        return not any(char in os.path.basename(path) for char in dangerous_chars)
    except (ValueError, OSError):
        return False
