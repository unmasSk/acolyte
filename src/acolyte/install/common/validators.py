#!/usr/bin/env python3
"""
Input validation utilities for ACOLYTE
"""

import re
import socket
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
    """Sanitize string for YAML output to prevent injection and formatting issues"""
    if not value:
        return ""

    # Remove or escape YAML control characters that could cause injection
    # These characters can break YAML structure or cause parsing issues
    dangerous_chars = {
        '\x00': '',  # Null byte
        '\x01': '',  # Start of heading
        '\x02': '',  # Start of text
        '\x03': '',  # End of text
        '\x04': '',  # End of transmission
        '\x05': '',  # Enquiry
        '\x06': '',  # Acknowledge
        '\x07': '',  # Bell
        '\x08': '',  # Backspace
        '\x0b': '',  # Vertical tab
        '\x0c': '',  # Form feed
        '\x0e': '',  # Shift out
        '\x0f': '',  # Shift in
        '\x10': '',  # Data link escape
        '\x11': '',  # Device control 1
        '\x12': '',  # Device control 2
        '\x13': '',  # Device control 3
        '\x14': '',  # Device control 4
        '\x15': '',  # Negative acknowledge
        '\x16': '',  # Synchronous idle
        '\x17': '',  # End of transmission block
        '\x18': '',  # Cancel
        '\x19': '',  # End of medium
        '\x1a': '',  # Substitute
        '\x1b': '',  # Escape
        '\x1c': '',  # File separator
        '\x1d': '',  # Group separator
        '\x1e': '',  # Record separator
        '\x1f': '',  # Unit separator
    }

    # Clean the string
    sanitized = value
    for char, replacement in dangerous_chars.items():
        sanitized = sanitized.replace(char, replacement)

    # Remove leading/trailing whitespace and normalize line endings
    sanitized = sanitized.strip().replace('\r\n', '\n').replace('\r', '\n')

    # Escape quotes if the string contains them and might be interpreted as YAML
    if '"' in sanitized or "'" in sanitized:
        # Use YAML's literal scalar style for strings with quotes
        if '\n' in sanitized:
            # Multi-line string with quotes - use literal style
            return f"|\n  {sanitized.replace(chr(10), chr(10) + '  ')}"
        else:
            # Single-line string with quotes - escape properly
            sanitized = sanitized.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{sanitized}"'

    return sanitized


def validate_path(path: str) -> bool:
    """Validate path doesn't contain dangerous patterns including encoded traversal attempts"""
    try:
        import urllib.parse

        # Normalize and resolve the path
        normalized = os.path.normpath(os.path.expanduser(path))

        # Decode URL-encoded sequences that might contain traversal attempts
        try:
            decoded_path = urllib.parse.unquote(normalized)
        except Exception:
            # If URL decoding fails, use the original path
            decoded_path = normalized

        # Split path into segments for analysis
        path_segments = decoded_path.split(os.sep)

        # Enhanced path traversal detection
        traversal_patterns = [
            "..",  # Standard traversal
            "%2e%2e",  # URL-encoded ".."
            "%2E%2E",  # URL-encoded ".." (uppercase)
            "..%2f",  # Mixed encoding
            "%2e%2e%2f",  # Fully encoded
            "..%5c",  # Windows backslash encoding
            "%2e%2e%5c",  # Windows backslash fully encoded
        ]

        # Check for traversal patterns in the path
        path_lower = decoded_path.lower()
        for pattern in traversal_patterns:
            if pattern in path_lower:
                return False

        # Check individual segments for traversal attempts
        for segment in path_segments:
            # Decode each segment individually
            try:
                decoded_segment = urllib.parse.unquote(segment)
            except Exception:
                decoded_segment = segment

            # Check if segment represents traversal
            if decoded_segment in ["..", ".", ""]:
                return False

            # Check for encoded traversal in segment
            if any(pattern in decoded_segment.lower() for pattern in traversal_patterns):
                return False

        # Check for shell metacharacters (but allow them in filenames)
        dangerous_chars = ["$", "|", ">", "<", "&", ";"]
        # Only check at path boundaries to avoid false positives
        return not any(char in os.path.basename(path) for char in dangerous_chars)

    except (ValueError, OSError, UnicodeDecodeError):
        return False
