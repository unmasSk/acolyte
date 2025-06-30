"""
Centralized datetime utilities for ACOLYTE.

Provides consistent datetime handling across the entire project.
All datetimes are handled in UTC to avoid timezone issues.

This module replaces all direct uses of datetime.utcnow() with
centralized, mockeable functions that ensure consistency and
enable proper testing.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Union


def utc_now() -> datetime:
    """
    Get current UTC datetime.

    This is the primary replacement for datetime.utcnow() throughout
    the project. Always returns timezone-aware UTC datetime.

    Returns:
        Current datetime in UTC with timezone info
    """
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """
    Get current UTC datetime as ISO string.

    This replaces the common pattern datetime.utcnow().isoformat()
    and ensures consistent formatting with 'Z' suffix.

    Returns:
        Current datetime in ISO format with 'Z' suffix
        Example: "2024-01-15T10:30:45.123456Z"
    """
    return utc_now().isoformat().replace('+00:00', 'Z')


def ensure_utc(dt: datetime) -> datetime:
    """
    Ensure datetime is in UTC timezone.

    Handles both naive and timezone-aware datetimes, converting
    them to UTC as needed.

    Args:
        dt: Datetime to convert

    Returns:
        Datetime in UTC with timezone info

    Raises:
        ValueError: If datetime has non-UTC timezone and can't be converted
    """
    if dt.tzinfo is None:
        # Naive datetime - assume it's already UTC
        return dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo == timezone.utc:
        # Already UTC
        return dt
    else:
        # Convert to UTC
        return dt.astimezone(timezone.utc)


def parse_iso_datetime(iso_string: str) -> datetime:
    """
    Parse ISO datetime string to datetime object.

    Handles various ISO formats commonly found in the project:
    - 2024-01-01T12:00:00
    - 2024-01-01T12:00:00Z
    - 2024-01-01T12:00:00+00:00
    - 2024-01-01T12:00:00.123456Z

    Args:
        iso_string: ISO format datetime string

    Returns:
        Parsed datetime in UTC with timezone info

    Raises:
        ValueError: If string cannot be parsed as ISO datetime
    """
    # Handle 'Z' suffix
    if iso_string.endswith('Z'):
        iso_string = iso_string[:-1] + '+00:00'

    try:
        # Try parsing with timezone
        dt = datetime.fromisoformat(iso_string)
        return ensure_utc(dt)
    except ValueError:
        try:
            # Try without timezone (assume UTC)
            dt = datetime.fromisoformat(iso_string)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError as e:
            raise ValueError(f"Invalid ISO datetime string: {iso_string}") from e


def format_iso(dt: datetime) -> str:
    """
    Format datetime to ISO string with Z suffix.

    Ensures consistent ISO formatting across the project.
    Always outputs with 'Z' suffix for UTC times.

    Args:
        dt: Datetime to format (will be converted to UTC)

    Returns:
        ISO formatted string with 'Z' suffix
        Example: "2024-01-15T10:30:45.123456Z"
    """
    utc_dt = ensure_utc(dt)
    return utc_dt.isoformat().replace('+00:00', 'Z')


def time_ago(dt: Union[datetime, str]) -> str:
    """
    Get human-readable time difference from now.

    Useful for displaying "last updated" or "created" times
    in a user-friendly format.

    Args:
        dt: Past datetime or ISO string

    Returns:
        Human readable string like "2 hours ago", "3 days ago", etc.
    """
    if isinstance(dt, str):
        dt = parse_iso_datetime(dt)

    dt = ensure_utc(dt)
    now = utc_now()

    # Ensure we're comparing UTC times
    if dt > now:
        return "in the future"

    diff = now - dt

    # Calculate the most appropriate unit
    if diff.days > 365:
        years = diff.days // 365
        return f"{years} year{'s' if years != 1 else ''} ago"
    elif diff.days > 30:
        months = diff.days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"

    # For times less than a day, calculate hours/minutes/seconds
    total_seconds = diff.seconds

    if total_seconds >= 3600:  # 1 hour or more
        hours = total_seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif total_seconds >= 60:  # 1 minute or more
        minutes = total_seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "just now"


def add_time(dt: datetime, **kwargs) -> datetime:
    """
    Add time to datetime with UTC preservation.

    Convenience function for adding time deltas while
    ensuring the result remains in UTC.

    Args:
        dt: Base datetime
        **kwargs: Arguments for timedelta (days, hours, minutes, seconds, etc)

    Returns:
        New datetime in UTC

    Example:
        tomorrow = add_time(utc_now(), days=1)
        in_an_hour = add_time(meeting_time, hours=1)
    """
    utc_dt = ensure_utc(dt)
    return utc_dt + timedelta(**kwargs)


# For testing and mocking
_mock_time: Optional[datetime] = None


def set_mock_time(dt: Optional[datetime]) -> None:
    """
    Set mock time for testing.

    Allows tests to control the current time returned by
    utc_now_testable(). Set to None to disable mocking.

    Args:
        dt: Mock datetime to use, or None to use real time

    Example:
        # In tests
        test_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        set_mock_time(test_time)

        # Code under test will see the mocked time
        current = utc_now_testable()  # Returns test_time

        # Cleanup
        set_mock_time(None)
    """
    global _mock_time
    _mock_time = ensure_utc(dt) if dt else None


def utc_now_testable() -> datetime:
    """
    Get current time (mockable for tests).

    This function respects mock time if set via set_mock_time().
    Use this in code that needs to be tested with specific times.

    Returns:
        Mock time if set, otherwise current UTC time
    """
    return _mock_time if _mock_time else utc_now()


# Commonly used constants
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
"""Unix epoch as timezone-aware datetime"""

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
"""Standard ISO format string with microseconds and Z suffix"""
