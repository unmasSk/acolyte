"""
Resource management for ACOLYTE installation

Provides functions to access packaged resources using importlib.resources.
Works both in development mode and when installed via pip.
"""

import sys
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources


def get_resource_path(resource_name: str) -> Optional[Path]:
    """
    Get the path to a packaged resource.

    Args:
        resource_name: Name of the resource relative to the resources directory

    Returns:
        Path to the resource or None if not found
    """
    try:
        if sys.version_info >= (3, 9):
            files = resources.files('acolyte.install.resources')
            resource = files / resource_name
            if resource.is_file():
                # Convert Traversable to Path
                return Path(str(resource))
        else:
            # For older Python, use the backport
            with resources.path('acolyte.install.resources', resource_name) as path:
                return Path(path)
    except Exception:
        # Fallback for development mode
        dev_path = Path(__file__).parent / 'resources' / resource_name
        if dev_path.exists():
            return dev_path
    return None


def get_resource_text(resource_name: str) -> Optional[str]:
    """
    Get the text content of a packaged resource.

    Args:
        resource_name: Name of the resource relative to the resources directory

    Returns:
        Text content of the resource or None if not found
    """
    try:
        if sys.version_info >= (3, 9):
            files = resources.files('acolyte.install.resources')
            return (files / resource_name).read_text(encoding='utf-8')
        else:
            # For older Python, use the backport
            return resources.read_text('acolyte.install.resources', resource_name, encoding='utf-8')
    except Exception:
        # Fallback for development mode
        dev_path = Path(__file__).parent / 'resources' / resource_name
        if dev_path.exists():
            return dev_path.read_text(encoding='utf-8')
    return None


def get_resource_bytes(resource_name: str) -> Optional[bytes]:
    """
    Get the binary content of a packaged resource.

    Args:
        resource_name: Name of the resource relative to the resources directory

    Returns:
        Binary content of the resource or None if not found
    """
    try:
        if sys.version_info >= (3, 9):
            files = resources.files('acolyte.install.resources')
            return (files / resource_name).read_bytes()
        else:
            # For older Python, use the backport
            return resources.read_binary('acolyte.install.resources', resource_name)
    except Exception:
        # Fallback for development mode
        dev_path = Path(__file__).parent / 'resources' / resource_name
        if dev_path.exists():
            return dev_path.read_bytes()
    return None


def copy_resource_to_path(resource_name: str, destination: Path) -> bool:
    """
    Copy a packaged resource to a destination path.

    Args:
        resource_name: Name of the resource relative to the resources directory
        destination: Path where to copy the resource

    Returns:
        True if successful, False otherwise
    """
    try:
        content = get_resource_bytes(resource_name)
        if content is None:
            return False

        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Write the content
        destination.write_bytes(content)
        return True
    except Exception:
        return False


def list_resources(directory: str = "") -> list[str]:
    """
    List all resources in a directory.

    Args:
        directory: Subdirectory to list (empty for root)

    Returns:
        List of resource names
    """
    try:
        if sys.version_info >= (3, 9):
            files = resources.files('acolyte.install.resources')
            if directory:
                files = files / directory
            return [f.name for f in files.iterdir() if f.is_file()]
        else:
            # For older Python, this is more limited
            # We'd need to know the exact files beforehand
            return []
    except Exception:
        # Fallback for development mode
        dev_path = Path(__file__).parent / 'resources'
        if directory:
            dev_path = dev_path / directory
        if dev_path.exists():
            return [f.name for f in dev_path.iterdir() if f.is_file()]
    return []


# Specific resource getters
def get_git_hook(hook_name: str) -> Optional[str]:
    """Get a git hook template."""
    return get_resource_text(f'hooks/{hook_name}')


def get_docker_template(template_name: str) -> Optional[str]:
    """Get a Docker template."""
    return get_resource_text(f'docker/{template_name}')


def get_config_template(template_name: str) -> Optional[str]:
    """Get a configuration template."""
    return get_resource_text(f'configs/{template_name}')


def get_modelfile() -> Optional[str]:
    """Get the ACOLYTE Modelfile."""
    return get_resource_text('configs/Modelfile.acolyte')
