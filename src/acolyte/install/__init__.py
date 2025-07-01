"""
ACOLYTE Installation module

This module contains all installation and initialization logic as Python functions.
No more subprocess calls to external scripts!
"""

# Re-export commonly used functions
from .common import (
    Colors,
    ACOLYTE_LOGO,
    animate_text,
    show_spinner,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_step,
    print_progress_bar,
    SystemDetector,
    ModelRecommender,
    DockerGenerator,
    GPUDetector,
    validate_project_name,
    validate_port,
    sanitize_yaml_string,
    PortManager,
)

from .resources_manager import (
    get_resource_path,
    get_resource_text,
    get_resource_bytes,
    copy_resource_to_path,
    get_git_hook,
    get_docker_template,
    get_config_template,
    get_modelfile,
)

from .init import (
    DependencyChecker,
    ProjectValidator,
    GitHooksManager,
    ProjectInitializer,
)

from .installer import (
    ProjectInfoCollector,
    AdvancedConfiguration,
    LanguageConfiguration,
    ProjectInstaller,
)

__all__ = [
    # UI utilities
    "Colors",
    "ACOLYTE_LOGO",
    "animate_text",
    "show_spinner",
    "print_header",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "print_step",
    "print_progress_bar",
    # Hardware detection
    "SystemDetector",
    "ModelRecommender",
    "DockerGenerator",
    "GPUDetector",
    # Validators
    "validate_project_name",
    "validate_port",
    "sanitize_yaml_string",
    # Port management
    "PortManager",
    # Resource management
    "get_resource_path",
    "get_resource_text",
    "get_resource_bytes",
    "copy_resource_to_path",
    "get_git_hook",
    "get_docker_template",
    "get_config_template",
    "get_modelfile",
    # Init module
    "DependencyChecker",
    "ProjectValidator",
    "GitHooksManager",
    "ProjectInitializer",
    # Installer module
    "ProjectInfoCollector",
    "AdvancedConfiguration",
    "LanguageConfiguration",
    "ProjectInstaller",
]
