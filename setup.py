#!/usr/bin/env python
"""
Setup script for ACOLYTE
This file is optional - modern pip can install directly from pyproject.toml
Included for compatibility with older systems
"""
import tomllib
import traceback
from setuptools import setup, find_packages
from setuptools.command.install import install
from pathlib import Path
import subprocess
import sys

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from pyproject.toml

with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
    version = pyproject["project"]["version"]


class VerboseInstall(install):
    """Custom install command with better user feedback"""

    def run(self):
        print("\n" + "=" * 60)
        print("ACOLYTE INSTALLATION")
        print("=" * 60)
        print("\nInstalling dependencies (this includes PyTorch ~2GB)")
        print("Grab a coffee, this will take 2-5 minutes...")
        print("This is a one-time download - all projects share the same models")
        print("\n" + "=" * 60)

        # Run the original install
        install.run(self)

        # Run post-install setup
        print("\n" + "=" * 60)
        print("Running post-installation setup...")
        try:
            self.setup_acolyte_home()
        except PermissionError as e:
            print(f"❌ Permission error during post-installation setup: {e}")
            print("   This usually means insufficient permissions to create ~/.acolyte directory")
            print("   Solutions:")
            print("   • Run with elevated privileges: sudo pip install acolyte")
            print("   • Or create the directory manually: mkdir -p ~/.acolyte")
            print("   • Then run: python -m acolyte.install.post_install")
        except OSError as e:
            print(f"❌ System error during post-installation setup: {e}")
            print("   This could be due to disk space, file system issues, or path problems")
            print("   Solutions:")
            print("   • Check available disk space")
            print("   • Ensure ~/.acolyte is writable")
            print("   • Try running manually: python -m acolyte.install.post_install")
        except ImportError as e:
            print(f"❌ Import error during post-installation setup: {e}")
            print("   This suggests a dependency or module loading issue")
            print("   Solutions:")
            print("   • Reinstall ACOLYTE: pip install --force-reinstall acolyte")
            print("   • Check Python environment and dependencies")
            print("   • Try running manually: python -m acolyte.install.post_install")
        except Exception as e:
            print(f"❌ Unexpected error during post-installation setup: {e}")
            print("   Full error details:")
            print("   " + "=" * 50)
            traceback.print_exc()
            print("   " + "=" * 50)
            print("   Troubleshooting steps:")
            print("   • Check the error details above")
            print("   • Try running manually: python -m acolyte.install.post_install")
            print("   • Report this issue with the full traceback at:")
            print("     https://github.com/unmasSk/acolyte/issues")

        print("\nInstallation complete!")
        print("Next: Run 'acolyte init' in your project")
        print("Run 'acolyte doctor' to verify your setup")
        print("=" * 60 + "\n")

    def setup_acolyte_home(self):
        """Setup ~/.acolyte directory with necessary files."""
        acolyte_home = Path.home() / ".acolyte"
        acolyte_home.mkdir(exist_ok=True)

        # Create a marker file to indicate pip installation
        marker_file = acolyte_home / ".pip_installed"
        marker_file.touch()

        print(f"ACOLYTE home directory created: {acolyte_home}")


setup(
    name="acolyte",
    version=version,
    author="Bextia",
    description="Local AI Programming Assistant with infinite memory and OpenAI-compatible API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/unmasSk/acolyte",
    project_urls={
        "Bug Tracker": "https://github.com/unmasSk/acolyte/issues",
        "Documentation": "https://github.com/unmasSk/acolyte/tree/main/docs",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.110.0",
        "pydantic>=2.6.0",
        "loguru>=0.7.2",
        "gitpython>=3.1.40",
        "uvicorn[standard]>=0.29.0",
        "pyyaml>=6.0.0",
        "numpy>=2.3.0",
        "transformers>=4.52.4",
        "aiohttp>=3.9.0",
        "asyncio>=3.4.3",
        "psutil>=7.0.0",
        "tree-sitter>=0.20.4",
        "tree-sitter-languages>=1.10.2",
        "torch>=2.7.1",
        "click>=8.1.0",
        "rich>=13.0.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
        "weaviate-client>=3.26.7,<4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.4.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=5.0.0",
            "mypy>=1.8.0",
            "black>=22.0.0",
            "ruff>=0.12.0",
            "types-pyyaml>=6.0.12",
            "types-requests>=2.32.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "acolyte=acolyte.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "acolyte": [
            "**/*.yaml",
            "**/*.yml",
            "**/*.json",
            "**/*.txt",
            "**/*.sql",
            "install/resources/**/*",
        ],
    },
    cmdclass={
        'install': VerboseInstall,
    },
)
