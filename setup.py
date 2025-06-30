#!/usr/bin/env python
"""
Setup script for ACOLYTE
This file is optional - modern pip can install directly from pyproject.toml
Included for compatibility with older systems
"""
import tomllib
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from pyproject.toml

with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
    version = pyproject["project"]["version"]

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
            "weaviate-client>=3.26.7,<4.0.0",
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
            "scripts/**/*.py",
            "scripts/**/*.sh",
            "scripts/**/*.bat",
        ],
    },
)
