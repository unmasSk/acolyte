# ACOLYTE Installation Scripts

This directory contains the installation system for ACOLYTE.

## Structure

```
scripts/install/
├── init.py          # Project initialization (acolyte init)
├── install.py       # Full installation process (acolyte install)  
├── common/          # Shared utilities
│   ├── __init__.py
│   ├── ui.py        # Terminal UI (colors, progress bars, animations)
│   ├── hardware.py  # Hardware detection and model recommendations
│   ├── docker.py    # Docker compose generation with GPU support
│   └── validators.py # Input validation
└── README.md        # This file
```

## How It Works

### Global Installation

ACOLYTE is installed globally to `~/.acolyte` (or `%USERPROFILE%\.acolyte` on Windows) using:
- `install.sh` (Linux/Mac)
- `install.bat` (Windows)

### Per-Project Setup

1. **`acolyte init`** - Creates project configuration
   - Detects hardware (CPU, RAM, GPU)
   - Recommends optimal model
   - Creates configuration in `~/.acolyte/projects/{project_id}/`
   - Only creates `.acolyte.project` in user's project

2. **`acolyte install`** - Installs services for the project
   - Downloads Ollama model
   - Creates Docker infrastructure
   - Initializes databases
   - Sets up vector store

## Key Features

### Hardware Detection
- Detects OS, CPU, RAM, disk space
- GPU detection for NVIDIA, AMD, Apple Silicon
- Recommends optimal model based on resources

### Docker Generation
- Creates docker-compose.yml with proper resource limits
- Auto-detects and mounts GPU libraries for acceleration
- Configures health checks and networking

### Clean Project Structure
- User's project only has `.acolyte.project` file
- All data/infrastructure in `~/.acolyte/projects/{id}/`
- No pollution of user's repository

## Environment Variables

Scripts use these environment variables:
- `ACOLYTE_PROJECT_ID` - Unique project identifier
- `ACOLYTE_PROJECT_PATH` - Path to user's project
- `ACOLYTE_GLOBAL_DIR` - ACOLYTE installation directory
- `ACOLYTE_PROJECT_NAME` - Project name

## UI Components

The `common/ui.py` module provides:
- Colored terminal output
- ASCII art logo
- Progress bars with "consciousness tips"
- Typing animations
- Spinner animations

## Model Selection

Available models (qwen2.5-coder):
- 0.5B - 2GB RAM minimum
- 1.5B - 4GB RAM minimum  
- 3B - 8GB RAM minimum
- 7B - 16GB RAM minimum
- 14B - 32GB RAM minimum
- 32B - 64GB RAM minimum

## Dependencies

- Python 3.11+
- PyYAML (for configuration)
- psutil (for hardware detection)
- Docker (for services)
- Poetry (for Python dependencies)
