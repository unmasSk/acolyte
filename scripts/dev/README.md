# Development Scripts

This directory contains scripts used during ACOLYTE development.

## Scripts

### dev.sh

Development helper script for testing ACOLYTE locally without installing globally.

**Usage:**
```bash
./scripts/dev/dev.sh <command>
```

**Commands:**
- `init` - Initialize project (dev mode)
- `install` - Install services (dev mode)
- `start` - Start services (dev mode)
- `stop` - Stop services (dev mode)
- `status` - Check status (dev mode)
- `index` - Index files (dev mode)
- `clean` - Clean cache (dev mode)
- `projects` - List projects (dev mode)
- `test` - Run pytest tests
- `lint` - Run linters (ruff, black, mypy)
- `format` - Format code with black and ruff
- `coverage` - Generate coverage report

**Note:** This script sets `ACOLYTE_DEV` environment variable to use source code from the current directory instead of the global installation.
