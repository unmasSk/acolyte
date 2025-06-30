# 🚀 ACOLYTE Installation Guide

ACOLYTE can now be installed as a proper Python package using pip!

## 📦 Installation Methods

### Method 1: Install with pip (Recommended)

#### From GitHub (Latest)
```bash
# Linux/Mac
pip install git+https://github.com/unmasSk/acolyte.git

# Windows
pip install git+https://github.com/unmasSk/acolyte.git
```

#### From PyPI (When published)
```bash
pip install acolyte
```

#### Development Installation
```bash
# Clone the repository
git clone https://github.com/unmasSk/acolyte.git
cd acolyte

# Install in editable mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Method 2: Quick Install Scripts

We provide convenience scripts that handle pip installation:

#### Linux/Mac
```bash
curl -sSL https://raw.githubusercontent.com/unmasSk/acolyte/main/install-pip.sh | bash
```

#### Windows
```powershell
Invoke-WebRequest -Uri https://raw.githubusercontent.com/unmasSk/acolyte/main/install-pip.bat -OutFile install-pip.bat
.\install-pip.bat
```

### Method 3: Traditional Installation (Legacy)

The original installation scripts are still available:

#### Linux/Mac
```bash
curl -sSL https://raw.githubusercontent.com/unmasSk/acolyte/main/scripts/install.sh | bash
```

#### Windows
```powershell
Invoke-WebRequest -Uri https://raw.githubusercontent.com/unmasSk/acolyte/main/scripts/install.bat -OutFile install.bat
.\install.bat
```

## 🔧 Requirements

- Python 3.11 or newer
- pip (comes with Python)
- Git (for installation from repository)
- Docker (for running services)

## 📁 What Gets Installed

When you install ACOLYTE with pip:

1. **Python Package**: Installed in your Python environment
2. **CLI Command**: `acolyte` command available globally
3. **Dependencies**: All Python dependencies automatically installed

On first run, ACOLYTE will create:
```
~/.acolyte/
├── projects/      # Per-project configurations
├── models/        # Shared models
├── templates/     # Docker templates
├── examples/      # Example configurations
└── logs/          # Global logs
```

## 🚀 Post-Installation

After installation, follow these steps:

```bash
# 1. Verify installation
acolyte --version

# 2. Go to your project
cd /path/to/your/project

# 3. Initialize ACOLYTE
acolyte init

# 4. Configure services (interactive)
acolyte install

# 5. Start services
acolyte start

# 6. Index your project
acolyte index

# 7. Check status
acolyte status
```

## 🔄 Updating

### With pip
```bash
# Update to latest version
pip install --upgrade git+https://github.com/unmasSk/acolyte.git

# Or if installed from PyPI
pip install --upgrade acolyte
```

### Check current version
```bash
acolyte --version
```

## 🗑️ Uninstalling

### Remove Python package
```bash
pip uninstall acolyte
```

### Remove data and configurations (optional)
```bash
# Linux/Mac
rm -rf ~/.acolyte

# Windows
rmdir /s %USERPROFILE%\.acolyte
```

## 🐛 Troubleshooting

### "acolyte: command not found"

The command may not be in your PATH. Try:

```bash
# Linux/Mac
export PATH="$HOME/.local/bin:$PATH"

# Windows
# Add %APPDATA%\Python\Python311\Scripts to your PATH
```

### "pip: command not found"

Ensure pip is installed:
```bash
python -m ensurepip --upgrade
```

### Permission errors

Use `--user` flag:
```bash
pip install --user git+https://github.com/unmasSk/acolyte.git
```

## 🧑‍💻 Development Setup

For development work:

```bash
# Clone repository
git clone https://github.com/unmasSk/acolyte.git
cd acolyte

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
black --check .
mypy src/acolyte
```

## 📋 Installation Options Comparison

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| `pip install` from GitHub | Most users | Simple, standard | Requires Git |
| `pip install` from PyPI | Future release | Simplest | Not yet available |
| Quick install scripts | Beginners | One command | Less control |
| Development mode | Contributors | Editable, includes dev tools | More complex |
| Traditional scripts | Legacy systems | Handles all setup | More complex |

## 🔗 Links

- [Documentation](https://github.com/unmasSk/acolyte/tree/main/docs)
- [CLI Reference](https://github.com/unmasSk/acolyte/tree/main/bin/README.md)
- [Contributing](https://github.com/unmasSk/acolyte/blob/main/CONTRIBUTING.md)
- [Issues](https://github.com/unmasSk/acolyte/issues)

---

Choose the installation method that best fits your needs. For most users, we recommend the pip installation method.
