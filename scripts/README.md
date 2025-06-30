# 🚀 ACOLYTE Installation Scripts

This directory contains the installation and utility scripts for ACOLYTE.

## 📁 Directory Structure

```
scripts/
├── install.sh          # Linux/Mac global installer
├── install.bat         # Windows global installer
├── dev/               # Development scripts
│   └── dev.sh         # Development utilities
└── install/           # Installation modules
    ├── init.py        # Project initialization
    ├── install.py     # Project configuration
    └── common/        # Shared utilities
```

## 🔧 Installation Scripts

### `install.sh` (Linux/Mac)

Global installation script that:
1. Checks system requirements (Python 3.11+, Git, Docker)
2. Clones ACOLYTE from GitHub (or copies from local in dev mode)
3. Installs Python dependencies with Poetry
4. **Copies the executable from `bin/acolyte`** to `~/.local/bin/`
5. Updates PATH in shell configuration

**Usage:**
```bash
# Production install from GitHub
curl -sSL https://raw.githubusercontent.com/unmasSk/acolyte/main/scripts/install.sh | bash

# Development install from local directory
./scripts/install.sh --dev /path/to/acolyte
```

### `install.bat` (Windows)

Global installation script that:
1. Checks system requirements (Python 3.11+, Git, Docker)
2. Clones ACOLYTE from GitHub (or copies from local in dev mode)
3. Installs Python dependencies with Poetry
4. **Verifies the executable exists in `bin/acolyte.bat`**
5. Updates PATH for the user

**Usage:**
```batch
REM Production install from GitHub
powershell -Command "Invoke-WebRequest -Uri https://raw.githubusercontent.com/unmasSk/acolyte/main/scripts/install.bat -OutFile install.bat"
install.bat

REM Development install from local directory
scripts\install.bat --dev C:\path\to\acolyte
```

## 📝 Important Notes

### Executable Management

The installation scripts **DO NOT** create executables anymore. Instead:

1. **Master executables** are maintained in `/bin/`:
   - `/bin/acolyte` - Linux/Mac executable with full validation
   - `/bin/acolyte.bat` - Windows executable with full validation

2. **Installation copies** these files:
   - Linux/Mac: Copies to `~/.local/bin/acolyte` and makes it executable
   - Windows: Verifies the file exists (already copied during xcopy)

3. **Benefits**:
   - Single source of truth for executable logic
   - Full Python version checking
   - Development mode support
   - Better error messages

### Development Mode

Both installers support `--dev` flag for local development:

```bash
# Linux/Mac
./scripts/install.sh --dev /path/to/acolyte/source

# Windows
scripts\install.bat --dev C:\path\to\acolyte\source
```

This copies from a local directory instead of cloning from GitHub.

## 🔄 Update Process

To update the executables:

1. **Modify the master files** in `/bin/`
2. **Test locally** using development mode
3. **Commit changes** to the repository
4. Users can update by re-running the installation script

## 🐛 Troubleshooting

### "Executable not found" during installation

This means the `/bin/acolyte` or `/bin/acolyte.bat` file is missing from the source:
- Check that the files exist in the source directory
- Ensure they weren't excluded during copying
- Verify the repository is complete

### PATH not updated

- **Linux/Mac**: Source your shell config or restart terminal
- **Windows**: Open a new terminal window (PATH updates require new session)

### Python version errors

The executables now check for Python 3.11+. If you see version errors:
- Install Python 3.11 or newer
- Ensure it's in your PATH
- On Windows, try both `python` and `python3` commands

## 🔨 Maintenance

When updating installation scripts:

1. **Never recreate executable logic** - use the files from `/bin/`
2. **Test both production and development modes**
3. **Ensure backward compatibility** with existing installations
4. **Update this README** if behavior changes

---

For more information about the CLI itself, see `/bin/README.md`
