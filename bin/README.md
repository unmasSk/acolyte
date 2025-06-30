# 🤖 ACOLYTE CLI Reference

## Overview

The ACOLYTE Command Line Interface (CLI) is your primary tool for managing ACOLYTE projects. It handles initialization, installation, service management, and monitoring.

## Installation

The CLI is automatically installed when you install ACOLYTE:

```bash
# Linux/Mac
curl -sSL https://raw.githubusercontent.com/unmasSk/acolyte/main/scripts/install.sh | bash

# Windows
Invoke-WebRequest -Uri https://raw.githubusercontent.com/unmasSk/acolyte/main/scripts/install.bat -OutFile install.bat
.\install.bat
```

After installation, the `acolyte` command will be available globally.

## Commands

### 🚀 `acolyte init`

Initialize ACOLYTE in a project directory.

```bash
acolyte init [OPTIONS]
```

**Options:**
- `--path PATH` - Project path (default: current directory)
- `--name NAME` - Project name (default: directory name)
- `--force` - Force re-initialization

**Example:**
```bash
cd /path/to/your/project
acolyte init --name "My Awesome Project"
```

**What it does:**
1. Validates the directory is a valid project (has .git, package.json, etc.)
2. Generates a unique project ID
3. Creates project configuration
4. Sets up `.acolyte.project` file
5. Prepares directory structure

---

### 📦 `acolyte install`

Install and configure ACOLYTE services for the project.

```bash
acolyte install [OPTIONS]
```

**Options:**
- `--path PATH` - Project path (default: current directory)

**Example:**
```bash
acolyte install
```

**What it does:**
1. Interactive configuration wizard:
   - Hardware detection (CPU, RAM, GPU)
   - Model selection based on your hardware
   - Port configuration
   - Language detection and linter setup
2. Generates Docker infrastructure
3. Creates Ollama Modelfile
4. Initializes database structure

**Note:** Run this after `acolyte init`.

---

### ▶️ `acolyte start`

Start all ACOLYTE services.

```bash
acolyte start [OPTIONS]
```

**Options:**
- `--path PATH` - Project path (default: current directory)

**Example:**
```bash
acolyte start
```

**What it does:**
1. Starts Docker containers:
   - Weaviate (vector database)
   - Ollama (AI model server)
   - Backend (FastAPI)
2. Shows service URLs
3. Waits for services to be healthy

**Service URLs after start:**
- Weaviate: `http://localhost:{weaviate_port}`
- Ollama: `http://localhost:{ollama_port}`
- API: `http://localhost:{backend_port}`

---

### ⏹️ `acolyte stop`

Stop all ACOLYTE services.

```bash
acolyte stop [OPTIONS]
```

**Options:**
- `--path PATH` - Project path (default: current directory)

**Example:**
```bash
acolyte stop
```

---

### 📊 `acolyte status`

Check the status of ACOLYTE services and project.

```bash
acolyte status [OPTIONS]
```

**Options:**
- `--path PATH` - Project path (default: current directory)

**Example:**
```bash
acolyte status
```

**Shows:**
- Project information (name, ID, path)
- Service status (Running/Stopped)
- Database size
- Configuration location

---

### 🔍 `acolyte index`

Index project files for AI understanding.

```bash
acolyte index [OPTIONS]
```

**Options:**
- `--path PATH` - Project path (default: current directory)
- `--full` - Force full reindexing (ignore cache)

**Example:**
```bash
# Initial indexing
acolyte index

# Force complete reindex
acolyte index --full
```

**What it does:**
1. Scans all project files
2. Respects .gitignore and .acolyteignore
3. Creates code chunks with AST parsing
4. Generates embeddings
5. Stores in Weaviate vector database

**Important:** You MUST run this after starting services for ACOLYTE to work!

---

### 📋 `acolyte logs`

View logs from ACOLYTE services.

```bash
acolyte logs [OPTIONS]
```

**Options:**
- `--path PATH` - Project path (default: current directory)
- `-f, --follow` - Follow log output (like tail -f)
- `-n, --lines N` - Number of lines to show (default: 100)
- `-s, --service SERVICE` - Service to show logs for (backend/weaviate/ollama/all)
- `--file` - Show debug.log file instead of Docker logs
- `-g, --grep TEXT` - Filter logs containing text
- `--level LEVEL` - Filter by log level (DEBUG/INFO/WARNING/ERROR) - only with --file

**Examples:**
```bash
# View all service logs
acolyte logs

# Follow backend logs in real-time
acolyte logs -f -s backend

# Search for errors in last 500 lines
acolyte logs -n 500 -g "error"

# View only ERROR level logs from debug.log
acolyte logs --file --level ERROR

# Follow debug.log filtering for Git events
acolyte logs --file -f -g "GitService"
```

**Log Locations:**
- Docker logs: Real-time container output
- debug.log: `~/.acolyte/projects/{project_id}/data/debug.log`

---

### 📂 `acolyte projects`

List all ACOLYTE projects on this machine.

```bash
acolyte projects
```

**Example output:**
```
📂 ACOLYTE Projects

• My Awesome Project (a1b2c3d4...)
  Path: /home/user/projects/awesome

• Another Project (e5f6g7h8...)
  Path: /home/user/work/another
```

---

### 🧹 `acolyte clean`

Clean ACOLYTE cache and temporary files.

```bash
acolyte clean [OPTIONS]
```

**Options:**
- `--path PATH` - Project path (default: current directory)

**Example:**
```bash
acolyte clean
```

**What it cleans:**
- Log files
- Temporary cache (coming soon)
- Build artifacts (coming soon)

**Note:** This will ask for confirmation before cleaning.

---

## Typical Workflow

1. **Initialize a new project:**
   ```bash
   cd /path/to/your/project
   acolyte init
   ```

2. **Install and configure ACOLYTE:**
   ```bash
   acolyte install
   ```

3. **Start services:**
   ```bash
   acolyte start
   ```

4. **Check everything is running:**
   ```bash
   acolyte status
   ```

5. **Index your code (CRITICAL!):**
   ```bash
   acolyte index
   ```

6. **Monitor logs while working:**
   ```bash
   # In another terminal
   acolyte logs -f -s backend
   ```

7. **Stop when done:**
   ```bash
   acolyte stop
   ```

## File Structure

After initialization, ACOLYTE creates:

```
Your Project/
├── .acolyte.project          # Project identifier (12 bytes)
└── ... (your code)

~/.acolyte/
├── projects/
│   └── {project-id}/         # Unique per project
│       ├── .acolyte          # Configuration
│       ├── data/
│       │   ├── acolyte.db    # SQLite database
│       │   ├── debug.log     # Application logs
│       │   └── dreams/       # Analysis results
│       └── infra/
│           ├── docker-compose.yml
│           └── Modelfile
└── src/                      # ACOLYTE source code
```

## Environment Variables

- `ACOLYTE_DEBUG=true` - Enable debug output
- `ACOLYTE_DEV=/path/to/dev` - Development mode
- `ACOLYTE_PORT=8001` - Override backend port
- `ACOLYTE_LOG_LEVEL=DEBUG` - Set log level

## Troubleshooting

### "Project not initialized"
Run `acolyte init` first in your project directory.

### "Services not installed"
Run `acolyte install` after initialization.

### "Backend not running"
1. Check Docker is running: `docker ps`
2. Start services: `acolyte start`
3. Check logs: `acolyte logs -s backend`

### "No logs found"
Services may not have started yet. Run `acolyte status` to check.

### Port conflicts
Edit `~/.acolyte/projects/{id}/.acolyte` and change port numbers.

## Tips

1. **Always run `acolyte index` after starting services** - ACOLYTE needs to analyze your code first!

2. **Use `acolyte logs -f` during development** to monitor what ACOLYTE is doing.

3. **The `.acolyte.project` file is tiny (12 bytes)** - safe to commit to git.

4. **Each project gets unique ports** to avoid conflicts when running multiple projects.

5. **Use `--file` option with logs** to see the persistent debug.log instead of Docker output.

## Version

Current version: 1.0.0

Check version:
```bash
acolyte --version
```

## Help

Get help for any command:
```bash
acolyte --help
acolyte init --help
acolyte logs --help
```

---

For more information, visit the [ACOLYTE documentation](https://github.com/unmasSk/acolyte/docs).
