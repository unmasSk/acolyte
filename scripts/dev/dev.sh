#!/bin/bash
#
# ACOLYTE Development Helper Script
# Use this during development to test ACOLYTE locally
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# Set development mode
export ACOLYTE_DEV=$(pwd)

echo -e "${GREEN}ACOLYTE Development Mode${NC}"
echo "Using source from: $ACOLYTE_DEV"
echo

# Function to run acolyte commands
run_acolyte() {
    export PYTHONPATH="$ACOLYTE_DEV/src:$PYTHONPATH"
    python "$ACOLYTE_DEV/src/acolyte/cli.py" "$@"
}

# Parse commands
case "$1" in
    init|install|start|stop|status|index|clean|projects)
        run_acolyte "$@"
        ;;
    test)
        echo -e "${YELLOW}Running tests...${NC}"
        poetry run pytest tests/
        ;;
    lint)
        echo -e "${YELLOW}Running linters...${NC}"
        poetry run ruff check .
        poetry run black --check .
        poetry run mypy src/acolyte --strict
        ;;
    format)
        echo -e "${YELLOW}Formatting code...${NC}"
        poetry run black .
        poetry run ruff check --fix .
        ;;
    coverage)
        echo -e "${YELLOW}Running coverage...${NC}"
        poetry run pytest --cov=acolyte --cov-report=html tests/
        echo "Coverage report: htmlcov/index.html"
        ;;
    *)
        echo "ACOLYTE Development Helper"
        echo
        echo "Usage: ./dev.sh <command>"
        echo
        echo "Commands:"
        echo "  init      - Initialize project"
        echo "  install   - Install services"
        echo "  start     - Start services"
        echo "  stop      - Stop services"
        echo "  status    - Check status"
        echo "  index     - Index files"
        echo "  clean     - Clean cache"
        echo "  projects  - List projects"
        echo
        echo "Development commands:"
        echo "  test      - Run tests"
        echo "  lint      - Run linters"
        echo "  format    - Format code"
        echo "  coverage  - Run coverage report"
        ;;
esac
