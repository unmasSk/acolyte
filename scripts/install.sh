#!/bin/bash
#
# ACOLYTE Global Installation Script
# Installs ACOLYTE system-wide for the current user
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
INSTALL_DIR="$HOME/.acolyte"
BIN_DIR="$HOME/.local/bin"
REPO_URL="https://github.com/unmasSk/acolyte.git"

# Logo
print_logo() {
    echo -e "${CYAN}"
    cat << 'EOF'
    ▄▄▄       ▄████▄   ▒█████   ██▓    ▓██   ██▓▄▄▄█████▓▓█████
   ▒████▄    ▒██▀ ▀█  ▒██▒  ██▒▓██▒     ▒██  ██▒▓  ██▒ ▓▒▓█   ▀
   ▒██  ▀█▄  ▒▓█    ▄ ▒██░  ██▒▒██░      ▒██ ██░▒ ▓██░ ▒░▒███
   ░██▄▄▄▄██ ▒▓▓▄ ▄██▒▒██   ██░▒██░      ░ ▐██▓░░ ▓██▓ ░ ▒▓█  ▄
    ▓█   ▓██▒▒ ▓███▀ ░░ ████▓▒░░██████▒  ░ ██▒▓░  ▒██▒ ░ ░▒████▒
EOF
    echo -e "${NC}"
    echo -e "${BOLD}ACOLYTE Global Installer${NC}"
    echo
}

# Helper functions
print_step() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check requirements
check_requirements() {
    print_step "Checking requirements..."
    
    local missing=()
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing+=("python3")
    else
        python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if (( $(echo "$python_version < 3.11" | bc -l) )); then
            print_error "Python 3.11+ required (found $python_version)"
            exit 1
        fi
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        missing+=("git")
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    fi
    
    # Check pip
    if ! python3 -m pip --version &> /dev/null; then
        missing+=("python3-pip")
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing requirements: ${missing[*]}"
        echo "Please install them first:"
        echo "  sudo apt update && sudo apt install ${missing[*]}"
        exit 1
    fi
    
    print_success "All requirements met"
}

# Install from source
install_from_source() {
    local source_dir="$1"
    
    print_step "Installing from source directory: $source_dir"
    
    # Create install directory
    if [ -d "$INSTALL_DIR" ]; then
        print_warning "Installation directory already exists: $INSTALL_DIR"
        read -p "Remove and reinstall? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$INSTALL_DIR"
        else
            print_error "Installation cancelled"
            exit 1
        fi
    fi
    
    # Copy source
    print_step "Copying files..."
    cp -r "$source_dir" "$INSTALL_DIR"
    
    # Install Python dependencies
    print_step "Installing Python dependencies..."
    cd "$INSTALL_DIR"
    
    # Install PyYAML globally for git hooks
    print_step "Installing PyYAML for git hooks..."
    python3 -m pip install --user pyyaml requests
    
    # Install Poetry if not present
    if ! command -v poetry &> /dev/null; then
        print_step "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Install project dependencies
    poetry install --only main
    
    print_success "Dependencies installed"
}

# Install from git
install_from_git() {
    print_step "Cloning from git repository..."
    
    # Create temp directory
    temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    # Clone repository
    git clone "$REPO_URL" "$temp_dir/acolyte"
    
    # Install from cloned source
    install_from_source "$temp_dir/acolyte"
}

# Create executable
create_executable() {
    print_step "Creating executable..."
    
    # Create bin directory
    mkdir -p "$BIN_DIR"
    
    # Create acolyte executable
    cat > "$BIN_DIR/acolyte" << 'EOF'
#!/bin/bash
# ACOLYTE executable wrapper

ACOLYTE_HOME="$HOME/.acolyte"
export PYTHONPATH="$ACOLYTE_HOME/src:$PYTHONPATH"

# Check if we're in development mode
if [ -n "$ACOLYTE_DEV" ]; then
    ACOLYTE_HOME="$ACOLYTE_DEV"
fi

# Run ACOLYTE CLI
exec python3 "$ACOLYTE_HOME/src/acolyte/cli.py" "$@"
EOF
    
    chmod +x "$BIN_DIR/acolyte"
    
    print_success "Executable created at $BIN_DIR/acolyte"
}

# Update PATH
update_path() {
    print_step "Updating PATH..."
    
    # Detect shell
    shell_rc=""
    if [ -n "$BASH_VERSION" ]; then
        shell_rc="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        shell_rc="$HOME/.zshrc"
    else
        shell_rc="$HOME/.profile"
    fi
    
    # Check if PATH already contains bin directory
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        echo "" >> "$shell_rc"
        echo "# ACOLYTE" >> "$shell_rc"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$shell_rc"
        
        print_success "PATH updated in $shell_rc"
        print_warning "Run 'source $shell_rc' or restart your terminal"
    else
        print_success "PATH already configured"
    fi
}

# Main installation
main() {
    print_logo
    
    # Parse arguments
    if [ "$1" = "--dev" ] && [ -n "$2" ]; then
        # Development install from local directory
        source_dir="$2"
        if [ ! -d "$source_dir" ]; then
            print_error "Source directory not found: $source_dir"
            exit 1
        fi
        mode="development"
    else
        # Production install from git
        mode="production"
    fi
    
    echo -e "${BOLD}Installation mode:${NC} $mode"
    echo
    
    # Check requirements
    check_requirements
    
    # Install
    if [ "$mode" = "development" ]; then
        install_from_source "$source_dir"
    else
        install_from_git
    fi
    
    # Create executable
    create_executable
    
    # Update PATH
    update_path
    
    # Final message
    echo
    print_success "${BOLD}ACOLYTE installed successfully!${NC}"
    echo
    echo "Next steps:"
    echo "  1. Reload your shell: source ~/.bashrc"
    echo "  2. Go to any project: cd /path/to/project"
    echo "  3. Initialize ACOLYTE: acolyte init"
    echo "  4. Install services: acolyte install"
    echo "  5. Start ACOLYTE: acolyte start"
    echo
    echo "Documentation: https://github.com/unmasSk/acolyte"
}

# Run main
main "$@"
