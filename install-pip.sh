#!/bin/bash
#
# ACOLYTE Simplified Installation Script using pip
# This script installs ACOLYTE as a proper Python package
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
    echo -e "${BOLD}ACOLYTE Pip Installer${NC}"
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

# Check Python version
check_python() {
    print_step "Checking Python version..."
    
    # Try python3 first, then python
    PYTHON_CMD=""
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.11 or newer."
        exit 1
    fi
    
    # Check version
    python_version=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
        print_error "Python 3.11+ required (found $python_version)"
        exit 1
    fi
    
    print_success "Python $python_version found"
    echo "PYTHON_CMD=$PYTHON_CMD"  # Export for use in other functions
}

# Check other requirements
check_requirements() {
    print_step "Checking requirements..."
    
    local missing=()
    
    # Check Git
    if ! command -v git &> /dev/null; then
        missing+=("git")
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_warning "Docker not found. You'll need it to run ACOLYTE services."
    fi
    
    # Check pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        missing+=("pip")
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing requirements: ${missing[*]}"
        echo "Please install them first."
        exit 1
    fi
    
    print_success "All requirements met"
}

# Install ACOLYTE
install_acolyte() {
    local install_mode="$1"
    local source_path="$2"
    
    print_step "Installing ACOLYTE via pip..."
    
    if [ "$install_mode" = "development" ]; then
        # Development install from local directory
        if [ ! -d "$source_path" ]; then
            print_error "Source directory not found: $source_path"
            exit 1
        fi
        
        print_step "Installing from local directory in editable mode..."
        $PYTHON_CMD -m pip install -e "$source_path"
        
    elif [ "$install_mode" = "local" ]; then
        # Install from local directory (non-editable)
        if [ ! -d "$source_path" ]; then
            print_error "Source directory not found: $source_path"
            exit 1
        fi
        
        print_step "Installing from local directory..."
        $PYTHON_CMD -m pip install "$source_path"
        
    else
        # Production install from GitHub
        print_step "Installing from GitHub..."
        $PYTHON_CMD -m pip install "git+${REPO_URL}"
    fi
    
    # Verify installation
    if ! command -v acolyte &> /dev/null; then
        print_warning "acolyte command not found in PATH"
        echo "You may need to add Python scripts to your PATH:"
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    else
        print_success "ACOLYTE installed successfully!"
    fi
}

# Update PATH if needed
update_path() {
    print_step "Checking PATH..."
    
    # Check if ~/.local/bin is in PATH
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        print_warning "~/.local/bin is not in your PATH"
        
        # Detect shell
        shell_rc=""
        if [ -n "$BASH_VERSION" ]; then
            shell_rc="$HOME/.bashrc"
        elif [ -n "$ZSH_VERSION" ]; then
            shell_rc="$HOME/.zshrc"
        else
            shell_rc="$HOME/.profile"
        fi
        
        # Add to PATH
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
    install_mode="production"
    source_path=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                install_mode="development"
                source_path="${2:-.}"
                shift 2
                ;;
            --local)
                install_mode="local"
                source_path="${2:-.}"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --dev [PATH]    Install in development mode (editable)"
                echo "  --local [PATH]  Install from local directory"
                echo "  --help          Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                    # Install from GitHub"
                echo "  $0 --dev              # Install current directory in dev mode"
                echo "  $0 --dev /path/to/acolyte  # Install specific path in dev mode"
                echo "  $0 --local ./acolyte  # Install from local directory"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    echo -e "${BOLD}Installation mode:${NC} $install_mode"
    if [ -n "$source_path" ]; then
        echo -e "${BOLD}Source path:${NC} $source_path"
    fi
    echo
    
    # Check Python and requirements
    check_python
    check_requirements
    
    # Install ACOLYTE
    install_acolyte "$install_mode" "$source_path"
    
    # Update PATH
    update_path
    
    # Final message
    echo
    print_success "${BOLD}ACOLYTE installed successfully!${NC}"
    echo
    echo "Next steps:"
    echo "  1. Reload your shell: source ~/.bashrc"
    echo "  2. Verify installation: acolyte --version"
    echo "  3. Go to any project: cd /path/to/project"
    echo "  4. Initialize ACOLYTE: acolyte init"
    echo "  5. Install services: acolyte install"
    echo "  6. Start ACOLYTE: acolyte start"
    echo
    echo "Documentation: https://github.com/unmasSk/acolyte"
}

# Run main
main "$@"
