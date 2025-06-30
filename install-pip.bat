@echo off
REM ACOLYTE Simplified Installation Script using pip for Windows
REM This script installs ACOLYTE as a proper Python package

setlocal enabledelayedexpansion

REM Configuration
set REPO_URL=https://github.com/unmasSk/acolyte.git

REM Colors (Windows 10+)
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set CYAN=[96m
set NC=[0m
set BOLD=[1m

REM Logo
echo %CYAN%
echo     ▄▄▄       ▄████▄   ▒█████   ██▓    ▓██   ██▓▄▄▄█████▓▓█████
echo    ▒████▄    ▒██▀ ▀█  ▒██▒  ██▒▓██▒     ▒██  ██▒▓  ██▒ ▓▒▓█   ▀
echo    ▒██  ▀█▄  ▒▓█    ▄ ▒██░  ██▒▒██░      ▒██ ██░▒ ▓██░ ▒░▒███
echo    ░██▄▄▄▄██ ▒▓▓▄ ▄██▒▒██   ██░▒██░      ░ ▐██▓░░ ▓██▓ ░ ▒▓█  ▄
echo     ▓█   ▓██▒▒ ▓███▀ ░░ ████▓▒░░██████▒  ░ ██▒▓░  ▒██▒ ░ ░▒████▒
echo %NC%
echo %BOLD%ACOLYTE Pip Installer for Windows%NC%
echo.

REM Parse arguments
set INSTALL_MODE=production
set SOURCE_PATH=

:parse_args
if "%1"=="" goto :check_python
if "%1"=="--dev" (
    set INSTALL_MODE=development
    if "%2"=="" (
        set SOURCE_PATH=.
    ) else (
        set SOURCE_PATH=%2
        shift
    )
    shift
    goto :parse_args
)
if "%1"=="--local" (
    set INSTALL_MODE=local
    if "%2"=="" (
        set SOURCE_PATH=.
    ) else (
        set SOURCE_PATH=%2
        shift
    )
    shift
    goto :parse_args
)
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help

echo %RED%✗%NC% Unknown option: %1
exit /b 1

:show_help
echo Usage: %0 [OPTIONS]
echo.
echo Options:
echo   --dev [PATH]    Install in development mode (editable)
echo   --local [PATH]  Install from local directory
echo   --help          Show this help message
echo.
echo Examples:
echo   %0                        # Install from GitHub
echo   %0 --dev                  # Install current directory in dev mode
echo   %0 --dev C:\path\to\acolyte  # Install specific path in dev mode
echo   %0 --local .\acolyte      # Install from local directory
exit /b 0

:check_python
echo %BLUE%[%TIME%]%NC% Checking Python version...

REM Check if Python is available
set PYTHON_CMD=
where python >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :verify_version
)

where python3 >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3
    goto :verify_version
)

echo %RED%✗%NC% Python not found. Please install Python 3.11 or newer from python.org
exit /b 1

:verify_version
REM Check Python version
%PYTHON_CMD% -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" > temp_version.txt 2>nul
if errorlevel 1 (
    echo %RED%✗%NC% Failed to check Python version
    exit /b 1
)
set /p PYTHON_VERSION=<temp_version.txt
del temp_version.txt

REM Verify minimum version
%PYTHON_CMD% -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo %RED%✗%NC% Python 3.11+ required ^(found %PYTHON_VERSION%^)
    exit /b 1
)

echo %GREEN%✓%NC% Python %PYTHON_VERSION% found

REM Check other requirements
echo %BLUE%[%TIME%]%NC% Checking requirements...

REM Check Git
git --version >nul 2>&1
if errorlevel 1 (
    echo %RED%✗%NC% Git not found. Please install Git from git-scm.com
    exit /b 1
)

REM Check Docker (warning only)
docker --version >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%⚠%NC% Docker not found. You'll need it to run ACOLYTE services.
)

REM Check pip
%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo %RED%✗%NC% pip not found. Please ensure pip is installed.
    exit /b 1
)

echo %GREEN%✓%NC% All requirements met

echo.
echo %BOLD%Installation mode:%NC% %INSTALL_MODE%
if not "%SOURCE_PATH%"=="" (
    echo %BOLD%Source path:%NC% %SOURCE_PATH%
)
echo.

REM Install ACOLYTE
echo %BLUE%[%TIME%]%NC% Installing ACOLYTE via pip...

if "%INSTALL_MODE%"=="development" (
    REM Development install from local directory
    if not exist "%SOURCE_PATH%" (
        echo %RED%✗%NC% Source directory not found: %SOURCE_PATH%
        exit /b 1
    )
    
    echo Installing from local directory in editable mode...
    %PYTHON_CMD% -m pip install -e "%SOURCE_PATH%"
) else if "%INSTALL_MODE%"=="local" (
    REM Install from local directory (non-editable)
    if not exist "%SOURCE_PATH%" (
        echo %RED%✗%NC% Source directory not found: %SOURCE_PATH%
        exit /b 1
    )
    
    echo Installing from local directory...
    %PYTHON_CMD% -m pip install "%SOURCE_PATH%"
) else (
    REM Production install from GitHub
    echo Installing from GitHub...
    %PYTHON_CMD% -m pip install "git+%REPO_URL%"
)

if errorlevel 1 (
    echo %RED%✗%NC% Installation failed
    exit /b 1
)

REM Verify installation
where acolyte >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%⚠%NC% acolyte command not found in PATH
    echo You may need to add Python Scripts to your PATH
    
    REM Try to find Python scripts directory
    for /f "tokens=*" %%i in ('%PYTHON_CMD% -m site --user-base') do set PYTHON_USER_BASE=%%i
    if exist "%PYTHON_USER_BASE%\Scripts" (
        echo Add this to your PATH: %PYTHON_USER_BASE%\Scripts
        
        REM Offer to add to PATH
        set /p ADD_PATH="Add to PATH now? [Y/n] "
        if /i "!ADD_PATH!"=="" set ADD_PATH=Y
        if /i "!ADD_PATH!"=="Y" (
            setx PATH "%PATH%;%PYTHON_USER_BASE%\Scripts"
            echo %GREEN%✓%NC% PATH updated. Please restart your terminal.
        )
    )
) else (
    echo %GREEN%✓%NC% ACOLYTE command found in PATH
)

REM Final message
echo.
echo %GREEN%✓%NC% %BOLD%ACOLYTE installed successfully!%NC%
echo.
echo Next steps:
echo   1. Open a new terminal window
echo   2. Verify installation: acolyte --version
echo   3. Go to any project: cd C:\path\to\project
echo   4. Initialize ACOLYTE: acolyte init
echo   5. Install services: acolyte install
echo   6. Start ACOLYTE: acolyte start
echo.
echo Documentation: https://github.com/unmasSk/acolyte
echo.

pause
