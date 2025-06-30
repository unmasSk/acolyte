@echo off
REM ACOLYTE Global Installation Script for Windows
REM Installs ACOLYTE system-wide for the current user

setlocal enabledelayedexpansion

REM Configuration
set INSTALL_DIR=%USERPROFILE%\.acolyte
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
echo %BOLD%ACOLYTE Global Installer for Windows%NC%
echo.

REM Check Python
echo %BLUE%[%TIME%]%NC% Checking requirements...
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%✗%NC% Python not found. Please install Python 3.11+ from python.org
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Check Git
git --version >nul 2>&1
if errorlevel 1 (
    echo %RED%✗%NC% Git not found. Please install Git from git-scm.com
    exit /b 1
)

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo %RED%✗%NC% Docker not found. Please install Docker Desktop
    exit /b 1
)

echo %GREEN%✓%NC% All requirements met

REM Parse arguments
set MODE=production
if "%1"=="--dev" (
    if not "%2"=="" (
        set MODE=development
        set SOURCE_DIR=%2
    )
)

echo %BOLD%Installation mode:%NC% %MODE%
echo.

REM Check if already installed
if exist "%INSTALL_DIR%" (
    echo %YELLOW%⚠%NC% Installation directory already exists: %INSTALL_DIR%
    set /p REINSTALL="Remove and reinstall? [y/N] "
    if /i "!REINSTALL!"=="y" (
        echo Removing existing installation...
        rmdir /s /q "%INSTALL_DIR%"
    ) else (
        echo %RED%✗%NC% Installation cancelled
        exit /b 1
    )
)

REM Install based on mode
if "%MODE%"=="development" (
    REM Development install from local directory
    echo %BLUE%[%TIME%]%NC% Installing from source directory: %SOURCE_DIR%
    
    if not exist "%SOURCE_DIR%" (
        echo %RED%✗%NC% Source directory not found: %SOURCE_DIR%
        exit /b 1
    )
    
    echo Copying files...
    xcopy /E /I /Y "%SOURCE_DIR%" "%INSTALL_DIR%"
) else (
    REM Production install from git
    echo %BLUE%[%TIME%]%NC% Cloning from git repository...
    
    REM Create temp directory
    set TEMP_DIR=%TEMP%\acolyte_install_%RANDOM%
    mkdir "%TEMP_DIR%"
    
    REM Clone repository
    git clone %REPO_URL% "%TEMP_DIR%\acolyte"
    if errorlevel 1 (
        echo %RED%✗%NC% Failed to clone repository
        rmdir /s /q "%TEMP_DIR%"
        exit /b 1
    )
    
    REM Copy to install directory
    echo Copying files...
    xcopy /E /I /Y "%TEMP_DIR%\acolyte" "%INSTALL_DIR%"
    
    REM Clean up
    rmdir /s /q "%TEMP_DIR%"
)

REM Install Python dependencies
echo %BLUE%[%TIME%]%NC% Installing Python dependencies...
cd /d "%INSTALL_DIR%"

REM Install PyYAML globally for git hooks
echo Installing PyYAML for git hooks...
python -m pip install --user pyyaml requests

REM Check if Poetry is installed
poetry --version >nul 2>&1
if errorlevel 1 (
    echo Installing Poetry...
    curl -sSL https://install.python-poetry.org | python -
    REM Add Poetry to PATH for current session
    set PATH=%APPDATA%\Python\Scripts;%PATH%
)

REM Install project dependencies
poetry install --only main
if errorlevel 1 (
    echo %RED%✗%NC% Failed to install dependencies
    exit /b 1
)

echo %GREEN%✓%NC% Dependencies installed

REM Create batch file wrapper
echo %BLUE%[%TIME%]%NC% Creating executable...

REM Create acolyte.bat
echo @echo off > "%INSTALL_DIR%\bin\acolyte.bat"
echo setlocal >> "%INSTALL_DIR%\bin\acolyte.bat"
echo set ACOLYTE_HOME=%%USERPROFILE%%\.acolyte >> "%INSTALL_DIR%\bin\acolyte.bat"
echo set PYTHONPATH=%%ACOLYTE_HOME%%\src;%%PYTHONPATH%% >> "%INSTALL_DIR%\bin\acolyte.bat"
echo if defined ACOLYTE_DEV set ACOLYTE_HOME=%%ACOLYTE_DEV%% >> "%INSTALL_DIR%\bin\acolyte.bat"
echo python "%%ACOLYTE_HOME%%\src\acolyte\cli.py" %%* >> "%INSTALL_DIR%\bin\acolyte.bat"

echo %GREEN%✓%NC% Executable created

REM Update PATH
echo %BLUE%[%TIME%]%NC% Updating PATH...

REM Check if already in PATH
echo %PATH% | find /i "%INSTALL_DIR%\bin" >nul
if errorlevel 1 (
    REM Add to user PATH
    setx PATH "%PATH%;%INSTALL_DIR%\bin"
    echo %GREEN%✓%NC% PATH updated. Please restart your terminal.
) else (
    echo %GREEN%✓%NC% PATH already configured
)

REM Final message
echo.
echo %GREEN%✓%NC% %BOLD%ACOLYTE installed successfully!%NC%
echo.
echo Next steps:
echo   1. Open a new terminal window
echo   2. Go to any project: cd C:\path\to\project
echo   3. Initialize ACOLYTE: acolyte init
echo   4. Install services: acolyte install
echo   5. Start ACOLYTE: acolyte start
echo.
echo Documentation: https://github.com/unmasSk/acolyte
echo.

pause
