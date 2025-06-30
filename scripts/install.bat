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
REM Check Python version
python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" > temp_version.txt 2>nul
if errorlevel 1 (
    echo %RED%✗%NC% Failed to check Python version
    exit /b 1
)
set /p PYTHON_VERSION=<temp_version.txt
del temp_version.txt
echo Found Python %PYTHON_VERSION%

REM Verify minimum version (simplified check)
python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo %RED%✗%NC% Python 3.11+ required ^(found %PYTHON_VERSION%^)
    exit /b 1
)

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

REM Secure Poetry installation with checksum verification
SET "POETRY_INSTALLER_URL=https://install.python-poetry.org"
SET "POETRY_INSTALLER_FILE=%TEMP%\poetry-installer.py"
SET "POETRY_INSTALLER_HASH_URL=https://install.python-poetry.org/sha256sum.txt"
SET "POETRY_INSTALLER_HASH_FILE=%TEMP%\poetry-installer.sha256"

REM Download installer script
powershell -Command "Invoke-WebRequest -Uri %POETRY_INSTALLER_URL% -OutFile '%POETRY_INSTALLER_FILE%'"
REM Download official hash
powershell -Command "Invoke-WebRequest -Uri %POETRY_INSTALLER_HASH_URL% -OutFile '%POETRY_INSTALLER_HASH_FILE%'"

REM Extract expected hash for the installer file
FOR /F "tokens=1,2" %%A IN ('findstr /I "poetry-installer.py" "%POETRY_INSTALLER_HASH_FILE%"') DO SET "EXPECTED_HASH=%%A"

REM Compute actual hash
FOR /F %%H IN ('powershell -Command "Get-FileHash -Algorithm SHA256 '%POETRY_INSTALLER_FILE%' | Select-Object -ExpandProperty Hash"') DO SET "ACTUAL_HASH=%%H"

REM Compare hashes
IF /I NOT "%EXPECTED_HASH%"=="%ACTUAL_HASH%" (
    echo Error: Poetry installer checksum verification failed.
    del "%POETRY_INSTALLER_FILE%"
    exit /b 1
)

REM Run the installer
py "%POETRY_INSTALLER_FILE%"

REM Add Poetry to PATH (user scope)
setx PATH "%APPDATA%\Python\Scripts;%PATH%"

REM Verify Poetry installation
poetry --version

REM Install project dependencies
poetry install --only main
if errorlevel 1 (
    echo %RED%✗%NC% Failed to install dependencies
    exit /b 1
)

echo %GREEN%✓%NC% Dependencies installed

REM Install executable
echo %BLUE%[%TIME%]%NC% Installing executable...

REM Verify the executable exists
if not exist "%INSTALL_DIR%\bin\acolyte.bat" (
    echo %RED%✗%NC% Executable not found in %INSTALL_DIR%\bin\acolyte.bat
    echo Installation may be corrupted
    exit /b 1
)

REM The executable is already in place from the copy operation
echo %GREEN%✓%NC% Executable installed

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
