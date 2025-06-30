@echo off
REM ACOLYTE executable wrapper for Windows
REM 
REM This file is the master executable for Windows systems.
REM It gets copied to %USERPROFILE%\.acolyte\bin\acolyte.bat during installation by install.bat
REM 
REM DO NOT MODIFY the copy in .acolyte\bin - modify this source file instead
REM 
REM Features:
REM - Python version checking (3.11+)
REM - Supports both 'python' and 'python3' commands
REM - Development mode support via ACOLYTE_DEV
REM - Proper error messages and installation guidance
REM - Tries installed entry point first, falls back to source

setlocal enabledelayedexpansion

REM Determine ACOLYTE home directory
if defined ACOLYTE_DEV (
    REM Development mode
    set "ACOLYTE_HOME=%ACOLYTE_DEV%"
    set "DEVELOPMENT_MODE=1"
) else (
    REM Production mode
    if not defined ACOLYTE_HOME (
        set "ACOLYTE_HOME=%USERPROFILE%\.acolyte"
    )
    set "DEVELOPMENT_MODE=0"
)

REM Check if ACOLYTE is installed
if not exist "%ACOLYTE_HOME%" (
    echo Error: ACOLYTE not found at %ACOLYTE_HOME%
    echo.
    echo Please run the installation script first:
    echo   Download and run install.bat from:
    echo   https://github.com/unmasSk/acolyte/releases
    echo.
    echo Or set ACOLYTE_DEV for development mode:
    echo   set ACOLYTE_DEV=C:\path\to\acolyte\project
    exit /b 1
)

REM Check if Python is available
where python >nul 2>nul
if errorlevel 1 (
    where python3 >nul 2>nul
    if errorlevel 1 (
        echo Error: Python is not installed or not in PATH.
        echo Please install Python 3.11 or newer from python.org
        echo.
        echo Make sure to check "Add Python to PATH" during installation.
        exit /b 1
    ) else (
        set "PYTHON_CMD=python3"
    )
) else (
    set "PYTHON_CMD=python"
)

REM Check Python version
for /f "tokens=*" %%i in ('%PYTHON_CMD% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i

REM Simple version check (requires at least 3.11)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    if %%a LSS 3 (
        goto :version_error
    ) else if %%a EQU 3 (
        if %%b LSS 11 goto :version_error
    )
)

REM Try to use installed entry point first (if ACOLYTE was installed via pip/poetry)
if "%DEVELOPMENT_MODE%"=="0" (
    where acolyte >nul 2>nul
    if not errorlevel 1 (
        REM Check if it's our acolyte (not this batch file)
        for /f "tokens=*" %%i in ('where acolyte') do (
            if /i not "%%i"=="%~f0" (
                REM Use the installed version
                acolyte %*
                exit /b !errorlevel!
            )
        )
    )
)

REM Otherwise, run from source
set "PYTHONPATH=%ACOLYTE_HOME%\src;%PYTHONPATH%"

REM Check if CLI exists
if not exist "%ACOLYTE_HOME%\src\acolyte\cli.py" (
    echo Error: ACOLYTE CLI not found at %ACOLYTE_HOME%\src\acolyte\cli.py
    echo Installation may be corrupted. Please reinstall.
    exit /b 1
)

REM Run ACOLYTE CLI
%PYTHON_CMD% "%ACOLYTE_HOME%\src\acolyte\cli.py" %*
exit /b %errorlevel%

:version_error
echo Error: Python %PYTHON_VERSION% found, but 3.11 or newer is required.
echo Please download Python 3.11+ from https://python.org
exit /b 1
