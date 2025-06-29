@echo off
REM ACOLYTE executable wrapper for Windows
REM This file is copied to %USERPROFILE%\.acolyte\bin\acolyte.bat during installation

setlocal

REM Set ACOLYTE home directory
if not defined ACOLYTE_HOME set ACOLYTE_HOME=%USERPROFILE%\.acolyte

REM Check if in development mode
if defined ACOLYTE_DEV set ACOLYTE_HOME=%ACOLYTE_DEV%

REM Set Python path
set PYTHONPATH=%ACOLYTE_HOME%\src;%PYTHONPATH%

REM Check if ACOLYTE is installed
if not exist "%ACOLYTE_HOME%" (
    echo Error: ACOLYTE not found at %ACOLYTE_HOME%
    echo Please run the installation script first:
    echo   Download and run install.bat
    exit /b 1
)

REM Run ACOLYTE CLI
python "%ACOLYTE_HOME%\src\acolyte\cli.py" %*
