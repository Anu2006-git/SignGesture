@echo off
setlocal
title SignGesture AI - Two-Way Communication System
color 0B

echo.
echo  ========================================================================
echo    SignGesture AI - Real-Time Two-Way Sign Language System
echo  ========================================================================
echo.

cd /d "%~dp0"

set "PY_EXE="

:: 1. Check user AppData Python 3.11 installation
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :FOUND
)

:: 2. Check py launcher for Python 3.11
py -3.11 --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_EXE=py -3.11"
    goto :FOUND
)

:: 3. Check generic py launcher
py --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_EXE=py"
    goto :FOUND
)

:: 4. Check system python in PATH
python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_EXE=python"
    goto :FOUND
)

:: 5. Common installation directories
if exist "C:\Python311\python.exe" (
    set "PY_EXE=C:\Python311\python.exe"
    goto :FOUND
)
if exist "C:\Program Files\Python311\python.exe" (
    set "PY_EXE=C:\Program Files\Python311\python.exe"
    goto :FOUND
)

:NOT_FOUND
echo  [ERROR] Python was not found on your system!
echo  Please ensure Python 3.11 is installed.
echo.
pause
exit /b 1

:FOUND
echo  [OK] Found Python: %PY_EXE%
echo  [OK] Launching Two-Way Sign Language Dashboard...
echo.

%PY_EXE% "%~dp0realtime_detector.py" %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo  ========================================================================
    echo  [NOTICE] The application stopped with exit code %ERRORLEVEL%.
    echo  ========================================================================
    echo.
    pause
)
