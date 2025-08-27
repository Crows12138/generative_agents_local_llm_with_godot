@echo off
echo ============================================================
echo COZY BAR GAME SERVER
echo ============================================================
echo.
echo Starting LLM server for Cozy Bar game...
echo Using fast 4B model for sub-second responses
echo.

REM Check if server is already running
netstat -an | findstr :9997 >nul
if %errorlevel%==0 (
    echo [!] Server already running on port 9997
    echo.
    echo You can now start the Godot game!
    pause
    exit /b 0
)

REM Start the cozy bar server
echo [*] Loading Qwen3-4B model...
echo     This may take a moment on first run...
echo.

.venv\Scripts\python.exe start_cozy_bar_server.py 9997

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start server
    echo Please check:
    echo   1. Python virtual environment exists in .venv
    echo   2. Model file exists: models\llms\Qwen3-4B-Instruct-2507-Q4_0.gguf
    echo   3. Required packages are installed
)

pause