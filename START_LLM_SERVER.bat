@echo off
echo ============================================================
echo STANDALONE LLM SERVER
echo ============================================================
echo.

REM Check if port 9999 is in use
netstat -an | findstr :9999 >nul
if %errorlevel%==0 (
    echo [!] Server already running on port 9999
    pause
    exit /b 0
)

echo Starting LLM Server (Standalone Version)...
echo.
echo [*] Loading Qwen3-4B model...
echo     This will take a few seconds...
echo.

REM Run the standalone server
.venv\Scripts\python.exe server_client\llm_server_standalone.py 9999

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server failed to start
    echo.
    echo Please check:
    echo   1. Virtual environment exists (.venv folder)
    echo   2. Model file exists: models\llms\Qwen3-4B-Instruct-2507-Q4_0.gguf
    echo   3. Python packages installed: pip install llama-cpp-python
)

pause