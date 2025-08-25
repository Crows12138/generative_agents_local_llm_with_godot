@echo off
echo Starting LLM Server (No Cache)...
echo ================================

REM Check if server is already running
netstat -an | findstr :9999 >nul
if %errorlevel%==0 (
    echo Server already running on port 9999
    pause
    exit /b 0
)

REM Start the server
echo Loading model, please wait...
.venv\Scripts\python.exe llm_server_nocache.py

pause