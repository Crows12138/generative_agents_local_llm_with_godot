@echo off
echo ========================================
echo    Godot + LLM Integration Quick Start
echo ========================================
echo.

echo Step 1: Starting LLM Server...
start /min server_client\start_llm_server.bat
echo Server starting in background...

echo.
echo Step 2: Waiting for server to load (20 seconds)...
timeout /t 20 /nobreak >nul

echo.
echo Step 3: Instructions
echo ----------------------------------------
echo 1. Open Godot project: godot\live-with-ai\
echo 2. Open scene: scenes\cozy_bar.tscn
echo 3. Press F6 to run
echo 4. Click on NPCs (Bob, Alice, Sam) to interact!
echo.
echo Server is running on port 9999
echo.
pause