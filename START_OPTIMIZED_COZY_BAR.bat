@echo off
echo =======================================
echo OPTIMIZED COZY BAR SERVER LAUNCHER
echo =======================================
echo.
echo Starting optimized LLM server...
echo Using: Qwen3-1.7B with all optimizations
echo.

REM Use virtual environment Python
call .venv\Scripts\activate.bat

REM Start the optimized server
python server_client\optimized_cozy_bar_server.py 9999

pause