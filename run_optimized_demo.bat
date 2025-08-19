@echo off
echo ========================================
echo    Optimized Demo Launcher
echo ========================================
echo.

:menu
echo Select performance mode:
echo 1. Low Performance (2GB RAM, basic features)
echo 2. Medium Performance (4GB RAM, standard features) [Default]
echo 3. High Performance (6GB RAM, advanced features)
echo 4. Ultra Performance (8GB+ RAM, all features)
echo 5. Test Mode (2 minutes auto-test)
echo 6. No Optimization (debug mode)
echo.

set /p choice="Enter your choice (1-6) or press Enter for default: "

if "%choice%"=="" set choice=2

if "%choice%"=="1" (
    echo Starting LOW performance mode...
    .venv\Scripts\python.exe run_optimized_demo.py --performance low --start-ai --ignore-ai-failure
) else if "%choice%"=="2" (
    echo Starting MEDIUM performance mode...
    .venv\Scripts\python.exe run_optimized_demo.py --performance medium --start-ai --ignore-ai-failure
) else if "%choice%"=="3" (
    echo Starting HIGH performance mode...
    .venv\Scripts\python.exe run_optimized_demo.py --performance high --start-ai --start-godot
) else if "%choice%"=="4" (
    echo Starting ULTRA performance mode...
    .venv\Scripts\python.exe run_optimized_demo.py --performance ultra --start-ai --start-godot
) else if "%choice%"=="5" (
    echo Starting TEST mode...
    .venv\Scripts\python.exe run_optimized_demo.py --test-mode --duration 2 --performance medium --start-ai --ignore-ai-failure
) else if "%choice%"=="6" (
    echo Starting NO OPTIMIZATION mode...
    .venv\Scripts\python.exe run_optimized_demo.py --no-optimization --start-ai --ignore-ai-failure
) else (
    echo Invalid choice. Please try again.
    echo.
    goto menu
)

echo.
echo Demo completed. Press any key to exit...
pause >nul