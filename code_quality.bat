@echo off
echo ========================================
echo    Code Quality Tools
echo ========================================
echo.

:menu
echo Select operation:
echo 1. Install development tools
echo 2. Format code (black)
echo 3. Check code quality (flake8)
echo 4. Check types (mypy)
echo 5. Run tests
echo 6. Clean temporary files
echo 7. Run all checks
echo 8. Run cleanup script
echo 9. Exit
echo.

set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto format
if "%choice%"=="3" goto lint
if "%choice%"=="4" goto typecheck
if "%choice%"=="5" goto test
if "%choice%"=="6" goto clean
if "%choice%"=="7" goto all
if "%choice%"=="8" goto cleanup
if "%choice%"=="9" goto exit
goto menu

:install
echo Installing development tools...
.venv\Scripts\python.exe -m pip install black mypy flake8 pytest
echo Done!
pause
goto menu

:format
echo Formatting code with black...
.venv\Scripts\python.exe -m black --line-length=88 --target-version=py38 --exclude="reverie|backup_environment_configs|models|\.venv" .
echo Done!
pause
goto menu

:lint
echo Checking code quality with flake8...
.venv\Scripts\python.exe -m flake8 --config=.flake8 .
echo Done!
pause
goto menu

:typecheck
echo Checking types with mypy...
.venv\Scripts\python.exe -m mypy --config-file=pyproject.toml ai_service
.venv\Scripts\python.exe -m mypy --config-file=pyproject.toml agents
.venv\Scripts\python.exe -m mypy --config-file=pyproject.toml performance_optimizer.py
.venv\Scripts\python.exe -m mypy --config-file=pyproject.toml memory_optimizer.py
.venv\Scripts\python.exe -m mypy --config-file=pyproject.toml network_optimizer.py
.venv\Scripts\python.exe -m mypy --config-file=pyproject.toml demo_performance_suite.py
echo Done!
pause
goto menu

:test
echo Running tests...
.venv\Scripts\python.exe simple_test.py
.venv\Scripts\python.exe integration_test.py
echo Done!
pause
goto menu

:clean
echo Cleaning temporary files...
for /r %%i in (*.pyc) do del "%%i" 2>nul
for /r %%i in (__pycache__) do rmdir /s /q "%%i" 2>nul
for /r %%i in (*.egg-info) do rmdir /s /q "%%i" 2>nul
for /r %%i in (.pytest_cache) do rmdir /s /q "%%i" 2>nul
for /r %%i in (.mypy_cache) do rmdir /s /q "%%i" 2>nul
del .coverage 2>nul
echo Done!
pause
goto menu

:all
echo Running all quality checks...
echo.
echo 1/4 Formatting code...
.venv\Scripts\python.exe -m black --line-length=88 --target-version=py38 --exclude="reverie|backup_environment_configs|models|\.venv" .
echo.
echo 2/4 Checking code quality...
.venv\Scripts\python.exe -m flake8 --config=.flake8 .
echo.
echo 3/4 Checking types...
.venv\Scripts\python.exe -m mypy --config-file=pyproject.toml ai_service
.venv\Scripts\python.exe -m mypy --config-file=pyproject.toml agents
echo.
echo 4/4 Running tests...
.venv\Scripts\python.exe simple_test.py
echo.
echo All checks completed!
pause
goto menu

:cleanup
echo Running code cleanup script...
.venv\Scripts\python.exe cleanup_code.py
echo Done!
pause
goto menu

:exit
echo Goodbye!
exit /b 0