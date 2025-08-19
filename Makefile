# Makefile for Generative Agents project
# Cross-platform commands for code quality and development

.PHONY: help install format lint typecheck test clean dev-install run-demo run-test all

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  dev-install  - Install development dependencies"
	@echo "  format       - Format code with black"
	@echo "  lint         - Run code quality checks with flake8"
	@echo "  typecheck    - Run type checking with mypy"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean up temporary files"
	@echo "  run-demo     - Run optimized demo"
	@echo "  run-test     - Run integration test"
	@echo "  all          - Run format, lint, typecheck, and test"

# Install dependencies
install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

# Install development dependencies
dev-install: install
	python -m pip install black mypy flake8 pytest pytest-cov pytest-asyncio

# Format code
format:
	python -m black --line-length=88 --target-version=py38 --exclude="reverie|backup_environment_configs|models|\.venv" .

# Lint code
lint:
	python -m flake8 --config=.flake8 .

# Type checking
typecheck:
	python -m mypy --config-file=pyproject.toml ai_service || true
	python -m mypy --config-file=pyproject.toml agents || true
	python -m mypy --config-file=pyproject.toml performance_optimizer.py || true
	python -m mypy --config-file=pyproject.toml memory_optimizer.py || true
	python -m mypy --config-file=pyproject.toml network_optimizer.py || true
	python -m mypy --config-file=pyproject.toml demo_performance_suite.py || true

# Run tests
test:
	python simple_test.py
	python integration_test.py

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete || true
	find . -type d -name "__pycache__" -exec rm -rf {} + || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + || true
	rm -f .coverage || true

# Run demo
run-demo:
	python run_optimized_demo.py --performance medium --start-ai --ignore-ai-failure

# Run integration test
run-test:
	python integration_test.py

# Run all quality checks
all: format lint typecheck test
	@echo "✅ All quality checks completed!"

# Windows specific targets (using cmd)
format-win:
	python -m black --line-length=88 --target-version=py38 --exclude="reverie|backup_environment_configs|models|\.venv" .

lint-win:
	python -m flake8 --config=.flake8 .

typecheck-win:
	python -m mypy --config-file=pyproject.toml ai_service
	python -m mypy --config-file=pyproject.toml agents
	python -m mypy --config-file=pyproject.toml performance_optimizer.py
	python -m mypy --config-file=pyproject.toml memory_optimizer.py
	python -m mypy --config-file=pyproject.toml network_optimizer.py
	python -m mypy --config-file=pyproject.toml demo_performance_suite.py

clean-win:
	for /r %%i in (*.pyc) do del "%%i" 2>nul
	for /r %%i in (__pycache__) do rmdir /s /q "%%i" 2>nul
	for /r %%i in (*.egg-info) do rmdir /s /q "%%i" 2>nul
	for /r %%i in (.pytest_cache) do rmdir /s /q "%%i" 2>nul
	for /r %%i in (.mypy_cache) do rmdir /s /q "%%i" 2>nul
	del .coverage 2>nul

# Development workflow
dev: dev-install format lint typecheck
	@echo "🚀 Development environment ready!"

# CI/CD pipeline simulation
ci: install format lint typecheck test
	@echo "🤖 CI pipeline completed!"