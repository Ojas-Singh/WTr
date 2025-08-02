# Makefile for WTr development

.PHONY: help install install-dev test test-coverage clean docs format lint check example

help:
	@echo "WTr Development Commands:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  clean        Clean build artifacts"
	@echo "  docs         Build documentation"
	@echo "  format       Format code with black"
	@echo "  lint         Run code linting"
	@echo "  check        Run all checks (lint + test)"
	@echo "  example      Run example script"

install:
	pip install -e .

install-dev:
	pip install -e .
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=WTr --cov-report=html --cov-report=term

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

docs:
	@echo "Documentation build not yet configured"
	@echo "Would use sphinx-build docs/ docs/_build/"

format:
	black WTr/ tests/ example.py

lint:
	flake8 WTr/ tests/
	mypy WTr/ --ignore-missing-imports

check: lint test

example:
	python example.py

# Development environment setup
setup-dev:
	python -m venv venv
	@echo "Activate environment with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
	@echo "Then run: make install-dev"

# Package building
build:
	python setup.py sdist bdist_wheel

# Install from local build
install-local: build
	pip install dist/*.whl --force-reinstall
