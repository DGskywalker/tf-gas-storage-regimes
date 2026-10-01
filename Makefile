.PHONY: all install run test lint format typecheck data features train report docker-build docker-run clean help

PYTHON ?= .venv/bin/python
UV ?= uv

help:
	@echo "Natural Gas Storage & TTF Price Relationship Analysis"
	@echo "Available targets:"
	@echo "  install      - Install dependencies using uv"
	@echo "  run          - Run full reproducible end-to-end pipeline"
	@echo "  data         - Ingest and align storage and price datasets"
	@echo "  features     - Compute engineered features and regime states"
	@echo "  train        - Train Bayesian and GBM models and baselines"
	@echo "  report       - Generate publication figures and research report"
	@echo "  test         - Run test suite with pytest and coverage"
	@echo "  lint         - Check code style with ruff"
	@echo "  format       - Auto-format code with ruff"
	@echo "  typecheck    - Run strict mypy type checking"
	@echo "  docker-build - Build Docker containers"
	@echo "  docker-run   - Run pipeline inside Docker container"
	@echo "  clean        - Remove build artifacts, caches, and temp files"

install:
	$(UV) venv --python 3.11
	$(UV) pip install -e ".[dev]"

run:
	$(PYTHON) -m src.data.pipeline --run-all

data:
	$(PYTHON) -m src.data.pipeline --stage ingestion

features:
	$(PYTHON) -m src.data.pipeline --stage features

train:
	$(PYTHON) -m src.data.pipeline --stage models

report:
	$(PYTHON) -m src.data.pipeline --stage report

test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests

typecheck:
	$(PYTHON) -m mypy src

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-run:
	docker compose -f docker/docker-compose.yml up pipeline

clean:
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
