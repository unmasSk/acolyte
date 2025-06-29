# Cross-platform Makefile for ACOLYTE
# Works on Windows, macOS, and Linux
all:
	@cls
	poetry run black .
	poetry run ruff check --fix src/
	poetry run python -c "import shutil, os, glob; [shutil.rmtree(p) for p in glob.glob('**/__pycache__', recursive=True) if os.path.exists(p)]; [shutil.rmtree(p) for p in glob.glob('**/*.egg-info', recursive=True) if os.path.exists(p)]; [os.remove(p) for p in glob.glob('**/*.pyc', recursive=True) + glob.glob('**/*.pyo', recursive=True) + glob.glob('**/*.pyd', recursive=True) if os.path.exists(p)]; [shutil.rmtree(d) for d in ['.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov', 'build', 'dist'] if os.path.exists(d)]; [os.remove('.coverage') if os.path.exists('.coverage') else None]; [os.remove('debug.log') if os.path.exists('debug.log') else None]"
	
format:
	poetry run black .

lint:
	poetry run ruff check --fix src/

mypy:
	poetry run mypy .

up:
	docker compose up -d

down:
	docker compose down

# Cross-platform clean using Python
clean:
	poetry run python -c "import shutil, os, glob; [shutil.rmtree(p) for p in glob.glob('**/__pycache__', recursive=True) if os.path.exists(p)]; [shutil.rmtree(p) for p in glob.glob('**/*.egg-info', recursive=True) if os.path.exists(p)]; [os.remove(p) for p in glob.glob('**/*.pyc', recursive=True) + glob.glob('**/*.pyo', recursive=True) + glob.glob('**/*.pyd', recursive=True) if os.path.exists(p)]; [shutil.rmtree(d) for d in ['.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov', 'build', 'dist'] if os.path.exists(d)]; [os.remove('.coverage') if os.path.exists('.coverage') else None]; [os.remove('debug.log') if os.path.exists('debug.log') else None]"

test:
	poetry run pytest tests

test.dream:
	poetry run pytest tests/dream

test.embeddings:
	poetry run pytest tests/embeddings

test.semantic:
	poetry run pytest tests/semantic

test.models:
	poetry run pytest tests/models

test.core:
	poetry run pytest tests/core

test.services:
	poetry run pytest tests/services

test.api:
	poetry run pytest tests/api

test.rag:
	poetry run pytest tests/rag

test.retrieval:
	poetry run pytest tests/rag/retrieval

test.enrichment:
	poetry run pytest tests/rag/enrichment

test.graph:
	poetry run pytest tests/rag/graph

test.collections:
	poetry run pytest tests/rag/collections

test.chunking:
	poetry run pytest tests/rag/chunking

test.compression:
	poetry run pytest tests/rag/compression

# Tests with coverage
test.c:
	poetry run pytest --cov=acolyte --cov-report=term-missing --cov-report=html

test.core.c:
	poetry run pytest tests/core --cov=acolyte.core --cov-report=term-missing --cov-report=html

test.services.c:
	poetry run pytest tests/services --cov=acolyte.services --cov-report=term-missing --cov-report=html

test.semantic.c:
	poetry run pytest tests/semantic --cov=acolyte.semantic --cov-report=term-missing --cov-report=html

test.models.c:
	poetry run pytest tests/models --cov=acolyte.models --cov-report=term-missing --cov-report=html

test.api.c:
	poetry run pytest tests/api --cov=acolyte.api --cov-report=term-missing --cov-report=html

test.embeddings.c:
	poetry run pytest tests/embeddings --cov=acolyte.embeddings --cov-report=term-missing --cov-report=html

test.dream.c:
	poetry run pytest tests/dream --cov=acolyte.dream --cov-report=term-missing --cov-report=html

test.rag.c:
	poetry run pytest tests/rag --cov=acolyte.rag --cov-report=term-missing --cov-report=html

test.retrieval.c:
	poetry run pytest tests/rag/retrieval --cov=acolyte.rag.retrieval --cov-report=term-missing --cov-report=html

test.compression.c:
	poetry run pytest tests/rag/compression --cov=acolyte.rag.compression --cov-report=term-missing --cov-report=html

test.chunking.c:
	poetry run pytest tests/rag/chunking --cov=acolyte.rag.chunking --cov-report=term-missing --cov-report=html

test.enrichment.c:
	poetry run pytest tests/rag/enrichment --cov=acolyte.rag.enrichment --cov-report=term-missing --cov-report=html

test.graph.c:
	poetry run pytest tests/rag/graph --cov=acolyte.rag.graph --cov-report=term-missing --cov-report=html

test.collections.c:
	poetry run pytest tests/rag/collections --cov=acolyte.rag.collections --cov-report=term-missing --cov-report=html

# Install/update project
install:
	poetry install

# Run the ACOLYTE server
run:
	poetry run uvicorn acolyte.api.server:app --host 127.0.0.1 --port 8000 --reload

# Quick quality check
check: lint mypy test

.PHONY: format lint mypy up down clean test test.rag test.retrieval test.enrichment test.graph test.collections test.chunking test-cov test.core.c test.models.c test.embeddings.c test.dream.c test.rag.c test.retrieval.c test.compression.c test.chunking.c test.enrichment.c test.graph.c test.collections.c install run check
