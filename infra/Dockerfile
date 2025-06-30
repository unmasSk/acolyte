FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy pyproject.toml and poetry.lock first for dependency caching
COPY pyproject.toml poetry.lock ./

# Install poetry and dependencies
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root

# Copy the entire project
COPY . .

# Install the project itself
RUN poetry install --only main --no-interaction --no-ansi

# Create non-root user
RUN useradd -m -u 1000 acolyte && \
    chown -R acolyte:acolyte /app

USER acolyte

# Expose port
EXPOSE 8000

# Default command for development
CMD ["python", "-m", "uvicorn", "acolyte.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
