FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, deterministic dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.5.24 /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy dependency definition
COPY pyproject.toml .

# Install dependencies into system environment
RUN uv pip install --system -e .[dev]

# Copy project source code
COPY . .

# Default command runs the end-to-end pipeline
CMD ["python", "-m", "src.data.pipeline", "--run-all"]
