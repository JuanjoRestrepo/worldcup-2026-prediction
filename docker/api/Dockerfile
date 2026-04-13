# Build stage: install dependencies using uv
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
# Install only production dependencies (no dev group)
RUN uv sync --no-dev --frozen --no-cache

# ─────────────────────────────────────────────────────────
# Runtime stage: lean production image
FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Create non-root user for security (DevOps best practice)
RUN groupadd --system appgroup && useradd --system --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY src ./src

# Data and models are mounted via docker-compose volumes (not baked in)
RUN mkdir -p /app/data /app/models && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host ${API_HOST} --port ${API_PORT}"]
