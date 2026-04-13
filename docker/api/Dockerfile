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

# Create non-root user with a home directory (Required by some Python libs/Gunicorn)
RUN groupadd --system appgroup && useradd --system --gid appgroup --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY src ./src

# Bake in the pre-trained model artifact and gold feature CSV
# (both are small: ~100KB model, ~14MB CSV — no persistent disk needed on Render)
COPY models ./models
COPY data/gold ./data/gold
RUN mkdir -p /app/data /app/models && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Use gunicorn with 1 worker to stay within Render Free Tier 512MB RAM limit
CMD ["sh", "-c", "gunicorn src.api.main:app --workers ${WEB_CONCURRENCY:-1} --worker-class uvicorn.workers.UvicornWorker --bind ${API_HOST:-0.0.0.0}:${API_PORT:-8000}"]
