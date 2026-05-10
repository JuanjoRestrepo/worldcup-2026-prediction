# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder: install production dependencies via uv
# ─────────────────────────────────────────────────────────────────────────────
# Pin uv to a specific digest (not :latest) for reproducible builds.
# The builder is a full Python image; the runner will be a slim copy.
# Layer order follows the "cache invalidation pyramid":
#   base image → system deps → dependency manifests → package install → source.
# Placing COPY pyproject.toml before COPY . . ensures that source-only
# changes do NOT bust the expensive pip/uv dependency-install layer.
FROM python:3.12-slim AS builder

# Pin uv to a specific stable release for reproducible builds.
# Using :latest would silently change behaviour on re-builds.
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /usr/local/bin/uv

# Disable .pyc files and enable unbuffered stdout/stderr for clean container logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell uv to install into the project venv (not the system Python)
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Copy only the dependency manifests first — this layer is only invalidated
# when pyproject.toml or uv.lock changes, not on every code change.
COPY pyproject.toml uv.lock ./

# Install production deps. --no-dev excludes dev tools (ruff, mypy, pytest).
# --frozen guarantees the lockfile is never auto-updated in CI.
# --no-cache prevents uv from writing a cache layer inside the builder.
RUN uv sync --no-dev --frozen --no-cache

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runner: lean production image (no uv, no build tools)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runner

# Inherit non-functional env vars for consistency
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Default server binding — overridden at runtime via Render/docker-compose env
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    # Prepend the venv bin to PATH so `python` and `gunicorn` resolve without activation
    PATH="/app/.venv/bin:$PATH"

# Create a non-root system user. Running as root in a container is a security
# anti-pattern — if the process is exploited, the attacker gets root on the host.
# --system: no login shell; --gid: assign to the group created above.
RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home appuser

WORKDIR /app

# Copy only the pre-built virtual environment from the builder stage.
# This drops all of uv, pip, and the build toolchain from the final image.
COPY --from=builder /app/.venv /app/.venv

# Copy application source — placed AFTER the venv copy so that code changes
# only invalidate the cheaper COPY layer, not the package install.
COPY src ./src

# Bake in the pre-trained model artifact and gold feature CSV snapshot.
# Both are small (~100KB model, ~14MB CSV) — no persistent disk needed on Render free tier.
COPY models ./models
COPY data/gold ./data/gold

# Set correct ownership in a single RUN command (combining into one layer is
# more efficient than a separate RUN chown because it avoids intermediate layers).
RUN mkdir -p /app/data /app/models \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Use gunicorn + UvicornWorker for async support on a production ASGI app.
# WEB_CONCURRENCY defaults to 1 to stay within Render Free Tier 512MB RAM.
# Shell form required to expand env vars at container start time.
CMD ["sh", "-c", "gunicorn backend.api.main:app \
    --workers ${WEB_CONCURRENCY:-1} \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind ${API_HOST:-0.0.0.0}:${API_PORT:-8000} \
    --timeout 120 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile -"]
