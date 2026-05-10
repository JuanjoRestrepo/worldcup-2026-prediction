# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder: install production dependencies via uv
# ─────────────────────────────────────────────────────────────────────────────
# Pin uv to a specific stable release for reproducible builds.
# Layer order: manifests first → install → source, so dependency caching
# is not busted by every source-file change.
FROM python:3.12-slim AS builder

# Pin to a specific uv release — :latest silently breaks reproducibility.
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Copy manifests first — only invalidated on dependency changes, not source changes.
COPY pyproject.toml uv.lock ./

# Install production-only dependencies; no dev tools in the image.
RUN uv sync --no-dev --frozen --no-cache

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runner: lean production image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Non-root user for security — running as root is a container anti-pattern.
RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy the built venv from the builder — no build tools land in the runner.
COPY --from=builder /app/.venv /app/.venv

# Only the frontend module and its Streamlit config are needed.
# Placing these after the venv copy keeps the expensive install layer cached
# across source-only changes.
COPY frontend ./frontend
COPY frontend/.streamlit ./.streamlit

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8501

# Disable Streamlit's browser-open and telemetry for headless container operation.
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
