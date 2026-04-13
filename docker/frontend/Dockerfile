# Build stage: install dependencies using uv
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
# We install exactly the same virtual environment using the lockfile to ensure uniformity
# However, this microservice will only launch the Streamlit frontend.
RUN uv sync --no-dev --frozen --no-cache

# ─────────────────────────────────────────────────────────
# Runtime stage: lean production image
FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN groupadd --system appgroup && useradd --system --gid appgroup --no-create-home appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY src/frontend ./src/frontend
COPY src/__init__.py ./src/__init__.py
COPY .streamlit ./.streamlit

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8501

CMD ["sh", "-c", "streamlit run src/frontend/app.py --server.port=8501 --server.address=0.0.0.0"]
