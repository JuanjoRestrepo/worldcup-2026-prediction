FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.21 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

COPY src ./src

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host ${API_HOST} --port ${API_PORT}"]
