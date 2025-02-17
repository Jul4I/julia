# syntax = docker/dockerfile:1.2

# choose your Python version
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.5.7 /uv /bin/uv

ENV \
    # do not buffer python output at all
    PYTHONUNBUFFERED=1 \
    # do not write `__pycache__` bytecode
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY . .

RUN uv sync \
        --frozen \
        --compile-bytecode \
        --no-editable \
        --no-dev

RUN uv export --format requirements-txt -o requirements.txt 

FROM python:${PYTHON_VERSION}-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv

WORKDIR /app

ENTRYPOINT [ "python", "-m", "julia.main"]
