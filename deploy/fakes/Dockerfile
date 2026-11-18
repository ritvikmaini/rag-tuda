# The harness's fakes: one image for every fake of deploy/compose.ci.yaml,
# each started as `python -m rag_testing.serve <name> --port <port>`. The one image that installs rag_testing; it is built
# only for the harness (Mac, CI), never for the stack. Build context: the repository root (its .dockerignore allowlist lets
# in the workspace). Base: the evals image's python:3.12 slim on bookworm, pinned by its index digest.
ARG PYTHON_IMAGE=python:3.12.15-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

FROM ${PYTHON_IMAGE} AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
RUN pip install --no-cache-dir uv==0.12.23
WORKDIR /src
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY apps/ apps/
RUN uv sync --frozen --no-default-groups --no-editable --package rag-testing

FROM ${PYTHON_IMAGE}
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin fakes
COPY --from=builder /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /app
USER fakes
