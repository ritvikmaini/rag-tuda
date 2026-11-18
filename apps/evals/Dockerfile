# The evals job image: runner, gate, stats, compare, rescore and smoke against the assistant over HTTP. A `builder`, a `base`
# and the `job` stage.
# Build context: the repository root (its .dockerignore lets in only the workspace and the profiles). The base is
# the assistant's python:3.12 slim on bookworm, pinned by its index digest; no syntax line, so BuildKit pulls no frontend image.
ARG PYTHON_IMAGE=python:3.12.15-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

FROM ${PYTHON_IMAGE} AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
RUN pip install --no-cache-dir uv==0.12.23
WORKDIR /src
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY apps/ apps/
RUN uv sync --frozen --no-default-groups --no-editable --package evals

FROM ${PYTHON_IMAGE} AS base
ARG PROFILE=example
# /app/out: everything a command writes (evals.settings out_dir); the deploy copies it out with docker cp, because the job's
# non-root user may not be able to write the bind mount data/eval_runs, which the job reads read-only
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin evals && install -d -o evals -g evals /app/out
COPY --from=builder /app/.venv /app/.venv
COPY profiles/${PROFILE}/ /app/profile/
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS job
USER evals
ENTRYPOINT ["evals"]
