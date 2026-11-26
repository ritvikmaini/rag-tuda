# The assistant API image. Root build context (.dockerignore admits only the workspace, the profiles, the model cards and the
# page). The workspace lock installs the locked packages plus rag-core and assistant-api (tests/repo/test_assistant.py); the
# dependency layer comes first and is rebuilt only when a pyproject or the lock changes. The profile is baked (RAG_PROFILE) and
# so are the model cards (/app/cards; RAG__CARD names one) and the page (/app/page, served at /; RAG_PAGE_DIR): an image tag
# fixes every answer-affecting value. No locale: the prompt's weekday comes from the profile; TZ stays Berlin (the prompt's
# date and time).
# gunicorn runs the app factory with its config at /app (one worker: one GPU pipeline slot).
ARG PYTHON_IMAGE=python:3.12.15-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

FROM ${PYTHON_IMAGE} AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
RUN pip install --no-cache-dir uv==0.12.23
WORKDIR /src
COPY pyproject.toml uv.lock ./
COPY packages/rag_core/pyproject.toml packages/rag_core/
COPY packages/rag_testing/pyproject.toml packages/rag_testing/
COPY apps/evals/pyproject.toml apps/evals/
COPY apps/indexer/pyproject.toml apps/indexer/
COPY apps/assistant_api/pyproject.toml apps/assistant_api/
RUN uv sync --frozen --no-default-groups --no-editable --no-install-workspace --package assistant-api
COPY packages/ packages/
COPY apps/ apps/
RUN uv sync --frozen --no-default-groups --no-editable --package assistant-api

FROM ${PYTHON_IMAGE}
ARG PROFILE=example
# build-essential and the -dev packages may not be needed by the runtime (unverified)
RUN apt-get update && apt-get install -y build-essential libssl-dev libffi-dev python3-dev libgl1 libglib2.0-0 libsm6 curl \
    postgresql-client tzdata \
    && ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/.venv /app/.venv
COPY profiles/${PROFILE}/ /app/profile/
COPY models/cards/ /app/cards/
COPY apps/assistant_api/gunicorn_conf.py /app/
COPY web/page/ /app/page/
RUN mkdir -p /app/data/vector_database_contents
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 TZ=Europe/Berlin
ENV RAG_PROFILE=/app/profile/profile.yaml RAG_PAGE_DIR=/app/page
WORKDIR /app
EXPOSE 8000
CMD ["gunicorn", "assistant_api.app:create_app()", "-c", "gunicorn_conf.py"]
ARG GIT_COMMIT
ENV GIT_COMMIT=${GIT_COMMIT}
