# The indexer image. Build context: the repository root (its .dockerignore lets in only the workspace and the profiles).
# The base is python:3.12 slim on bookworm, pinned by its index digest; there is no syntax line, so BuildKit pulls no
# frontend image. The packages come from uv.lock. Root, because the indexer writes the bind mount /app/data and its
# model caches live under /root. Settings: the baked profile and the overlay at RAG_OVERLAY. A container runs one command, `indexer build` for a build.
ARG PYTHON_IMAGE=python:3.12.15-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

FROM ${PYTHON_IMAGE} AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
RUN pip install --no-cache-dir uv==0.12.23
WORKDIR /src
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
COPY apps/ apps/
RUN uv sync --frozen --no-default-groups --no-editable --package indexer
# The profile's connector package (deploy/compose.ci.example.yaml passes example-profile); the heavy layer above
# is shared by builds of different packages.
ARG CONNECTORS=example-profile
RUN uv sync --frozen --no-default-groups --no-editable --package indexer --package ${CONNECTORS}

FROM ${PYTHON_IMAGE}
# the compilers and -dev packages may not be needed by the runtime (unverified)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libmagic1 \
    gcc \
    g++ \
    make \
    postgresql-client \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ARG PROFILE=example
# the profile is data baked into the image: the image tag fixes every value that decides the index
COPY profiles/${PROFILE}/profile.yaml /app/profile/profile.yaml
ENV RAG_PROFILE=/app/profile/profile.yaml
RUN mkdir -p /app/data/vector_database_contents
COPY --from=builder /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
ENTRYPOINT ["indexer"]
CMD ["--help"]

# Provenance: the deploy passes --build-arg GIT_COMMIT=<HEAD>, and each index build records it in
# index_generations.git_commit. Last, so a new commit rebuilds only this layer.
ARG GIT_COMMIT
ENV GIT_COMMIT=${GIT_COMMIT}
