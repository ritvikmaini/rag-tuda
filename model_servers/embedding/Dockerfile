# The embedding server image. The workspace's digest-pinned Python 3.12 slim (bookworm) image in both stages: no CUDA base,
# because each torch wheel brings its own CUDA runtime and cuDNN. The packages come from uv.lock. The model, its revisions
# and its config overrides are in the environment, so the image tag fixes them; HF_HUB_OFFLINE: no download at runtime.
# The NVIDIA lines are set here (the container toolkit reads them). No syntax line, so BuildKit pulls no frontend image.
# The container runs as root.
ARG PYTHON_IMAGE=python:3.12.15-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

FROM ${PYTHON_IMAGE} AS builder

WORKDIR /app

# uv at a pinned version installs exactly uv.lock into /opt/venv (no pip and no wheel there: nothing at runtime needs them)
RUN pip3 install --no-cache-dir uv==0.12.23
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON=/usr/local/bin/python3.12 UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM ${PYTHON_IMAGE}

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY embedding_server/ embedding_server/

ENV EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-m-v2.0
ENV EMBEDDING_REVISION=95c2741480856aa9666782eb4afe11959938017f
ENV EMBEDDING_CODE_REVISION=95c2741480856aa9666782eb4afe11959938017f
ENV EMBEDDING_CONFIG_KWARGS='{"use_memory_efficient_attention": false, "unpad_inputs": false}'
ENV HF_HUB_OFFLINE=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8001
CMD ["uvicorn", "--factory", "embedding_server.main:create_app", "--host", "0.0.0.0", "--port", "8001"]
