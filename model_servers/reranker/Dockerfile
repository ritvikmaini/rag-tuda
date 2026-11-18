# The reranker server image. The workspace's digest-pinned Python 3.12 slim (bookworm) image in both stages: no CUDA base,
# because each torch wheel brings its own CUDA runtime and cuDNN. The packages come from uv.lock; no compiler at runtime
# (the embedding image runs the same torch and triton wheels without one). The model and its revision are in the
# environment, so the image tag fixes them; HF_HUB_OFFLINE: no download at runtime. The NVIDIA lines are set here (the
# container toolkit reads them). No syntax line, so BuildKit pulls no frontend image. The container runs as root.
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

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY reranker_server/ reranker_server/

ENV RERANKER_MODEL=BAAI/bge-reranker-v2-m3
ENV RERANKER_REVISION=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
ENV HF_HUB_OFFLINE=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8003
CMD ["uvicorn", "--factory", "reranker_server.main:create_app", "--host", "0.0.0.0", "--port", "8003"]
