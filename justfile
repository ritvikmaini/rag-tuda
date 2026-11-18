# Recipes, identical on the Mac and in CI. Every Python tool runs from uv.lock.
set shell := ["bash", "-euo", "pipefail", "-c"]

uv := "uv run --frozen --all-packages"
# Two worktrees can run `just ci` at once: a linked worktree's id is its name in ASCII ([a-z0-9-]) plus 6 hex of its
# absolute path's hash (so a.b, a_b and a-b differ). It names the harness project (rag-ci-<id>) and image tags (:ci-<id>),
# and Docker picks the harness host ports. The main checkout and GitHub CI have no id: project rag-ci and the plain tags
# and ports. RAG_CI_ID overrides it (empty: no id).
ci_id := env("RAG_CI_ID", `{ read -r dir; read -r common; read -r top; } < <(git rev-parse --path-format=absolute --git-dir --git-common-dir --show-toplevel 2>/dev/null) || exit 0; [ "$dir" = "$common" ] || printf '%s-%s' "$(printf '%s' "${dir##*/}" | LC_ALL=C tr A-Z a-z | LC_ALL=C tr -c a-z0-9 -)" "$(printf '%s' "$top" | git hash-object --stdin | cut -c1-6)"`)
export RAG_CI_TAG := if ci_id == "" { "" } else { ":ci-" + ci_id }
ci_project := if ci_id == "" { "rag-ci" } else { "rag-ci-" + ci_id }
# The coverage ratchet: branch coverage per member, measured and rounded down. A lowered number is a review finding; a change
# that raises coverage may raise its number.
cov_rag_core := "86"  # measured 86.41; domain/status_groups.py (25%) is mostly untested
# rag_core files measured with apps/assistant_api under the assistant's floor instead of rag_core's: untyped domain code,
# and the SQL and HTTP adapters the harness covers. The rag_core floor measures the other files (prompts, retrieval,
# analysis and the other services)
rag_core_assistant_covered := "*/packages/rag_core/src/rag_core/domain/child_text.py,*/packages/rag_core/src/rag_core/domain/citations.py,*/packages/rag_core/src/rag_core/domain/context_budget.py,*/packages/rag_core/src/rag_core/domain/doc_scores.py,*/packages/rag_core/src/rag_core/domain/parent_rerank.py,*/packages/rag_core/src/rag_core/domain/query_analysis.py,*/packages/rag_core/src/rag_core/domain/ranking.py,*/packages/rag_core/src/rag_core/domain/resilience.py,*/packages/rag_core/src/rag_core/domain/source_meta.py,*/packages/rag_core/src/rag_core/domain/answer_shapes.py,*/packages/rag_core/src/rag_core/domain/history.py,*/packages/rag_core/src/rag_core/adapters/http.py,*/packages/rag_core/src/rag_core/adapters/langfuse.py"
cov_evals := "90"  # measured 90.90
cov_rag_testing := "87"  # measured 87.77
# the indexer (measured 73.01; its builds are covered by the harness)
cov_indexer := "73"
# apps/assistant_api and the rag_core_assistant_covered files together (measured 88.86; rounded down)
cov_assistant := "88"

# all checks of the gate; db-tests, images and harness need Docker (Colima on the Mac: colima start --cpu 6 --memory 12)
ci: lint types imports unit cli db-tests contracts model-servers images (harness "example")

# what the pre-push hook runs (no Docker)
pre-push: lint types imports unit contracts

lint: lock ruff deptry leaks

# the lock matches every pyproject (`uv run --frozen` alone would run a stale lock without noticing)
lock:
    uv lock --check
    uv lock --check --project model_servers/embedding
    uv lock --check --project model_servers/reranker

ruff:
    {{uv}} ruff check
    {{uv}} ruff format --check

deptry:
    cd packages/rag_core && {{uv}} deptry src
    cd packages/rag_testing && {{uv}} deptry src
    cd apps/evals && {{uv}} deptry src
    cd apps/indexer && {{uv}} deptry src
    cd apps/assistant_api && {{uv}} deptry src

# gitleaks over the files git would commit (tracked, plus untracked files that are not ignored): `gitleaks dir` itself
# ignores .gitignore and would scan .venv/, data/, secrets/ and .env
leaks:
    #!/usr/bin/env bash
    set -euo pipefail
    snapshot=$(mktemp -d); trap 'rm -rf "$snapshot"' EXIT
    git ls-files -z --cached --others --exclude-standard \
      | while IFS= read -r -d '' f; do if [ -f "$f" ]; then printf '%s\0' "$f"; fi; done \
      | tar --null -T - -cf - | tar -xf - -C "$snapshot"
    gitleaks dir "$snapshot" --no-banner --redact

types:
    {{uv}} basedpyright

imports:
    {{uv}} lint-imports

# the suite except contracts and the harness (RUN_DOCKER_TESTS=1 adds the Docker-gated tests), the ratchet
unit:
    {{uv}} pytest -q --ignore=tests/contracts --ignore=tests/integration --cov=rag_core --cov=evals --cov=rag_testing --cov=indexer --cov=assistant_api --cov-branch --cov-report=
    {{uv}} coverage report --include='*/packages/rag_core/*' --omit='{{rag_core_assistant_covered}}' --fail-under={{cov_rag_core}}
    {{uv}} coverage report --include='*/apps/evals/*' --fail-under={{cov_evals}}
    {{uv}} coverage report --include='*/packages/rag_testing/*' --fail-under={{cov_rag_testing}}
    {{uv}} coverage report --include='*/apps/indexer/*' --fail-under={{cov_indexer}}
    {{uv}} coverage report --include='*/apps/assistant_api/*,{{rag_core_assistant_covered}}' --fail-under={{cov_assistant}}

# the model servers' tests, each in its own uv environment: fake models, no GPU; HF_HUB_OFFLINE so no
# test can download a model; the first run installs each environment (on the Mac: torch's CPU wheels)
model-servers:
    cd model_servers/embedding && HF_HUB_OFFLINE=1 uv run --frozen pytest -q
    cd model_servers/reranker && HF_HUB_OFFLINE=1 uv run --frozen pytest -q

# the command line's tests as the cli-macos job runs them: a throwaway environment holding the root project alone, so a test
# that needs a workspace package or yaml fails here (those belong in tests/repo)
cli:
    #!/usr/bin/env bash
    set -euo pipefail
    env=$(mktemp -d); trap 'rm -rf "$env"' EXIT
    UV_PROJECT_ENVIRONMENT="$env/venv" uv run --frozen pytest -q tests/cli

# cross-component contracts, transcripts, limit parity and the profile's data
contracts:
    {{uv}} pytest -q tests/contracts tests/profiles

# the answer stream's JSON schema (the contract tests read it): regenerate after a frame change
stream-schema:
    {{uv}} python -m assistant_api.sse.frames > apps/assistant_api/stream_v1.schema.json

# every tracked Dockerfile checked (`docker build --check`, any finding fails; the model servers' too, never built), then the images
# of the evals job, the harness's fakes and the two apps for the host's architecture (Mac: arm64, never shipped;
# with DOCKER_DEFAULT_PLATFORM=linux/amd64 to build amd64 under emulation; CI: amd64, build only)
images:
    for f in $(git ls-files '*Dockerfile'); do docker build --check --build-arg BUILDKIT_DOCKERFILE_CHECK=error=true -f "$f" .; done
    docker build -f apps/evals/Dockerfile --build-arg PROFILE=example -t rag-tuda-evals${RAG_CI_TAG:-:dev} .
    docker build -f deploy/fakes/Dockerfile -t rag-ci-fakes$RAG_CI_TAG .
    docker build -f apps/assistant_api/Dockerfile --build-arg PROFILE=example -t rag-ci-backend-example$RAG_CI_TAG .
    docker build -f apps/indexer/Dockerfile --build-arg PROFILE=example -t rag-ci-indexer-example$RAG_CI_TAG .

# the database tests (RUN_DOCKER_TESTS=1): the pgvector service of deploy/compose.ci.yaml under the CI project, with the secrets
# `state init` writes, the migrations run as in the harness, then the tests that need a migrated database, then teardown. Tests that
# need a built index run in the harness. In a linked worktree Docker picks the host port.
db-tests:
    #!/usr/bin/env bash
    set -euo pipefail
    G=deploy/ci/.generated
    compose() { docker compose -f deploy/compose.ci.yaml -f deploy/compose.ci.example.yaml "$@"; }
    export COMPOSE_PROJECT_NAME={{ci_project}}
    if [ -n "$RAG_CI_TAG" ]; then export RAG_CI_PG_PORT=; fi
    compose down -v --remove-orphans >/dev/null 2>&1 || true
    trap 'compose down -v --remove-orphans >/dev/null 2>&1 || true' EXIT
    rm -rf "$G" && mkdir -p "$G"
    PYTHONPATH=src {{uv}} python -m rag_tuda.state init --root "$G" >/dev/null
    compose up -d --wait pgvector
    compose build backend
    compose --profile migrate run --rm --no-deps migrate
    if [ -n "$RAG_CI_TAG" ]; then export RAG_CI_PG_PORT=$(compose port pgvector 5432 | cut -d: -f2); fi
    RUN_DOCKER_TESTS=1 {{uv}} pytest -q tests/pgvector tests/assistant_api/test_migrations.py

# the black-box harness: deploy/compose.ci.yaml with deploy/compose.ci.example.yaml merged in (project rag-ci), the real backend
# and indexer images, then tests/integration (gated on RAG_CI_PROFILE=example), then teardown. `state init` writes
# deploy/ci/.generated/ (secrets, tokens). HARNESS_PYTEST_ARGS runs only the named tests; HARNESS_KEEP=1 leaves the stack
# up for a look (in a linked worktree first `set -a; . deploy/ci/.generated/harness.env; set +a`, for a follow-up pytest too; `docker compose -f deploy/compose.ci.yaml
# -f deploy/compose.ci.example.yaml down -v` ends it). In a linked worktree: project rag-ci-<id>, ports by Docker (exported to the
# tests, written to harness.env). Without HARNESS_PYTEST_ARGS the Docker-gated tests that need the built index run last.
harness profile="example":
    #!/usr/bin/env bash
    set -euo pipefail
    G=deploy/ci/.generated
    [ "{{profile}}" = example ] || { echo "unknown profile {{profile}} (example)"; exit 2; }
    compose() { docker compose -f deploy/compose.ci.yaml -f deploy/compose.ci.example.yaml "$@"; }
    export COMPOSE_PROJECT_NAME={{ci_project}}
    if [ -n "$RAG_CI_TAG" ]; then export RAG_CI_PG_PORT= RAG_CI_LLM_PORT= RAG_CI_BACKEND_PORT=; fi
    compose --profile evals --profile build down -v --remove-orphans >/dev/null 2>&1 || true
    if [ "${HARNESS_KEEP:-}" != 1 ]; then trap 'compose --profile evals --profile build down -v --remove-orphans >/dev/null 2>&1 || true' EXIT; fi
    rm -rf "$G" && mkdir -p "$G/data"
    PYTHONPATH=src {{uv}} python -m rag_tuda.state init --root "$G" >/dev/null
    compose up -d --wait pgvector
    compose build backend indexer
    compose --profile migrate run --rm --no-deps migrate
    compose up -d --build --wait
    if [ -n "$RAG_CI_TAG" ]; then
      port() { compose port "$1" "$2" | cut -d: -f2; }
      RAG_CI_PG_PORT=$(port pgvector 5432)
      RAG_CI_LLM_PORT=$(port vllm 8000)
      RAG_CI_BACKEND_PORT=$(port backend 8000)
      for v in RAG_CI_PG_PORT RAG_CI_LLM_PORT RAG_CI_BACKEND_PORT; do
        [ -n "${!v}" ] || { echo "harness: no host port for $v (never fall back to the main checkout's)" >&2; exit 1; }
      done
    fi
    env | grep -E '^(COMPOSE_PROJECT_NAME|RAG_CI_)' > "$G/harness.env"
    RAG_CI_PROFILE=example {{uv}} pytest -q ${HARNESS_PYTEST_ARGS:-tests/integration}
    if [ -z "${HARNESS_PYTEST_ARGS:-}" ]; then
      # the first uses the image `just images` builds; the last needs the index the leg built
      RUN_DOCKER_TESTS=1 {{uv}} pytest -q tests/repo/test_assistant.py::test_the_assistant_image_installs_exactly_its_locked_set tests/assistant_api/chat/test_turns_store.py
    fi
