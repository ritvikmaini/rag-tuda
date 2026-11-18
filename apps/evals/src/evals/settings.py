"""Settings of the evals job: one technical section, evals, read by rag_core.settings.load.

In the container the token is the compose secret backend_token (/run/secrets/backend_token); finished runs are read from
runs_dir (data/eval_runs, mounted read-only) and every command writes into out_dir, the container's own folder, which the deploy
copies out with docker cp (the job's non-root user may not be able to write the bind mount). On a development machine
the environment sets what differs, e.g. RAG__EVALS__PROFILE_DIR=profiles/example and RAG__EVALS__RUNS_DIR and
RAG__EVALS__OUT_DIR both =data/eval_runs."""

import os
from collections.abc import Mapping

from pydantic import Field, SecretStr

from rag_core.settings import OverlaySection, load, secret


class EvalsSection(OverlaySection):
    assistant_url: str = Field(default="http://backend:8000", pattern=r"^https?://\S+$")
    timeout_s: float = Field(default=240.0, gt=0, allow_inf_nan=False)
    min_interval_s: float = 4.0
    runs_dir: str = "/app/data/eval_runs"  # read: finished runs (gate labels); the compose job mounts it read-only
    out_dir: str = "/app/out"  # written: runs, re-scored copies, gate reports, smokes; created for the evals user
    profile_dir: str = "/app/profile"
    token: SecretStr | None = secret("backend_token")


class EvalsSettings(OverlaySection):
    evals: EvalsSection = EvalsSection()


def load_settings(environ: Mapping[str, str] = os.environ, secrets_dir: str = "/run/secrets") -> EvalsSettings:
    return load(EvalsSettings, environ=environ, secrets_dir=secrets_dir)
