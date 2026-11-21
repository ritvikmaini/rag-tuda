"""`rag eval`: the evals job against the running stack, as one step for the executor.

The smoke runs in the evals job (`evals smoke`, the native stack's token read from its secret file); the retrieval-only
eval and its ablation are the assistant's module `assistant_api.retrieval_eval`, run natively with the backend's settings
(it needs the database and the model servers, and the card's analysis model)."""

from rag_tuda import runtime
from rag_tuda.stack import Args, Facts, Step, StackError, compose_for, uv_argv
from rag_tuda.state import State

RETRIEVAL_MODULE = "assistant_api.retrieval_eval"
NEEDS = {"smoke": ("backend",), "retrieval": ("embedding", "reranker")}


def plan(
    state: State, facts: Facts, args: Args, *, retrieval_only: bool, ablation: bool, check_ready: bool = True
) -> list[Step]:
    """The step that runs the eval on the card of `args`. `check_ready` false when the plan before it brings the stack up."""
    kind = "retrieval" if retrieval_only else "smoke"
    llm = runtime.resolve(facts.card, args.mode, model=args.model, llm_url=args.llm_url)
    if check_ready and (missing := [n for n in NEEDS[kind] if n not in facts.ready]):
        raise StackError(f"{', '.join(missing)}: not running: run `rag up` first")
    if retrieval_only and args.mode != "native":
        raise StackError("the retrieval-only eval runs against the native stack: use `--mode native`")
    if llm.provider in runtime.REMOTE and retrieval_only and not facts.groq_key:
        raise StackError(f"{llm.card}: its analysis needs GROQ_API_KEY from the environment: export it first")
    root, rag = facts.root, str(state.dir)
    if retrieval_only:
        env = runtime.backend_env(llm, root=root, secrets=str(state.secrets_dir("backend")), profile=args.profile)
        argv = uv_argv(
            "--package", "assistant-api", "python", "-m", RETRIEVAL_MODULE, *(("--ablation",) if ablation else ())
        )
        text = f"retrieval-only eval ({'ablation' if ablation else 'hit@k and MRR'}) on card {llm.card}"
        return [Step("run", name="eval", text=text, argv=argv, cwd=root, env=env, inherit=_key(llm))]
    text = f"smoke eval on card {llm.card}"
    if args.mode == "compose":
        argv = (
            *compose_for(facts, rag, llm),
            "--profile",
            "evals",
            "run",
            "--rm",
            "evals",
            "smoke",
            "--label",
            llm.card,
        )
        return [Step("run", name="eval", text=text, argv=argv, cwd=root)]
    env = {
        "RAG__EVALS__ASSISTANT_URL": runtime.endpoint("backend"),
        "RAG_SECRETS_DIR": str(state.secrets_dir("backend")),  # holds backend_token, the token file the job reads
        "RAG__EVALS__PROFILE_DIR": f"{root}/profiles/{args.profile}",
        "RAG__EVALS__RUNS_DIR": f"{rag}/eval_runs",
        "RAG__EVALS__OUT_DIR": f"{rag}/eval_runs",
    }
    argv = uv_argv("--package", "evals", "evals", "smoke", "--label", llm.card)
    return [Step("run", name="eval", text=text, argv=argv, cwd=root, env=env)]


def _key(llm: runtime.Llm) -> tuple[str, ...]:
    return ("GROQ_API_KEY",) if llm.provider in runtime.REMOTE else ()
