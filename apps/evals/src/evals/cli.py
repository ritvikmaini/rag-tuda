"""The console script `evals`: evals run|gate|stats|compare|rescore|smoke <arguments>.

Every command is its module's main(argv). argparse exits 2 on a usage error, which for `evals gate` means "re-runs needed",
so a usage error leaves this script as 64 (EX_USAGE); a command that could not run (an exception, an unreadable run) exits 70
(EX_SOFTWARE), never 1 (a failed gate) or 10 (a failed smoke). The deploy and CI read the exit codes."""

import sys
import traceback

from evals import compare, gate, rescore, runner, smoke, stats

COMMANDS = {
    "run": runner.main,
    "gate": gate.main,
    "stats": stats.main,
    "compare": compare.main,
    "rescore": rescore.main,
    "smoke": smoke.main,
}
USAGE = "usage: evals {run,gate,stats,compare,rescore,smoke} ...   (evals <command> --help)"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    if not argv or argv[0] not in COMMANDS:
        print(USAGE, file=sys.stderr)
        return 64
    try:
        return COMMANDS[argv[0]](argv[1:])
    except SystemExit as exc:
        if exc.code == 2:  # argparse's usage error
            return 64
        if exc.code is None or isinstance(exc.code, int):
            raise
        print(exc.code, file=sys.stderr)  # SystemExit("message"), e.g. stats.load_rows on a run without sections
        return 70
    except Exception:
        traceback.print_exc()
        return 70
