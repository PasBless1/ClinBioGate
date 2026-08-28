"""Unified command-line entry point.

Usage:
    olives audit    --config configs/data.yaml
    olives manifest --config configs/data.yaml
    olives splits   --config configs/data.yaml --folds
    olives train      --config configs/baseline_oct.yaml
    olives compare    --budget local_cpu --evaluate
    olives evaluate   --run-dir outputs/runs/<run_id>
    olives report     --experiment-dir outputs/runs/local_cpu
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

COMMANDS = {
    "audit": "audit_data",
    "manifest": "build_manifest",
    "splits": "make_splits",
    "train": "train",
    "compare": "run_comparison",
    "evaluate": "evaluate",
    "report": "generate_report",
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the matching script in ``scripts/``."""
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in {"-h", "--help"}:
        print(__doc__)
        print(f"commands: {', '.join(COMMANDS)}")
        return 0

    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        print(f"unknown command {command!r}; choose from {', '.join(COMMANDS)}")
        return 2

    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    import importlib

    module = importlib.import_module(COMMANDS[command])
    return int(module.main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
