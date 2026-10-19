"""
Phase 2 -- sensitivity analysis (proposal Section 5.3): population size x tournament size (fixed evaluation budget), adaptive tau (and k), Variant 4 gene/graph weight per encoding. See docs/PHASES_AND_CHPC.md for what this phase is for.

Thin wrapper around the manifest-based runner in src/experiments/cli.py; the run list
itself is defined in src/experiments/specs.py (build_phase2_specs).
"""
from __future__ import annotations

from src.experiments.cli import main


def run_phase2(config_path: str | None = None, smoke: bool = False, workers: int = 1) -> int:
    argv = ["run", "--phase", "2", "--workers", str(workers)]
    if config_path:
        argv += ["--config", config_path]
    if smoke:
        argv.append("--smoke")
    return main(argv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 2")
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(run_phase2(args.config, smoke=args.smoke, workers=args.workers))
