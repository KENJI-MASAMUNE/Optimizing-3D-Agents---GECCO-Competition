"""
Phase 1 -- controlled internal experiments (proposal Section 5.3): 5 variants x 2 encodings x 4 proxy tasks x n_repeats in the base training world.

Thin wrapper around the manifest-based runner in src/experiments/cli.py; the run list
itself is defined in src/experiments/specs.py (build_phase1_specs).
"""
from __future__ import annotations

from src.experiments.cli import main


def run_phase1(config_path: str | None = None, smoke: bool = False, workers: int = 1) -> int:
    argv = ["run", "--phase", "1", "--workers", str(workers)]
    if config_path:
        argv += ["--config", config_path]
    if smoke:
        argv.append("--smoke")
    return main(argv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 1")
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(run_phase1(args.config, smoke=args.smoke, workers=args.workers))
