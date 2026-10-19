"""
Phase 3 -- competition-style evaluation on 10 reconstructed world settings (configs/worlds/phase3_worlds.yaml) + Generalisation Score on held-out worlds. Requires 'cli phase3-select' after Phase 1 unless phase3.mode is all_variants.

Thin wrapper around the manifest-based runner in src/experiments/cli.py; the run list
itself is defined in src/experiments/specs.py (build_phase3_specs).
"""
from __future__ import annotations

from src.experiments.cli import main


def run_phase3(config_path: str | None = None, smoke: bool = False, workers: int = 1) -> int:
    argv = ["run", "--phase", "3", "--workers", str(workers)]
    if config_path:
        argv += ["--config", config_path]
    if smoke:
        argv.append("--smoke")
    return main(argv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 3")
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(run_phase3(args.config, smoke=args.smoke, workers=args.workers))
