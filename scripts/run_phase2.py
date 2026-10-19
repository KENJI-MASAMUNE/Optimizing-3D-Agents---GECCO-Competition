#!/usr/bin/env python3
"""Convenience entry point: `python scripts/run_phase2.py [--smoke] [--config PATH] [--workers N]`
(equivalent to `python -m src.experiments.cli run --phase 2 ...`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.phase2_sensitivity import run_phase2  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", action="store_true", help="Quick sanity run")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(run_phase2(args.config, smoke=args.smoke, workers=args.workers))
