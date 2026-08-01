#!/usr/bin/env python3
"""Convenience entry point: `python scripts/run_phase2.py [--smoke] [--config PATH]`"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.phase2_sensitivity import run_phase2  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_phase2(args.config, smoke=args.smoke)
