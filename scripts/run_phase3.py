#!/usr/bin/env python3
"""Convenience entry point: `python scripts/run_phase3.py [--config PATH] [--phase1-results DIR]`"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.phase3_evaluation import run_phase3  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--phase1-results", default=None)
    args = parser.parse_args()
    run_phase3(args.config, args.phase1_results)
