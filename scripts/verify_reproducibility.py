#!/usr/bin/env python3
"""
Reproducibility check on the REAL simulator (run this once on every new machine, e.g. on a
CHPC compute node before submitting the big jobs).

It runs the Phase 1 smoke manifest three times, each in fresh processes:
  A) all runs sequentially in one process,
  B) the same runs spread over a process pool (different run-to-worker assignment),
  C) a subset in reverse order with a different PYTHONHASHSEED,
and checks that every run produced exactly the same best fitness, best genotype and
per-generation history. Passing means a run's result depends only on its seed -- the
property the CHPC job arrays rely on.

    python scripts/verify_reproducibility.py --config configs/chpc_config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(results_dir: Path, config: str | None, extra: list[str], env_extra: dict | None = None) -> None:
    env = dict(os.environ, RESULTS_DIR=str(results_dir), **(env_extra or {}))
    cmd = [sys.executable, "-m", "src.experiments.cli", "run", "--phase", "1", "--smoke", "--quiet-gens"] + extra
    if config:
        cmd += ["--config", config]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _load(results_dir: Path) -> dict:
    out = {}
    for p in (results_dir / "smoke" / "phase1").glob("*.json"):
        r = json.loads(p.read_text())
        out[p.name] = (r["best_fitness"], r["best_genotype"], [h["best_fitness"] for h in r["history"]],
                       [h["mean_fitness"] for h in r["history"]], r["n_evaluations"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _run(tmp / "A", args.config, ["--workers", "1"])
        _run(tmp / "B", args.config, ["--workers", str(args.workers)])
        a = _load(tmp / "A")
        n = len(a)
        subset = list(range(n))[::-2]
        _run(tmp / "C", args.config, ["--index", *map(str, subset)], {"PYTHONHASHSEED": "12345"})
        b, c = _load(tmp / "B"), _load(tmp / "C")

    ok_ab = a == b
    ok_c = bool(c) and all(a.get(k) == v for k, v in c.items())
    print(f"runs: {n}")
    print(f"sequential == process pool          : {'YES' if ok_ab else 'NO'}")
    print(f"isolated, reversed, other hash seed : {'YES' if ok_c else 'NO'}")
    if ok_ab and ok_c:
        print("REPRODUCIBLE: every run's result depends only on its seed.")
        return 0
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            print(f"  differs: {k}: {a.get(k, (None,))[0]} vs {b.get(k, (None,))[0]}")
    print("NOT REPRODUCIBLE -- do not launch the full experiments until this is resolved.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
