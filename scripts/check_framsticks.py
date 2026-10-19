#!/usr/bin/env python3
"""
Environment check for a new machine (laptop, CHPC login or compute node):
  1. frams-objects library loads and the configured .sim files import,
  2. a random genotype can be generated, mutated, crossed over and evaluated,
  3. every Phase 3 world applies and restores cleanly,
  4. (--timing) seconds per evaluation and per dissimilarity matrix, and an estimate of the
     CPU time of one full-size run and of each phase.

    python scripts/check_framsticks.py [--config configs/chpc_config.yaml] [--timing]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    config = load_config(args.config)

    from src.framsticks_interface.interface import get_interface
    from src.framsticks_interface.worlds import load_worlds
    from src.experiments.specs import build_specs

    t0 = time.perf_counter()
    iface = get_interface(config)
    print(f"[check] Framsticks initialised in {time.perf_counter() - t0:.1f}s")
    iface.reseed(1)
    g = {enc: [iface.get_random_genotype(enc) for _ in range(10)] for enc in ("f0", "f1")}
    m = iface.mutate(g["f1"][0], "f1")
    c1, _ = iface.crossover(g["f1"][0], g["f1"][1], "f1")
    r = iface.evaluate(g["f1"][0], "f1", "straight_line_cog")
    print(f"[check] genotype ops OK; evaluation valid={r.is_valid} fitness={r.fitness:.4f} "
          f"trajectory steps={len(r.cog_trajectory)}")
    if not r.is_valid or len(r.cog_trajectory) < 10:
        print("[check] FAILED: evaluation did not return a COG trajectory -- check sim_files")
        return 1

    worlds, settings, held_out = load_worlds((config.get("phase3") or {}).get("worlds_file"))
    base = iface.evaluate(g["f1"][1], "f1", "straight_line_cog").cog_trajectory
    for name in worlds:
        iface.evaluate(g["f1"][1], "f1", "straight_line_cog", name)
    again = iface.evaluate(g["f1"][1], "f1", "straight_line_cog", None).cog_trajectory
    print(f"[check] {len(worlds)} worlds applied; base world restored exactly: {base == again}")
    if base != again:
        return 1

    if args.timing:
        iface.reset_evaluation_budget()
        for enc in ("f0", "f1"):
            t0 = time.perf_counter()
            for x in g[enc]:
                iface.evaluate(x, enc, "straight_line_cog")
            per_eval = (time.perf_counter() - t0) / len(g[enc])
            pop = [iface.get_random_genotype(enc) for _ in range(40)]
            t0 = time.perf_counter()
            iface.pairwise_dissimilarity(pop, enc, "gene")
            t_gene = time.perf_counter() - t0
            iface.clear_caches()
            t0 = time.perf_counter()
            iface.pairwise_dissimilarity(pop, enc, "graph")
            t_graph = time.perf_counter() - t0
            print(f"[timing] {enc}: {per_eval:.3f}s/eval (random genotypes), 40x40 gene matrix {t_gene:.3f}s, "
                  f"graph matrix {t_graph:.3f}s")
        ga = config["ga"]
        evals = ga["population_size"] * ga["n_generations"]
        print(f"[timing] one full run = {evals} evaluations -> roughly {evals * per_eval / 60:.0f} min of "
              "simulation (evolved creatures are often larger/slower than random ones; graph variants add "
              "dissimilarity time). See docs/PHASES_AND_CHPC.md for measured numbers.")
        for ph in (1, 2):
            print(f"[timing] phase {ph}: {len(build_specs(ph, config))} runs")
    print("[check] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
