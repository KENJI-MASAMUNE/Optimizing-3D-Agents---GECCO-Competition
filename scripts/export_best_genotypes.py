#!/usr/bin/env python3
"""
Export the best evolved creatures to a Framsticks .gen file so they can be watched in the
Framsticks GUI (Framsticks55/Framsticks.exe -> File > Open... -> pick the .gen file).

For every (task, encoding, variant) it takes the run with the highest best_fitness from the
aggregated runs.csv (written by `cli aggregate`), and names each genotype so it is easy to
find in the GUI's genotype list, e.g.  "vel f1 baseline 65.690 r7".

    python scripts/export_best_genotypes.py results/analysis/phase1/runs.csv results/analysis/best_phase1.gen
    python scripts/export_best_genotypes.py runs.csv out.gen --top 3          # 3 best runs per group
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


SHORT = {"straight_line_cog": "line", "sinusoidal_bounce": "sine", "velocity_forward": "vel",
         "jump_height": "jump", "nsga2_baseline": "baseline", "de_nsga2_gene": "gene",
         "de_nsga2_graph": "graph", "de_nsga2_hybrid": "hybrid", "adaptive_de_nsga2": "adaptive"}


def gen_record(name: str, genotype: str, info: str) -> str:
    """One `org:` object in Framsticks file format. Multi-line values use the ~...~ syntax;
    a literal tilde inside a value must be escaped as \\~."""
    esc = lambda s: s.replace("\\", "\\\\").replace("~", "\\~")  # noqa: E731
    g = genotype.strip("\n")
    return f"org:\nname:{name}\ninfo:~\n{esc(info)}~\ngenotype:~\n{esc(g)}~\n\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_csv", type=Path)
    ap.add_argument("out_gen", type=Path)
    ap.add_argument("--top", type=int, default=1, help="best N runs per (task, encoding, variant)")
    args = ap.parse_args()

    groups: dict[tuple, list[dict]] = defaultdict(list)
    with open(args.runs_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("best_genotype") or row.get("best_fitness") in ("", "nan", None):
                continue
            key = (row["task"], row["encoding"], row["variant"], row.get("setting") or "")
            groups[key].append(row)

    out = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: float(r["best_fitness"]), reverse=True)[: args.top]
        for r in rows:
            task, enc, var, setting = key
            fit = float(r["best_fitness"])
            # Framsticks keeps only the first 40 characters of a name, so use short labels
            name = f"{SHORT.get(task, task)} {enc} {SHORT.get(var, var)} {fit:.3f} r{r['repeat']}" + (f" {setting[:3]}" if setting else "")
            info = (f"phase {r.get('phase')}  run_id {r['run_id']}\nseed {r['seed']}  best_fitness {fit}\n"
                    f"evaluations {r.get('n_evaluations')}  world {r.get('world') or 'base'}")
            out.append(gen_record(name, r["best_genotype"], info))

    args.out_gen.parent.mkdir(parents=True, exist_ok=True)
    args.out_gen.write_text("".join(out), encoding="utf-8")
    print(f"wrote {len(out)} genotypes to {args.out_gen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
