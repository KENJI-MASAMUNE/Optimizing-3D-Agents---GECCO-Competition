"""
Diversity of the best solutions, as in Komosinski & Mensfelt (GECCO 2025, Sec. 4.2.2).

For each experimental setup (task x encoding x variant [x Phase 3 setting]) take the best
genotype of each of its 20 runs and measure how different these 20 solutions are from each
other, with both measures:
  * gene  -- Levenshtein distance between genotype strings,
  * graph -- Framsticks' structural (phenotype) dissimilarity, greedy matching with the
             paper's weights (parts 0, degree 1, neurons 0.1, geometry 1; set in
             framsticks_sim/eval-allcriteria.sim).
The setup's diversity is the average pairwise dissimilarity of its 20 best genotypes. For
the statistics each genotype contributes one value, its mean dissimilarity to the other 19,
and within each task/encoding/measure a Kruskal-Wallis test across crowding variants is
followed by Dunn's test (Bonferroni), exactly as in the paper. The paper's Table 3 lists the
crowding distances that gave significantly HIGHER diversity than NSGA-II; `vs_baseline`
reports that (and significantly lower) for every variant.

This is different from the per-run `gene_diversity` / `graph_diversity` in runs.csv, which
measure the final population of a single run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE = "nsga2_baseline"
REFERENCE_ONLY = {"simple_ea"}  # reported, but not part of the crowding comparison
MEASURES = ("gene", "graph")


def _per_genotype_mean(matrix) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    n = len(m)
    return m.sum(axis=1) / (n - 1) if n > 1 else np.zeros(n)


def best_solution_diversity(interface, runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per genotype: keys..., run_id, measure values (mean dissimilarity to the other
    best genotypes of the same setup)."""
    rows = []
    for key, g in runs.dropna(subset=["best_genotype"]).groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        genos = g["best_genotype"].tolist()
        enc = dict(zip(keys, key)).get("encoding", "f1")
        per = {m: _per_genotype_mean(interface.pairwise_dissimilarity(genos, enc, m)) for m in MEASURES}
        if hasattr(interface, "clear_caches"):
            interface.clear_caches()
        for i, rid in enumerate(g["run_id"].tolist()):
            rows.append({**dict(zip(keys, key)), "run_id": rid, **{m: float(per[m][i]) for m in MEASURES}})
    return pd.DataFrame(rows)


def diversity_stats(per_geno: pd.DataFrame, keys: list[str], alpha: float = 0.05) -> dict:
    from scipy import stats
    import scikit_posthocs as sp
    group_keys = [k for k in keys if k != "variant"]
    out = {}
    for key, g in per_geno.groupby(group_keys):
        key = key if isinstance(key, tuple) else (key,)
        name = "__".join(str(k) for k in key)
        core = g[~g["variant"].isin(REFERENCE_ONLY)]
        out[name] = {}
        for m in MEASURES:
            groups = {v: x[m].values for v, x in core.groupby("variant")}
            res = {"mean": {v: float(np.mean(x)) for v, x in g.groupby("variant")[m]}}
            if len(groups) >= 2 and not all(np.allclose(x, list(groups.values())[0][0]) for x in groups.values()):
                H, p = stats.kruskal(*groups.values())
                res.update(H=float(H), p=float(p), significant=bool(p < alpha))
                if p < alpha:
                    d = sp.posthoc_dunn(core, val_col=m, group_col="variant", p_adjust="bonferroni")
                    res["dunn_bonferroni"] = {str(i): {str(j): float(d.loc[i, j]) for j in d.columns} for i in d.index}
                    if BASELINE in d.index:
                        base = res["mean"][BASELINE]
                        res["vs_baseline"] = {
                            v: {"p": float(d.loc[v, BASELINE]), "higher": bool(res["mean"][v] > base)}
                            for v in d.index if v != BASELINE and d.loc[v, BASELINE] < alpha}
            out[name][m] = res
    return out


def run_best_diversity(interface, results_dir: Path, phase: int = 1, alpha: float = 0.05) -> Path:
    out = Path(results_dir) / "analysis" / f"phase{phase}"
    runs = pd.read_csv(out / "runs.csv")
    keys = ["task", "encoding", "variant"] + (["setting"] if phase == 3 else [])
    per_geno = best_solution_diversity(interface, runs, keys)
    per_geno.to_csv(out / "best20_diversity_per_genotype.csv", index=False)
    summary = per_geno.groupby(keys)[list(MEASURES)].mean().reset_index()
    summary.to_csv(out / "best20_diversity.csv", index=False)
    st = diversity_stats(per_geno, keys, alpha)
    with open(out / "best20_diversity_stats.json", "w") as f:
        json.dump(st, f, indent=2)
    print_report(summary, st, keys)
    print(f"[best-diversity] wrote {out}/best20_diversity*.csv and best20_diversity_stats.json")
    return out


def print_report(summary: pd.DataFrame, st: dict, keys: list[str]) -> None:
    print("\nDiversity of the 20 best solutions (mean pairwise dissimilarity)\n")
    print(summary.pivot_table(index=[k for k in keys if k != "variant"], columns="variant",
                              values=list(MEASURES)).round(3).to_string())
    print("\nSignificantly different from NSGA-II (Kruskal-Wallis + Dunn/Bonferroni, as in the paper's Table 3):")
    for name, by_m in st.items():
        for m, res in by_m.items():
            vs = res.get("vs_baseline") or {}
            if vs:
                txt = ", ".join(f"{v} {'HIGHER' if r['higher'] else 'lower'} (p={r['p']:.4f})" for v, r in vs.items())
                print(f"  {name:32s} {m:5s}: {txt}")
            else:
                print(f"  {name:32s} {m:5s}: none (KW p={res.get('p', float('nan')):.4f})")
