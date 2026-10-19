"""
Aggregation + statistics for each phase (writes CSV/JSON into <results>/analysis/phaseN/).

Phase 1: summary per (task, encoding, variant); Kruskal-Wallis + Dunn (Bonferroni) across
         variants for best fitness, gene/graph diversity and convergence; mean convergence
         curves per generation.
Phase 2: summary per sweep point; Kruskal-Wallis across the levels of each swept factor.
Phase 3: per-setting summary, competition-style normalised score (mean best fitness per
         setting divided by the best mean in that setting, then averaged over settings),
         generalisation score/retention.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCALAR_COLS = ["phase", "run_id", "variant", "encoding", "task", "repeat", "seed", "group", "world",
               "setting", "best_fitness", "gene_diversity", "graph_diversity", "convergence_evals",
               "convergence_evals_improvement", "n_evaluations", "n_cache_hits", "stopped_by_budget",
               "wall_time_s", "generalisation_score", "generalisation_retention", "best_genotype"]


def load_phase(results_dir: Path, phase: int) -> tuple[pd.DataFrame, list[dict]]:
    rows, raw = [], []
    for p in sorted((results_dir / f"phase{phase}").rglob("*.json")):
        if p.name == "selection.json":
            continue
        with open(p) as f:
            r = json.load(f)
        if "variant" not in r or "run_id" not in r:
            continue  # skip pre-audit records (old format, produced by the buggy pipeline)
        raw.append(r)
        row = {k: r.get(k) for k in SCALAR_COLS}
        for k, v in (r.get("sweep") or {}).items():
            row[f"sweep_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows), raw


def _kw_dunn(df: pd.DataFrame, group_col: str, value_col: str, alpha: float) -> dict:
    from scipy import stats
    groups = {k: g[value_col].dropna().values for k, g in df.groupby(group_col)}
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    out = {"metric": value_col, "groups": {str(k): int(len(v)) for k, v in groups.items()}}
    if len(groups) < 2:
        out["error"] = "fewer than 2 groups"
        return out
    vals = list(groups.values())
    if all(np.allclose(v, vals[0][0]) for v in vals):
        out.update(H=0.0, p=1.0, significant=False, note="all values identical")
        return out
    H, p = stats.kruskal(*vals)
    out.update(H=float(H), p=float(p), significant=bool(p < alpha))
    if p < alpha:
        import scikit_posthocs as sp
        d = sp.posthoc_dunn(df.dropna(subset=[value_col]), val_col=value_col, group_col=group_col,
                            p_adjust="bonferroni")
        out["dunn_bonferroni"] = {str(i): {str(j): float(d.loc[i, j]) for j in d.columns} for i in d.index}
    return out


def _summary(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    agg = df.groupby(by)[metrics].agg(["mean", "std", "median", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def _curves(raw: list[dict], by: list[str]) -> pd.DataFrame:
    rows = []
    for r in raw:
        for h in r.get("history", []):
            rows.append({**{k: r.get(k) for k in by}, "generation": h["generation"],
                         "n_evaluations": h["n_evaluations"], "best_fitness": h["best_fitness"],
                         "mean_fitness": h["mean_fitness"], "mean_diversity": h["mean_diversity"]})
    if not rows:
        return pd.DataFrame()
    c = pd.DataFrame(rows)
    return c.groupby(by + ["generation"]).agg(
        best_fitness_mean=("best_fitness", "mean"), best_fitness_std=("best_fitness", "std"),
        mean_fitness_mean=("mean_fitness", "mean"), mean_diversity_mean=("mean_diversity", "mean"),
        n_evaluations_mean=("n_evaluations", "mean"), n_runs=("best_fitness", "count")).reset_index()


BASELINE = "simple_ea"


def competition_scores(summ: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """GECCO competition normalisation (2026 slides): within each group (a task, or a
    Phase 3 setting) the organisers' Simple EA scores 0 and the best entry 1:
        score = (mean - mean_SimpleEA) / (best mean - mean_SimpleEA).
    Entries = (variant, encoding). The Simple EA reference is the better of its encodings
    in that group, so an algorithm cannot look good just because the f1 Simple EA is weak."""
    out = []
    for key, g in summ.groupby(by):
        ref = g.loc[g["variant"] == BASELINE, "best_fitness_mean"]
        if ref.empty:
            continue
        lo, hi = ref.max(), g["best_fitness_mean"].max()
        g = g.assign(competition_score=(g["best_fitness_mean"] - lo) / (hi - lo) if hi > lo else np.nan)
        out.append(g)
    return pd.concat(out) if out else pd.DataFrame()


def _vs_baseline(df: pd.DataFrame, alpha: float) -> dict:
    """Each algorithm vs the Simple EA (Mann-Whitney U, two-sided, Bonferroni over algorithms)."""
    from scipy import stats
    base = df.loc[df["variant"] == BASELINE, "best_fitness"].dropna().values
    others = [v for v in sorted(df["variant"].unique()) if v != BASELINE]
    res = {}
    if len(base) == 0:
        return res
    for v in others:
        x = df.loc[df["variant"] == v, "best_fitness"].dropna().values
        if len(x) == 0:
            continue
        u, p = stats.mannwhitneyu(x, base, alternative="two-sided")
        p_adj = min(1.0, p * len(others))
        res[v] = {"mean": float(x.mean()), "simple_ea_mean": float(base.mean()), "U": float(u),
                  "p_bonferroni": p_adj, "better": bool(x.mean() > base.mean()), "significant": bool(p_adj < alpha)}
    return res


def aggregate_phase(phase: int, results_dir: Path, alpha: float = 0.05) -> Path:
    results_dir = Path(results_dir)
    df, raw = load_phase(results_dir, phase)
    out = results_dir / "analysis" / f"phase{phase}"
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print(f"[aggregate] no phase {phase} results under {results_dir}")
        return out
    df.to_csv(out / "runs.csv", index=False)
    metrics = ["best_fitness", "gene_diversity", "graph_diversity", "convergence_evals_improvement", "wall_time_s"]
    report: dict = {"n_runs": int(len(df))}

    if phase == 1:
        _summary(df, ["task", "encoding", "variant"], metrics).to_csv(out / "summary.csv", index=False)
        _curves(raw, ["task", "encoding", "variant"]).to_csv(out / "convergence_curves.csv", index=False)
        summ = _summary(df, ["task", "encoding", "variant"], metrics)
        comp = competition_scores(summ, ["task"])
        if not comp.empty:
            comp[["task", "encoding", "variant", "best_fitness_mean", "competition_score"]].to_csv(
                out / "competition_scores.csv", index=False)
        stats, vs_simple = {}, {}
        for (task, enc), g in df.groupby(["task", "encoding"]):
            # the crowding comparison (research question) is among the NSGA-II variants only;
            # the Simple EA is tested separately against each of them
            core = g[g["variant"] != BASELINE]
            stats[f"{task}__{enc}"] = {m: _kw_dunn(core, "variant", m, alpha) for m in metrics[:4]}
            vs_simple[f"{task}__{enc}"] = _vs_baseline(g, alpha)
        report["vs_simple_ea"] = vs_simple
        # encoding effect per (task, variant): Mann-Whitney via KW with 2 groups
        enc_stats = {}
        for (task, var), g in df.groupby(["task", "variant"]):  # includes simple_ea
            enc_stats[f"{task}__{var}"] = _kw_dunn(g, "encoding", "best_fitness", alpha)
        report.update(variant_tests=stats, encoding_tests=enc_stats)
        if not comp.empty:
            print("\ncompetition score per task (Simple EA = 0, best = 1):")
            print(comp.pivot_table(index=["variant", "encoding"], columns="task", values="competition_score").round(3).to_string())
            print()
        for key, st in stats.items():
            bf = st["best_fitness"]
            print(f"{key:35s} best_fitness: H={bf.get('H', float('nan')):.3f} p={bf.get('p', float('nan')):.4f} "
                  f"{'SIGNIFICANT' if bf.get('significant') else 'n.s.'}")
            vs = vs_simple.get(key) or {}
            if vs:
                better = [v for v, t in vs.items() if t["significant"] and t["better"]]
                worse = [v for v, t in vs.items() if t["significant"] and not t["better"]]
                print(f"{'':35s} vs Simple EA: better {better or '-'}  worse {worse or '-'}")

    elif phase == 2:
        sweep_cols = [c for c in df.columns if c.startswith("sweep_")]
        tests = {}
        for grp, g in df.groupby("group"):
            cols = [c for c in sweep_cols if g[c].notna().any()]
            _summary(g, ["task", "encoding"] + cols, metrics).to_csv(out / f"summary_{grp}.csv", index=False)
            for (task, enc), gg in g.groupby(["task", "encoding"]):
                for c in cols:
                    tests[f"{grp}__{task}__{enc}__{c}"] = _kw_dunn(gg, c, "best_fitness", alpha)
                if len(cols) > 1:
                    gg = gg.assign(_combo=gg[cols].astype(str).agg("|".join, axis=1))
                    tests[f"{grp}__{task}__{enc}__combined"] = _kw_dunn(gg, "_combo", "best_fitness", alpha)
        report["sweep_tests"] = tests
        for k, t in tests.items():
            print(f"{k:60s} H={t.get('H', float('nan')):.3f} p={t.get('p', float('nan')):.4f} "
                  f"{'SIGNIFICANT' if t.get('significant') else 'n.s.'}")

    elif phase == 3:
        gen_metrics = metrics + ["generalisation_score", "generalisation_retention"]
        summ = _summary(df, ["setting", "task", "world", "variant", "encoding"], gen_metrics)
        summ["normalised_score"] = summ.groupby("setting")["best_fitness_mean"].transform(
            lambda s: s / s.max() if s.max() > 0 else s * np.nan)
        comp = competition_scores(summ, ["setting"])
        if not comp.empty:
            summ = summ.merge(comp[["setting", "variant", "encoding", "competition_score"]],
                              on=["setting", "variant", "encoding"], how="left")
        summ.to_csv(out / "summary.csv", index=False)
        overall = summ.groupby(["variant", "encoding"]).agg(
            settings=("setting", "nunique"), normalised_score=("normalised_score", "mean"),
            **({"competition_score": ("competition_score", "mean")} if "competition_score" in summ else {}),
            generalisation_score=("generalisation_score_mean", "mean"),
            generalisation_retention=("generalisation_retention_mean", "mean")).reset_index()
        overall.to_csv(out / "overall_scores.csv", index=False)
        tests = {}
        for setting, g in df.groupby("setting"):
            g = g.assign(_alg=g["variant"] + "/" + g["encoding"])
            tests[setting] = {m: _kw_dunn(g, "_alg", m, alpha) for m in ["best_fitness", "generalisation_score"]}
        report["setting_tests"] = tests
        report["vs_simple_ea"] = {f"{st}__{enc}": _vs_baseline(g, alpha) for (st, enc), g in df.groupby(["setting", "encoding"])}
        print(overall.to_string(index=False))

    with open(out / "stats.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[aggregate] wrote {out}")
    return out
