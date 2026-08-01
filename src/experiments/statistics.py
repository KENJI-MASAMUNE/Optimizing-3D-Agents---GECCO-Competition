"""
Statistical analysis.

"Statistical significance will be assessed using the Kruskal-Wallis test followed by Dunn's
post-hoc tests (alpha = 0.05), consistent with the methodology of Komosinski and Mensfelt
(2025)." (Section 5.4)
"""
from __future__ import annotations

import json
from pathlib import Path

import scikit_posthocs as sp
from scipy import stats


def load_results(results_dir: str) -> list[dict]:
    records = []
    for path in Path(results_dir).glob("*.json"):
        with open(path) as f:
            records.append(json.load(f))
    return records


def group_by_variant(records: list[dict], metric: str = "best_fitness",
                      task: str | None = None, encoding: str | None = None) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}
    for r in records:
        if task and r["task"] != task:
            continue
        if encoding and r["encoding"] != encoding:
            continue
        groups.setdefault(r["variant"], []).append(r[metric])
    return groups


def kruskal_wallis(groups: dict[str, list[float]]) -> dict:
    values = [v for v in groups.values() if len(v) > 0]
    if len(values) < 2:
        return {"error": "Need at least 2 groups with data for Kruskal-Wallis"}
    stat, p = stats.kruskal(*values)
    return {"statistic": stat, "p_value": p, "n_groups": len(values), "groups": list(groups.keys())}


def dunn_posthoc(groups: dict[str, list[float]], alpha: float = 0.05):
    """Returns a DataFrame of pairwise p-values (Dunn's test, Bonferroni-corrected) between
    every pair of variant groups."""
    import pandas as pd

    labels, values = [], []
    for name, vals in groups.items():
        labels.extend([name] * len(vals))
        values.extend(vals)
    df = pd.DataFrame({"group": labels, "value": values})
    result = sp.posthoc_dunn(df, val_col="value", group_col="group", p_adjust="bonferroni")
    return result


def analyse(results_dir: str, metric: str = "best_fitness", alpha: float = 0.05) -> dict:
    records = load_results(results_dir)
    tasks = sorted({r["task"] for r in records})
    encodings = sorted({r["encoding"] for r in records})

    report = {}
    for task in tasks:
        for encoding in encodings:
            groups = group_by_variant(records, metric=metric, task=task, encoding=encoding)
            if len(groups) < 2:
                continue
            kw = kruskal_wallis(groups)
            entry = {"kruskal_wallis": kw}
            if "p_value" in kw and kw["p_value"] < alpha:
                dunn = dunn_posthoc(groups, alpha=alpha)
                entry["dunn_posthoc"] = dunn.to_dict()
                entry["significant"] = True
            else:
                entry["significant"] = False
            report[f"{task}__{encoding}"] = entry
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kruskal-Wallis + Dunn's post-hoc analysis")
    parser.add_argument("--results", required=True, help="Path to a results/phaseN directory")
    parser.add_argument("--metric", default="best_fitness")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    report = analyse(args.results, metric=args.metric, alpha=args.alpha)
    for key, entry in report.items():
        kw = entry["kruskal_wallis"]
        sig = "SIGNIFICANT" if entry.get("significant") else "not significant"
        print(f"{key}: H={kw.get('statistic', float('nan')):.3f} "
              f"p={kw.get('p_value', float('nan')):.4f} -> {sig}")
