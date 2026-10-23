#!/usr/bin/env python3
"""
Animate the best evolved agent of every task as a GIF (used for the showcase repository and
the Poster Day QR code).

For each task in an aggregated runs.csv, the task's winning configuration (algorithm + encoding
with the highest mean best fitness, Simple EA excluded, as chosen by phase3-select) is found and
its best run is replayed in
Framsticks with `recording-body-coords.sim` (STORE_ALL_PART_COORDS = 1), and its body (parts
joined by sticks) and centre-of-gravity trail are drawn with matplotlib, one panel per task.

    python scripts/render_best_agents.py results/analysis/phase1/runs.csv best_agents.gif \
        --framspy /path/to/framspy --frams /path/to/Framsticks55

Notes
* The replay uses the reference world with Energy0 = 2500, as in Phase 1. Replayed fitness can
  differ slightly from the logged value (simulator state carried between evaluations), so the
  animation shows behaviour, not exact scores.
* Needs Framsticks' recording-body-coords.sim in <frams>/data (it ships with Framsticks 5.5).
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

NAMES = {"nsga2_baseline": "NSGA-II", "de_nsga2_gene": "DE gene", "de_nsga2_graph": "DE graph",
         "de_nsga2_hybrid": "DE hybrid", "adaptive_de_nsga2": "Adaptive", "simple_ea": "Simple EA"}
TASKS = {"straight_line_cog": "Straight line", "sinusoidal_bounce": "Sinusoid bounce",
         "velocity_forward": "Forward velocity", "jump_height": "Jump height",
         "comp_distance": "Distance (fn 3)", "comp_tall_runner": "Tall runner (fn 4)"}


def replay(lib, frams, genotype: str):
    rec = lib.evaluate([genotype])[0]["evaluations"][""]["data->bodyrecording"]
    cog = np.array([step[0] for step in rec])
    parts = [np.array(step[1:]) for step in rec]
    m = frams.Model.newFromString(genotype)
    joints = [(m.getJoint(i).p1._int(), m.getJoint(i).p2._int()) for i in range(m.numjoints._int())]
    return cog, parts, joints


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_csv")
    ap.add_argument("out_gif")
    ap.add_argument("--framspy", default=os.environ.get("FRAMSPY_PATH"))
    ap.add_argument("--frams", default=os.environ.get("FRAMS_DIR"))
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--energy0", type=float, default=2500)
    ap.add_argument("--select", choices=["winner", "single"], default="winner",
                    help="winner: best run of each task's winning configuration (default); "
                         "single: the single best run of any algorithm")
    a = ap.parse_args()

    sys.path.insert(0, a.framspy)
    from FramsticksLib import FramsticksLib
    lib = FramsticksLib(a.frams, None, "eval-allcriteria.sim;deterministic.sim;recording-body-coords.sim")
    import frams
    frams.ExpProperties.Energy0 = a.energy0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    runs = pd.read_csv(a.runs_csv)
    if a.select == "single":
        best = runs.loc[runs.groupby("task").best_fitness.idxmax()].set_index("task")
    else:
        cand = runs[runs.variant != "simple_ea"]
        means = cand.groupby(["task", "variant", "encoding"]).best_fitness.mean().reset_index()
        win = means.loc[means.groupby("task").best_fitness.idxmax()]
        best = pd.concat([cand[(cand.task == w.task) & (cand.variant == w.variant) & (cand.encoding == w.encoding)]
                          .nlargest(1, "best_fitness") for _, w in win.iterrows()]).set_index("task")
    order = [t for t in TASKS if t in best.index]
    data = {t: (*replay(lib, frams, best.loc[t, "best_genotype"]), best.loc[t]) for t in order}

    steps = len(next(iter(data.values()))[0])
    frames = []
    for k in np.linspace(0, steps - 1, a.frames).astype(int):
        fig = plt.figure(figsize=(9, 6.2), dpi=90)
        title = "Each task's winning algorithm: its best agent" if a.select == "winner" else "Best evolved agent per task"
        fig.suptitle(f"{title}   step {k + 1}/{steps}", fontsize=11, fontweight="bold")
        for i, t in enumerate(order):
            cog, parts, joints, row = data[t]
            P, c, R = parts[k], cog[k], 2.2
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            ax.set_xlim(c[0] - R, c[0] + R); ax.set_ylim(c[1] - R, c[1] + R); ax.set_zlim(0, 2 * R * 0.6)
            for x in np.arange(np.floor(c[0] - R), np.ceil(c[0] + R) + 1):
                ax.plot([x, x], [c[1] - R, c[1] + R], [0, 0], color="#cccccc", lw=0.5)
            for y in np.arange(np.floor(c[1] - R), np.ceil(c[1] + R) + 1):
                ax.plot([c[0] - R, c[0] + R], [y, y], [0, 0], color="#cccccc", lw=0.5)
            trail = cog[max(0, k - 400):k + 1].copy()
            inside = (np.abs(trail[:, 0] - c[0]) < R) & (np.abs(trail[:, 1] - c[1]) < R) & (trail[:, 2] < 2 * R * 0.6)
            trail[~inside] = np.nan
            ax.plot(trail[:, 0], trail[:, 1], np.maximum(trail[:, 2], 0), color="#d62728", lw=1.2)
            for p1, p2 in joints:
                ax.plot(*zip(P[p1], P[p2]), color="#1f3b6e", lw=3)
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], color="#4472c4", s=12)
            ax.set_title(f"{TASKS[t]}\n{NAMES.get(row.variant, row.variant)} {row.encoding}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=22, azim=-60); ax.set_box_aspect((1, 1, 0.6))
        fig.text(0.5, 0.015, "Blue: body (sticks), red: recent centre-of-gravity trail (view follows the agent). "
                 f"Replayed in Framsticks (Energy0 = {a.energy0:g}).", ha="center", fontsize=8, color="#444")
        fig.subplots_adjust(left=0, right=1, top=0.88, bottom=0.05, wspace=0, hspace=0.15)
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=64))
    frames[0].save(a.out_gif, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
    print(f"wrote {a.out_gif} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
