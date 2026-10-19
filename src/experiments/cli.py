"""
Command-line entry point for running Phases 1-3 locally or on the CHPC.

    python -m src.experiments.cli list    --phase 1                 # how many runs, sample ids
    python -m src.experiments.cli run     --phase 1 --workers 23    # everything, 23 processes
    python -m src.experiments.cli run     --phase 1 --chunk 3 --n-chunks 10 --workers 23
    python -m src.experiments.cli run     --phase 2 --index 17      # exactly one run
    python -m src.experiments.cli status  --phase 1                 # done / missing
    python -m src.experiments.cli phase3-select                     # freeze Phase 1 winners
    python -m src.experiments.cli aggregate --phase 1               # CSV + Kruskal-Wallis/Dunn
    add --smoke to any command for a tiny end-to-end check (separate results dir).

Every run re-seeds all RNGs from its own seed (see RealFramsticksInterface.reseed), so the
result of a run does not depend on which worker executes it or in which order -- chunks
and indices can be distributed across PBS array elements freely and re-run individually.
Existing outputs are skipped unless --force is given, so resubmitting a job resumes it.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.specs import build_specs, select_best_per_task, select_tuned_parameters  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _results_dir(config: dict, smoke: bool) -> Path:
    base = Path(os.environ.get("RESULTS_DIR") or config["experiment"]["results_dir"])
    if not base.is_absolute():
        base = ROOT / base
    return base / "smoke" if smoke else base


def _selection_path(config: dict, smoke: bool) -> Path:
    return _results_dir(config, smoke) / "phase3" / "selection.json"


def _load_selection(config: dict, smoke: bool) -> tuple[dict | None, dict | None]:
    p = _selection_path(config, smoke)
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        return d.get("selection"), d.get("tuned")
    return None, None


def get_specs(phase: int, config: dict, smoke: bool):
    selection, tuned = _load_selection(config, smoke) if phase == 3 else (None, None)
    return build_specs(phase, config, smoke=smoke, selection=selection, tuned=tuned)


def _slice(specs, args):
    idx = list(range(len(specs)))
    if args.index is not None:
        idx = [i for i in args.index if 0 <= i < len(specs)]
    elif args.n_chunks:
        idx = idx[args.chunk::args.n_chunks]  # strided -> balanced mix of variants per chunk
    return idx


# ---------------------------------------------------------------------------- worker
_W = {}


def _worker_init(config_path, smoke, quiet):
    from src.framsticks_interface.interface import get_interface

    config = load_config(config_path)
    _W["config"] = config
    _W["smoke"] = smoke
    _W["interface"] = get_interface(config, quiet=quiet)
    _W["specs"] = None


def _worker_run(task):
    phase, index, results_dir, verbose = task
    from src.experiments.run_single import run_spec

    if _W["specs"] is None or _W.get("phase") != phase:
        _W["specs"] = get_specs(phase, _W["config"], _W["smoke"])
        _W["phase"] = phase
    spec = _W["specs"][index]
    t0 = time.time()
    try:
        rec = run_spec(_W["interface"], spec, _W["config"], results_dir, smoke=_W["smoke"], verbose=verbose)
        return index, spec.run_id, True, f"best={rec['best_fitness']:.5f} evals={rec['n_evaluations']} {time.time() - t0:.0f}s"
    except Exception:
        return index, spec.run_id, False, traceback.format_exc()


def cmd_run(args, config):
    specs = get_specs(args.phase, config, args.smoke)
    results_dir = _results_dir(config, args.smoke)
    todo = []
    for i in _slice(specs, args):
        if not args.force and specs[i].output_path(results_dir).exists():
            continue
        todo.append(i)
    print(f"[cli] phase {args.phase}: {len(specs)} runs in manifest, {len(todo)} to do "
          f"(workers={args.workers}, results -> {results_dir})", flush=True)
    if not todo:
        return 0
    verbose = args.workers == 1 and not args.quiet_gens
    tasks = [(args.phase, i, str(results_dir), verbose) for i in todo]
    failures = 0
    t0 = time.time()
    if args.workers == 1:
        _worker_init(args.config, args.smoke, not args.verbose_frams)
        it = map(_worker_run, tasks)
        pool = None
    else:
        ctx = mp.get_context("spawn")  # each worker initialises its own Framsticks instance
        pool = ctx.Pool(args.workers, initializer=_worker_init,
                        initargs=(args.config, args.smoke, not args.verbose_frams))
        it = pool.imap_unordered(_worker_run, tasks)
    for k, (i, run_id, ok, msg) in enumerate(it, 1):
        if ok:
            print(f"[cli] {k}/{len(tasks)} #{i} {run_id}: {msg}", flush=True)
        else:
            failures += 1
            print(f"[cli] {k}/{len(tasks)} #{i} {run_id}: FAILED\n{msg}", flush=True)
    if pool:
        pool.close()
        pool.join()
    print(f"[cli] finished {len(tasks)} runs in {time.time() - t0:.0f}s, {failures} failed", flush=True)
    return 1 if failures else 0


def cmd_list(args, config):
    specs = get_specs(args.phase, config, args.smoke)
    print(f"phase {args.phase}: {len(specs)} runs")
    for i in _slice(specs, args)[: args.show]:
        print(f"  #{i:4d} {specs[i].output_path('').as_posix()}")
    if args.manifest:
        out = Path(args.manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for i, s in enumerate(specs):
                f.write(json.dumps({"index": i, **s.to_dict()}) + "\n")
        print(f"manifest written to {out}")
    return 0


def cmd_status(args, config):
    specs = get_specs(args.phase, config, args.smoke)
    results_dir = _results_dir(config, args.smoke)
    missing = [i for i, s in enumerate(specs) if not s.output_path(results_dir).exists()]
    print(f"phase {args.phase}: {len(specs) - len(missing)}/{len(specs)} done, {len(missing)} missing")
    if missing and args.show:
        print("missing indices:", " ".join(map(str, missing[: args.show])))
    return 0


def cmd_phase3_select(args, config):
    results_dir = _results_dir(config, args.smoke)
    p1 = Path(args.phase1_results) if args.phase1_results else results_dir / "phase1"
    sel = select_best_per_task(p1)
    if not sel and config.get("phase3", {}).get("mode", "best_per_task") == "best_per_task":
        print(f"No Phase 1 results found under {p1}")
        return 1
    p2 = results_dir / "phase2"
    tuned = select_tuned_parameters(p2) if p2.exists() else {}
    out = _selection_path(config, args.smoke)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"source_phase1": str(p1), "source_phase2": str(p2) if tuned else None,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S"), "selection": sel, "tuned": tuned}, f, indent=2)
    for task, s in sel.items():
        print(f"  {task:20s} -> {s['variant']} / {s['encoding']}  (mean best {s['mean_best_fitness']:.4f}, n={s['n_runs']})")
    print(f"  tuned from Phase 2: {tuned if tuned else 'none (Phase 2 results not found -> config defaults)'}")
    print(f"selection frozen in {out}")
    return 0


def cmd_aggregate(args, config):
    from src.experiments.aggregate import aggregate_phase
    aggregate_phase(args.phase, _results_dir(config, args.smoke), alpha=config["statistics"]["alpha"])
    return 0


def cmd_best_diversity(args, config):
    """Komosinski & Mensfelt (2025) diversity comparison of the 20 best solutions per setup.
    Needs Framsticks (graph measure) and the phase's analysis/runs.csv from `aggregate`."""
    from src.evaluation.best_diversity import run_best_diversity
    from src.framsticks_interface.interface import get_interface
    iface = get_interface(config, quiet=not args.verbose_frams)
    run_best_diversity(iface, _results_dir(config, args.smoke), args.phase, alpha=config["statistics"]["alpha"])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "list", "status", "phase3-select", "aggregate", "best-diversity"])
    ap.add_argument("--phase", type=int, choices=[1, 2, 3], default=1)
    ap.add_argument("--config", default=os.environ.get("EXPERIMENT_CONFIG"))
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check (results/smoke/)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--index", type=int, nargs="+", help="run only these manifest indices")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--manifest", default=None, help="(list) also write the manifest as JSONL")
    ap.add_argument("--phase1-results", default=None)
    ap.add_argument("--quiet-gens", action="store_true", help="no per-generation lines")
    ap.add_argument("--verbose-frams", action="store_true", help="show Framsticks start-up output")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    return {"run": cmd_run, "list": cmd_list, "status": cmd_status,
            "phase3-select": cmd_phase3_select, "aggregate": cmd_aggregate,
            "best-diversity": cmd_best_diversity}[args.command](args, config)


if __name__ == "__main__":
    sys.exit(main())
