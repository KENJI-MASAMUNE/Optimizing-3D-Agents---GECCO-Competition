#!/usr/bin/env python3
"""
Automates the Energy0 comparison you've been doing by hand: for each Energy0 value,
generates a small override .sim file (never touches your real eval-allcriteria.sim),
runs N independent repeats -- each a SEPARATE subprocess launch of
verify_reproducibility.py, avoiding calling frams.init() twice in one process -- and
prints a summary table with mean/min/max/stdev per Energy0 value.

IMPORTANT: each repeat also gets a DIFFERENT native random_seed (base + repeat index).
Without this, since frams.Math.seed() is fixed per config, all "repeats" of the same
Energy0 value would start from an IDENTICAL initial population (defeating the point of
having multiple repeats) -- see the reproducibility investigation earlier in this project.

Usage:
    python scripts/energy0_sweep.py --energy0-values 2500 5000 --repeats 3
    python scripts/energy0_sweep.py --energy0-values 2500 5000 10000 --repeats 5 --population-size 30 --generations 15
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.utils.config import load_config  # noqa: E402

ENERGY0_PATTERN = re.compile(
    r"^([ \t]*Energy0[ \t]*[:=][ \t]*)([-+]?(?:\d+(?:\.\d*)?|\.\d+))([ \t]*)$",
    re.IGNORECASE | re.MULTILINE,
)


def make_energy0_override_file(frams_data_dir: Path, base_sim_name: str, energy0_value: float) -> str:
    """Reads the real eval-allcriteria.sim, substitutes ONLY the Energy0 line's numeric
    value (preserving all other formatting exactly), and writes the result to a NEW file
    -- the original is never modified. Returns the override file's name (not full path;
    Framsticks resolves .sim filenames relative to the data/ directory)."""
    base_path = frams_data_dir / base_sim_name
    content = base_path.read_text()

    match = ENERGY0_PATTERN.search(content)
    if not match:
        raise RuntimeError(
            f"Could not find an 'Energy0 = <number>' line in {base_path}. "
            f"Open the file and confirm the exact format, then adjust ENERGY0_PATTERN "
            f"in this script to match it."
        )

    # Preserve the key and separator formatting exactly; replace ONLY the number;
    # explicitly strip trailing whitespace after the number (this is what caused the
    # 'ExtValue.parseDouble: Could not parse' error earlier in this project).
    new_line = f"{match.group(1)}{energy0_value}"
    new_content = content[:match.start()] + new_line + content[match.end():]

    energy0_str = f"{energy0_value:g}".replace(".", "_")  # e.g. 2500.0 -> "2500"
    override_name = f"_energy0_sweep_{energy0_str}.sim"
    override_path = frams_data_dir / override_name
    override_path.write_text(new_content)
    return override_name


def make_repeat_config(base_config: dict, base_config_path: str, override_sim_name: str,
                        repeat_seed: int, out_path: Path) -> None:
    """Writes a temp config YAML: same as base_config, but with the Energy0 override .sim
    file swapped in (in place of eval-allcriteria.sim -- loaded in the same position, so
    everything else about the settings stays identical) and a per-repeat random_seed."""
    config = dict(base_config)
    fconf = dict(config["framsticks"])

    original_sim_files = fconf.get("sim_files", "eval-allcriteria.sim;deterministic.sim;recording-body-coords.sim")
    sim_list = original_sim_files.split(";")
    sim_list[0] = override_sim_name  # assumes eval-allcriteria.sim is first, matching the default
    fconf["sim_files"] = ";".join(sim_list)
    # framsticks.base_world (added Oct 2026) is applied AFTER the .sim files and would
    # silently override the swept Energy0 -- drop its energy0 so the override file wins.
    if fconf.get("base_world"):
        fconf["base_world"] = {k: v for k, v in fconf["base_world"].items() if k != "energy0"}
    fconf["random_seed"] = repeat_seed

    config["framsticks"] = fconf
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(config, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Base config (default: configs/default_config.yaml)")
    parser.add_argument("--energy0-values", type=float, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--eval-budget", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=100,
                         help="First repeat's native random_seed; repeat i uses seed_base + i")
    parser.add_argument("--results-dir", default="results/energy0_sweep")
    args = parser.parse_args()

    base_config = load_config(args.config)
    base_config_path = args.config
    frams_data_dir = Path(base_config["framsticks"]["frams_dir"]) / "data"
    original_sim_files = base_config["framsticks"].get(
        "sim_files", "eval-allcriteria.sim;deterministic.sim;recording-body-coords.sim")
    base_sim_name = original_sim_files.split(";")[0]

    eval_budget = args.eval_budget or max(base_config["framsticks"]["eval_budget"],
                                           args.population_size * args.generations * 2)

    results_dir = Path(args.results_dir)
    configs_dir = Path("configs/_energy0_sweep")
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[float, list[float]] = {e: [] for e in args.energy0_values}
    total_runs = len(args.energy0_values) * args.repeats
    run_num = 0

    for energy0 in args.energy0_values:
        print(f"\n{'=' * 60}\nEnergy0 = {energy0}\n{'=' * 60}")
        override_sim_name = make_energy0_override_file(frams_data_dir, base_sim_name, energy0)
        print(f"Override .sim file: {override_sim_name} (original {base_sim_name} untouched)")

        for repeat in range(args.repeats):
            run_num += 1
            repeat_seed = args.seed_base + repeat
            repeat_config_path = configs_dir / f"energy0_{energy0}_repeat{repeat}.yaml"
            make_repeat_config(base_config, base_config_path, override_sim_name, repeat_seed, repeat_config_path)

            save_path = results_dir / f"energy0_{energy0}_repeat{repeat}.json"
            print(f"\n[{run_num}/{total_runs}] Energy0={energy0} repeat={repeat} (native_seed={repeat_seed})")

            subprocess.run([
                sys.executable, "scripts/verify_reproducibility.py",
                "--config", str(repeat_config_path),
                "--population-size", str(args.population_size),
                "--generations", str(args.generations),
                "--eval-budget", str(eval_budget),
                "--seed", str(repeat),
                "--save", str(save_path),
            ], check=True)

            with open(save_path) as f:
                data = json.load(f)
            final_best = data["history"][-1]["best_fitness"]
            all_results[energy0].append(final_best)

    print(f"\n\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"{'Energy0':>10}  {'n':>3}  {'mean':>8}  {'stdev':>8}  {'min':>8}  {'max':>8}  {'all values'}")
    for energy0 in args.energy0_values:
        values = all_results[energy0]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{energy0:10.0f}  {len(values):3d}  {mean:8.4f}  {stdev:8.4f}  "
              f"{min(values):8.4f}  {max(values):8.4f}  {[round(v, 4) for v in values]}")

    print("\nInterpretation: if the (mean +/- stdev) ranges overlap heavily across Energy0 "
          "values, the differences you're seeing are more likely noise than a genuine "
          "Energy0 effect -- more repeats would help separate signal from noise before "
          "drawing conclusions. If ranges separate cleanly, that's a real, reportable effect.")

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nRaw results also saved to {summary_path}")


if __name__ == "__main__":
    main()
