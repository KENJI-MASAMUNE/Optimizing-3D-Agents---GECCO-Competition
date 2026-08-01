"""
Shared configuration loading and run bookkeeping utilities.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "default_config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML experiment configuration."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class RunRecord:
    """A single (variant, encoding, task, repeat) result record."""

    variant: str
    encoding: str
    task: str
    repeat: int
    seed: int
    best_fitness: float
    gene_diversity: float
    graph_diversity: float
    convergence_evals: int
    n_evaluations: int
    wall_time_s: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d


def write_run_record(record: RunRecord, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{record.variant}__{record.encoding}__{record.task}__r{record.repeat}_s{record.seed}.json"
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)
    return out_path


def timer():
    return time.perf_counter()
