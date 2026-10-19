"""
World configurations ("environments") for Phase 3, and how they are applied to Framsticks.

Background -- what the competition actually varies
---------------------------------------------------
The organisers never publish the exact 10 evaluation settings (they are only revealed when
submissions are scored), but they DO publish which simulation parameters may change:

* GECCO 2023 / 2024 competition pages: "lifespan of a creature ('starting energy' and
  'idle metabolism'), world size and heightfield (map), water level, gravity, and creature
  body constraints (minimal and maximal joint length)".
* GECCO 2025 call: "gravity, water level, terrain, and initial agent rotation".

Each setting = one world configuration x one fitness function (COG path), run 20 times,
best fitness averaged, normalised against all submissions, and the 10 normalised values
averaged. `configs/worlds/phase3_worlds.yaml` reconstructs a suite of 10 such settings
plus extra held-out worlds (never used for optimisation) for the Generalisation Score,
using ONLY the parameters listed above, with values chosen to be plausible variations of
the standard evaluation world (see the YAML file for per-setting rationale).

How a world is applied
----------------------
Every parameter maps 1:1 onto a Framsticks property (names confirmed against the
Framsticks 5.5 binary with `_propFind`):

    yaml key        Framsticks property             meaning
    world_type      World.wrldtyp                   0 flat, 1 blocks, 2 heightfield
    world_size      World.wrldsiz                   side length of the square world
    world_map       World.wrldmap                   map string for blocks/heightfield
    water_level     World.wrldwat                   -1 = no water, >0 = water depth
    gravity         World.wrldg                     1.0 = "official" setting
    boundaries      World.wrldbnd                   0 none, 1 fence, 2 teleport
    energy0         ExpProperties.Energy0           starting energy -> lifespan (steps)
    e_meta          ExpProperties.e_meta            idle metabolism (energy per step)
    rotation        ExpProperties.rotation          0 = always 0 deg (others are random)
    creation_height ExpProperties.creath            spawn height above surface
    min_joint       CreatureSettings.minjoint       body constraint
    max_joint       CreatureSettings.maxjoint       body constraint

Changes take effect on the next `Simulator.init()`, which `FramsticksLib.evaluate()` calls
for every evaluation -- verified on the real simulator (gravity/water/terrain changes alter
the recorded COG path, and reverting restores the original path exactly).

`target_heading_deg` is NOT a Framsticks property: it rotates the desired direction of
travel for the direction-dependent fitness functions (straight_line_cog,
velocity_forward). Because physics on flat ground is isotropic, asking for travel at
heading h with the creature born facing 0 deg is equivalent to the creature being born
rotated by -h. That reproduces the 2025 "initial agent rotation" variation while keeping
evaluation deterministic (Framsticks' own rotation options other than 0 are random).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WORLDS_FILE = ROOT_DIR / "configs" / "worlds" / "phase3_worlds.yaml"

# yaml key -> (Framsticks object name, property id)
FRAMS_PROPERTY_MAP: dict[str, tuple[str, str]] = {
    "world_type": ("World", "wrldtyp"),
    "world_size": ("World", "wrldsiz"),
    "world_map": ("World", "wrldmap"),
    "water_level": ("World", "wrldwat"),
    "gravity": ("World", "wrldg"),
    "boundaries": ("World", "wrldbnd"),
    "energy0": ("ExpProperties", "Energy0"),
    "e_meta": ("ExpProperties", "e_meta"),
    "rotation": ("ExpProperties", "rotation"),
    "creation_height": ("ExpProperties", "creath"),
    "min_joint": ("CreatureSettings", "minjoint"),
    "max_joint": ("CreatureSettings", "maxjoint"),
}

# Ranges reported by the Framsticks 5.5 property metadata (used for validation only).
_RANGES = {
    "world_type": (0, 2), "world_size": (10, 10000), "water_level": (-20, 200),
    "gravity": (0, 10), "boundaries": (0, 2), "energy0": (0, 100000), "e_meta": (0, 1),
    "rotation": (0, 4), "creation_height": (-1, 50), "min_joint": (0, 100), "max_joint": (0, 100),
}

TASK_PARAM_KEYS = ("target_heading_deg",)


@dataclass(frozen=True)
class WorldConfig:
    name: str
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)       # Framsticks-level parameters
    task_params: dict[str, Any] = field(default_factory=dict)  # fitness-level parameters

    def key(self) -> tuple:
        """Hashable identity used to skip re-applying an unchanged world."""
        return (tuple(sorted(self.params.items())), tuple(sorted(self.task_params.items())))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                **self.params, **self.task_params}


@dataclass(frozen=True)
class Setting:
    """One competition-style setting: a world plus a fitness function (task)."""
    name: str
    world: str
    task: str
    rationale: str = ""


def world_from_dict(name: str, d: dict[str, Any]) -> WorldConfig:
    d = dict(d or {})
    description = d.pop("description", "")
    params, task_params = {}, {}
    for k, v in d.items():
        if k in FRAMS_PROPERTY_MAP:
            lo_hi = _RANGES.get(k)
            if lo_hi is not None and not (lo_hi[0] <= float(v) <= lo_hi[1]):
                raise ValueError(f"World '{name}': {k}={v} outside Framsticks range {lo_hi}")
            params[k] = v
        elif k in TASK_PARAM_KEYS:
            task_params[k] = float(v)
        else:
            raise ValueError(f"World '{name}': unknown key '{k}'. Known: "
                             f"{sorted(FRAMS_PROPERTY_MAP) + list(TASK_PARAM_KEYS)}")
    if params.get("world_type", 0) in (1, 2) and not params.get("world_map"):
        raise ValueError(f"World '{name}': blocks/heightfield worlds need a world_map")
    if "rotation" in params and int(params["rotation"]) != 0:
        raise ValueError(f"World '{name}': only rotation=0 is deterministic; use "
                         "target_heading_deg to emulate rotated starts")
    return WorldConfig(name=name, description=description, params=params, task_params=task_params)


def coerce_world(world: WorldConfig | dict | str | None, registry: dict[str, WorldConfig] | None = None
                 ) -> WorldConfig | None:
    """Accepts a WorldConfig, a plain dict (legacy `world_config` dicts), a registered world
    name, or None (= the base training world)."""
    if world is None or isinstance(world, WorldConfig):
        return world
    if isinstance(world, str):
        registry = registry or _default_registry()
        if world not in registry:
            raise KeyError(f"Unknown world '{world}'. Known: {sorted(registry)}")
        return registry[world]
    if isinstance(world, dict):
        d = dict(world)
        name = d.pop("name", "custom")
        return world_from_dict(name, d)
    raise TypeError(f"Cannot interpret world config of type {type(world)}")


_DEFAULT_REGISTRY: dict[str, WorldConfig] | None = None


def _default_registry() -> dict[str, WorldConfig]:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = load_worlds()[0]
    return _DEFAULT_REGISTRY


def load_worlds(path: str | Path | None = None
                ) -> tuple[dict[str, WorldConfig], list[Setting], list[str]]:
    """Returns (worlds by name, the 10 Phase 3 settings, held-out world names)."""
    path = Path(path) if path else DEFAULT_WORLDS_FILE
    with open(path) as f:
        spec = yaml.safe_load(f)
    worlds = {name: world_from_dict(name, d) for name, d in spec["worlds"].items()}
    settings = []
    for s in spec["settings"]:
        if s["world"] not in worlds:
            raise ValueError(f"Setting '{s['name']}' references unknown world '{s['world']}'")
        settings.append(Setting(name=s["name"], world=s["world"], task=s["task"],
                                rationale=s.get("rationale", "")))
    held_out = list(spec.get("held_out_worlds", []))
    used_for_training = {s.world for s in settings}
    for h in held_out:
        if h not in worlds:
            raise ValueError(f"Held-out world '{h}' is not defined")
        if h in used_for_training:
            raise ValueError(f"Held-out world '{h}' is also a training setting -- it would not "
                             "measure generalisation")
    return worlds, settings, held_out


class FramsWorldApplier:
    """Applies WorldConfig objects to the live `frams` module and restores the base world.

    The base values are snapshotted once (right after the .sim files and the project's
    `base_world` overrides are loaded), so `apply(None)` always returns to exactly the
    training world regardless of what was applied before."""

    def __init__(self, frams_module):
        self._frams = frams_module
        self._base: dict[str, Any] = {}
        for key, (obj_name, prop) in FRAMS_PROPERTY_MAP.items():
            obj = getattr(frams_module, obj_name)
            self._base[key] = _read_prop(getattr(obj, prop))
        self._current_key: tuple | None = None

    @property
    def base_values(self) -> dict[str, Any]:
        return dict(self._base)

    def apply(self, world: WorldConfig | None) -> None:
        key = world.key() if world else None
        if key == self._current_key:
            return
        target = dict(self._base)
        if world:
            target.update(world.params)
        for k, v in target.items():
            obj_name, prop = FRAMS_PROPERTY_MAP[k]
            setattr(getattr(self._frams, obj_name), prop, v)
        self._current_key = key


def _read_prop(ext_value):
    """Converts a Framsticks ExtValue to a plain Python value."""
    try:
        return ext_value._value()
    except Exception:  # pragma: no cover - defensive
        return ext_value
