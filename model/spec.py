"""Load and validate spec/robot.yaml, the single source of truth (ADR 0002)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "spec" / "robot.yaml"

# ADR 0001: fixed orderings. Tests assert generated models against these.
LEG_ORDER = ("FL", "FR", "RL", "RR")
JOINT_ORDER = ("abd", "hip", "knee")
JOINT_NAMES = tuple(f"{leg}_{j}" for leg in LEG_ORDER for j in JOINT_ORDER)

# Sign of the hip offset for each leg: (x sign, y sign). +x forward, +y left.
LEG_SIGNS = {"FL": (1, 1), "FR": (1, -1), "RL": (-1, 1), "RR": (-1, -1)}


def load_spec(path: Path | str = SPEC_PATH) -> dict[str, Any]:
    with open(path) as f:
        spec = yaml.safe_load(f)
    validate_spec(spec)
    return spec


def component_masses(spec: dict[str, Any]) -> dict[str, float]:
    legs = spec["legs"]
    n = len(legs["order"])
    return {
        "body": spec["body"]["mass"],
        "hips": n * legs["hip"]["mass"],
        "thighs": n * legs["thigh"]["mass"],
        "calves": n * legs["calf"]["mass"],
    }


def total_mass(spec: dict[str, Any]) -> float:
    return sum(component_masses(spec).values())


def validate_spec(spec: dict[str, Any]) -> None:
    legs = spec["legs"]
    if tuple(legs["order"]) != LEG_ORDER:
        raise ValueError(f"leg order must be {LEG_ORDER} (ADR 0001), got {legs['order']}")
    if tuple(spec["joints"]["names"]) != JOINT_ORDER:
        raise ValueError(f"joint order must be {JOINT_ORDER} (ADR 0001)")
    for j in JOINT_ORDER:
        lo, hi = spec["joints"]["range"][j]
        if not lo < spec["stance"][j] < hi:
            raise ValueError(f"stance {j}={spec['stance'][j]} outside range [{lo}, {hi}]")
        if spec["actuators"]["torque_limit"][j] <= 0:
            raise ValueError(f"torque limit for {j} must be positive")
    budget = spec["budget"]["total_mass"]
    if abs(total_mass(spec) - budget) > 1e-9:
        raise ValueError(f"component masses sum to {total_mass(spec)}, budget says {budget}")
