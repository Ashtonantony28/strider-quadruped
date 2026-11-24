"""Exact mass properties of B-rep solids (OCC GProp), combined per body with the parallel-axis theorem."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps


@dataclass
class MassProps:
    mass: float
    com: np.ndarray        # (3,)
    inertia: np.ndarray    # (3,3) about the COM

    def __add__(self, other: "MassProps") -> "MassProps":
        m = self.mass + other.mass
        com = (self.mass * self.com + other.mass * other.com) / m
        return MassProps(m, com, _shift(self, com) + _shift(other, com))


def _shift(mp: MassProps, point: np.ndarray) -> np.ndarray:
    """Inertia of mp about 'point' (parallel-axis theorem)."""
    d = mp.com - point
    return mp.inertia + mp.mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def solid_props(shape, density: float) -> MassProps:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, props)
    vol = props.Mass()                       # unit density -> volume
    c = props.CentreOfMass()
    mat = props.MatrixOfInertia()            # about the COM, unit density
    inertia = np.array([[mat.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    return MassProps(density * vol, np.array([c.X(), c.Y(), c.Z()]), density * inertia)


def volume(shape) -> float:
    """Volume of a solid; 0 for an empty boolean result."""
    if shape is None or getattr(shape, "_wrapped", None) is None:
        return 0.0
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, props)
    return props.Mass()
