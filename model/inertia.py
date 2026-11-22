"""Inertia tensors of uniform solid primitives about their own centroid.

Used by the M0 generator until M1 exports real mass properties from CAD.
"""
from __future__ import annotations


def box(mass: float, sx: float, sy: float, sz: float) -> tuple[float, float, float]:
    """Solid box with full side lengths sx, sy, sz. Returns (Ixx, Iyy, Izz)."""
    k = mass / 12.0
    return (k * (sy**2 + sz**2), k * (sx**2 + sz**2), k * (sx**2 + sy**2))


def cylinder(mass: float, radius: float, length: float, axis: str) -> tuple[float, float, float]:
    """Solid cylinder with its symmetry axis along 'x', 'y' or 'z'."""
    axial = 0.5 * mass * radius**2
    transverse = mass * (3.0 * radius**2 + length**2) / 12.0
    return tuple(axial if a == axis else transverse for a in "xyz")  # type: ignore[return-value]
