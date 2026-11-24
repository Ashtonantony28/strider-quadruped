"""The full geometry audit (docs/AUDIT-geometry.md) must pass: no overlaps, no floating parts,
valid mass properties, and no self-contact inside the locomotion envelope. ~70 s."""
from __future__ import annotations

import pytest

pytest.importorskip("build123d")


def test_geometry_audit_passes():
    from cad.audit import main
    assert main() == 0, "see docs/AUDIT-geometry.md for the findings"
