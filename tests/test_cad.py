"""CAD mass-property tests (ADR 0003): analytic references, then the committed massprops file."""
from __future__ import annotations

import numpy as np
import pytest

build123d = pytest.importorskip("build123d")
from build123d import Box, Cylinder, Pos  # noqa: E402

from cad.build import compute_massprops  # noqa: E402
from cad.massprops import MassProps, solid_props  # noqa: E402
from cad.parts import all_bodies  # noqa: E402
from model.spec import load_massprops, load_spec  # noqa: E402


def test_box_props_match_closed_form():
    m = solid_props(Box(0.3, 0.2, 0.1), density=1000.0)
    mass = 1000.0 * 0.3 * 0.2 * 0.1
    assert m.mass == pytest.approx(mass, rel=1e-9)
    expected = mass / 12 * np.array([0.2**2 + 0.1**2, 0.3**2 + 0.1**2, 0.3**2 + 0.2**2])
    assert np.allclose(np.diag(m.inertia), expected, rtol=1e-9)
    assert np.allclose(m.com, 0, atol=1e-12)


def test_cylinder_props_match_closed_form():
    r, h, rho = 0.048, 0.048, 1465.0
    m = solid_props(Cylinder(r, h), rho)
    mass = rho * np.pi * r**2 * h
    assert m.mass == pytest.approx(mass, rel=1e-9)
    assert m.inertia[2, 2] == pytest.approx(0.5 * mass * r**2, rel=1e-9)
    assert m.inertia[0, 0] == pytest.approx(mass * (3 * r**2 + h**2) / 12, rel=1e-9)


def test_parallel_axis_composition_matches_single_solid():
    """Two half boxes combined with the parallel-axis theorem equal the whole box."""
    whole = solid_props(Box(0.4, 0.1, 0.1), 2700.0)
    a = solid_props(Pos(-0.1, 0, 0) * Box(0.2, 0.1, 0.1), 2700.0)
    b = solid_props(Pos(0.1, 0, 0) * Box(0.2, 0.1, 0.1), 2700.0)
    both: MassProps = a + b
    assert both.mass == pytest.approx(whole.mass, rel=1e-12)
    assert np.allclose(both.inertia, whole.inertia, rtol=1e-9, atol=1e-15)


def test_committed_massprops_match_fresh_cad():
    """spec/massprops.yaml is a reviewed, generated file: it must equal a fresh CAD build."""
    spec = load_spec()
    fresh = compute_massprops(all_bodies(spec), spec)
    committed = load_massprops()
    assert set(fresh) == set(committed)
    for body, v in fresh.items():
        if body.startswith("_"):
            continue
        assert committed[body]["mass"] == pytest.approx(v["mass"], rel=1e-6), body
        assert np.allclose(committed[body]["com"], v["com"], atol=1e-6), body
        assert np.allclose(committed[body]["fullinertia"], v["fullinertia"], rtol=1e-5, atol=1e-10), body


def test_mirrored_legs_have_mirrored_mass_properties():
    mp = load_massprops()
    fl, fr, rl = (np.array(mp[f"{leg}_hip"]["com"]) for leg in ("FL", "FR", "RL"))
    assert np.allclose(fr, fl * [1, -1, 1], atol=1e-7)
    assert np.allclose(rl, fl * [-1, 1, 1], atol=1e-7)
    assert mp["FL_hip"]["mass"] == pytest.approx(mp["RR_hip"]["mass"], rel=1e-9)
