"""Analytic reference tests for the simulator itself (plan M0, CLAUDE.md "analytic before numeric").

Each case uses a tiny hand-written MJCF so the expected answer is a closed-form result.
Tolerances are stated next to each test and are never loosened silently.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from scipy.special import ellipk

G = 9.81


def _model(body: str, option: str = 'integrator="RK4" timestep="0.0005"', world: str = "",
           flags: str = "") -> mujoco.MjModel:
    xml = f"""<mujoco><option {option} gravity="0 0 -{G}">{flags}</option>
      <worldbody>{world}{body}</worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def test_free_fall_matches_kinematics():
    """z(t) = z0 - g t^2 / 2 to 0.1% of the drop (RK4, no contact). Plan M0 exit criterion."""
    m = _model('<body pos="0 0 10"><freejoint/><geom type="sphere" size="0.1" mass="1" contype="0" conaffinity="0"/></body>')
    d = mujoco.MjData(m)
    t_end = 1.0
    for _ in range(round(t_end / m.opt.timestep)):
        mujoco.mj_step(m, d)
    drop = 10.0 - d.qpos[2]
    expected = 0.5 * G * t_end**2
    assert abs(drop - expected) / expected < 1e-3
    assert abs(d.qvel[2] + G * t_end) / (G * t_end) < 1e-3


def _pendulum(length: float) -> mujoco.MjModel:
    # Point-mass pendulum: tiny sphere on a massless hinge arm, hinge about y.
    return _model(f"""<body><joint name="h" type="hinge" axis="0 1 0" damping="0"/>
        <geom type="sphere" size="0.001" pos="0 0 -{length}" mass="1" contype="0" conaffinity="0"/>
        </body>""")


def _period(m: mujoco.MjModel, theta0: float, periods: int = 3) -> float:
    d = mujoco.MjData(m)
    d.qpos[0] = theta0
    crossings: list[float] = []
    prev = d.qvel[0]
    while len(crossings) < periods + 1:
        mujoco.mj_step(m, d)
        v = d.qvel[0]
        if prev < 0 <= v:  # velocity sign change at the negative turning point, once per period
            # linear interpolation of the crossing time
            frac = -prev / (v - prev)
            crossings.append(d.time - m.opt.timestep * (1 - frac))
        prev = v
    return (crossings[-1] - crossings[0]) / periods


@pytest.mark.parametrize("theta0", [0.05, 1.0, 2.5])
def test_pendulum_period_exact_elliptic(theta0):
    """Period T = 4 sqrt(L/g) K(sin^2(theta0/2)) for any amplitude, to 0.1%. Plan M0 exit criterion."""
    length = 0.5
    m = _pendulum(length)
    expected = 4 * math.sqrt(length / G) * ellipk(math.sin(theta0 / 2) ** 2)
    got = _period(m, theta0)
    assert abs(got - expected) / expected < 1e-3


def test_double_pendulum_energy_conserved():
    """Frictionless chaotic double pendulum: total energy drift < 1e-4 relative over 10 s (RK4)."""
    m = _model("""<body><joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.02" mass="1" contype="0" conaffinity="0"/>
        <body pos="0 0 -0.4"><joint type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="0.7" contype="0" conaffinity="0"/>
        </body></body>""", flags='<flag energy="enable"/>')
    d = mujoco.MjData(m)
    d.qpos[:] = [2.0, -1.0]
    mujoco.mj_forward(m, d)
    e0 = d.energy.sum()
    scale = abs(d.energy[0]) + abs(d.energy[1])
    worst = 0.0
    for _ in range(round(10.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        worst = max(worst, abs(d.energy.sum() - e0))
    assert worst / scale < 1e-4


def test_torque_free_body_conserves_angular_momentum():
    """Asymmetric box spinning in zero g near its intermediate axis (Dzhanibekov): world-frame
    angular momentum is conserved to 1e-6 relative over 10 s while the body tumbles."""
    m = mujoco.MjModel.from_xml_string("""<mujoco><option integrator="RK4" timestep="0.0005" gravity="0 0 0"/>
      <worldbody><body><freejoint/>
        <geom type="box" size="0.3 0.2 0.1" mass="2" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qvel[3:6] = [0.01, 5.0, 0.01]  # body-frame angular velocity, mostly about the intermediate axis

    def h_world() -> np.ndarray:
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, d.qpos[3:7])
        rot = rot.reshape(3, 3)
        inertia = m.body_inertia[1]  # principal, body frame aligned with geom
        return rot @ (inertia * d.qvel[3:6])

    h0 = h_world()
    flipped = False
    for _ in range(round(10.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if d.qvel[4] < 0:
            flipped = True
    assert flipped, "intermediate-axis flip should occur (sanity check that the body actually tumbles)"
    assert np.linalg.norm(h_world() - h0) / np.linalg.norm(h0) < 1e-6


def _sliding_box(mu: float, contact: str = "") -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = _model(f'<body pos="0 0 0.05"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="1" '
               f'friction="{mu} 0 0" {contact}/></body>',
               option='timestep="0.001" cone="elliptic" impratio="10"',
               world=f'<geom type="plane" size="0 0 1" friction="{mu} 0 0"/>')
    d = mujoco.MjData(m)
    for _ in range(200):  # settle onto the plane
        mujoco.mj_step(m, d)
    d.qvel[0] = 3.0
    return m, d


def test_sliding_friction_obeys_coulomb_cone():
    """While sliding, total tangential contact force = mu * total normal force (to 1e-6).
    This is the friction law itself, independent of how soft the contact is."""
    mu = 0.5
    m, d = _sliding_box(mu)
    normal = tangential = 0.0
    f = np.zeros(6)
    for _ in range(300):
        mujoco.mj_step(m, d)
        for i in range(d.ncon):
            mujoco.mj_contactForce(m, d, i, f)
            normal += f[0]
            tangential += np.linalg.norm(f[1:3])
    assert normal > 0
    assert abs(tangential / normal - mu) < 1e-6


def test_sliding_block_coulomb_deceleration():
    """A box constrained to translate (slide x + slide z, no rotation) decelerates at mu*g, within 2%,
    with the same contact settings the robot uses (elliptic cone, impratio 10, default solref).

    Finding (2026-11-22): a *free-floating* box under the same settings rocks and hops on its corners
    after the velocity kick, so its mean normal force is 5-10% off m*g and the measured deceleration
    misses mu*g by that much, even though the Coulomb ratio is exact (test above). Corner chatter is
    tracked as a contact-artefact risk for the sim-to-sim check (plan M7)."""
    mu = 0.5
    m = _model(f'<body pos="0 0 0.05"><joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 0 1"/>'
               f'<geom type="box" size="0.05 0.05 0.05" mass="1" friction="{mu} 0 0"/></body>',
               option='timestep="0.001" cone="elliptic" impratio="10"',
               world=f'<geom type="plane" size="0 0 1" friction="{mu} 0 0"/>')
    d = mujoco.MjData(m)
    for _ in range(300):  # settle onto the plane
        mujoco.mj_step(m, d)
    d.qvel[0] = 3.0
    for _ in range(2):  # let the contact pick up the new velocity
        mujoco.mj_step(m, d)
    v1, t1 = d.qvel[0], d.time
    for _ in range(300):
        mujoco.mj_step(m, d)
    decel = (v1 - d.qvel[0]) / (d.time - t1)
    assert d.qvel[0] > 0.1, "still sliding over the measurement window"
    assert abs(decel - mu * G) / (mu * G) < 0.02


def test_resting_contact_penetration_small():
    """A 12 kg box resting on the floor settles with < 2 mm penetration and no drift."""
    m = _model('<body pos="0 0 0.06"><freejoint/><geom type="box" size="0.2 0.1 0.05" mass="12"/></body>',
               option='timestep="0.002"', world='<geom type="plane" size="0 0 1"/>')
    d = mujoco.MjData(m)
    for _ in range(2000):
        mujoco.mj_step(m, d)
    assert 0.05 - d.qpos[2] < 2e-3
    assert np.linalg.norm(d.qvel) < 1e-3
