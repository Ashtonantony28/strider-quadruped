"""Tests for the spec, the generated Strider model and the Go2 reference (plan M0)."""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np
import pytest

from model.build import load_model, robot_mjcf, stance_hip_height, to_xml
from model.spec import JOINT_NAMES, LEG_ORDER, load_massprops, load_spec, validate_spec
from sim.stand import NQ_FREE, run_stand

SPEC = load_spec()


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return load_model()


# ---- spec ----

def test_cad_mass_within_budget():
    """Plan M1 exit criterion: CAD-derived total mass within the budget tolerance (3%)."""
    total = load_massprops()["_total_mass"]
    budget = SPEC["budget"]["total_mass"]
    assert abs(total - budget) / budget <= SPEC["budget"]["tolerance"]


def test_spec_rejects_wrong_leg_order():
    bad = copy.deepcopy(SPEC)
    bad["legs"]["order"] = ["FR", "FL", "RL", "RR"]
    with pytest.raises(ValueError, match="ADR 0001"):
        validate_spec(bad)


def test_spec_rejects_stance_outside_range():
    bad = copy.deepcopy(SPEC)
    bad["stance"]["knee"] = 0.0
    with pytest.raises(ValueError, match="outside range"):
        validate_spec(bad)


def test_stance_hip_height_closed_form():
    st, legs = SPEC["stance"], SPEC["legs"]
    expected = (legs["thigh"]["length"] * math.cos(st["hip"])
                + legs["calf"]["length"] * math.cos(st["hip"] + st["knee"]) + legs["foot"]["radius"])
    assert stance_hip_height(SPEC) == pytest.approx(expected)
    assert 0.38 < expected < 0.45  # plan: ~0.4 m hip height


# ---- generated model ----

def test_model_compiles_with_spec_rates(model):
    assert model.opt.timestep == SPEC["sim"]["timestep"]
    assert model.nu == 12 and model.nq == 19 and model.nv == 18


def test_model_mass_properties_come_from_cad(model):
    mp = load_massprops()
    assert float(np.sum(model.body_mass)) == pytest.approx(mp["_total_mass"], rel=1e-6)
    for b in range(1, model.nbody):
        assert model.body_mass[b] == pytest.approx(mp[model.body(b).name]["mass"], rel=1e-9)


def test_self_collision_enabled_between_robot_geoms(model):
    """Robot collision geoms collide with each other (non-adjacent bodies) and with the floor."""
    floor = model.geom("floor").id
    for g in range(model.ngeom):
        if g == floor:
            continue
        ct, ca = model.geom_contype[g], model.geom_conaffinity[g]
        assert ct & 2 and ca & 2, model.geom(g).name               # robot <-> robot
        assert ca & model.geom_contype[floor], model.geom(g).name  # robot <-> world


def test_home_pose_has_no_self_contact(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("home").id)
    mujoco.mj_forward(model, d)
    floor = model.geom("floor").id
    robot_pairs = [(c.geom1, c.geom2) for c in d.contact[:d.ncon] if floor not in (c.geom1, c.geom2)]
    assert robot_pairs == []


def test_joint_and_actuator_order_follow_adr_0001(model):
    joints = [model.joint(i).name for i in range(1, model.njnt)]
    assert tuple(joints) == JOINT_NAMES
    actuated = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    assert tuple(actuated) == JOINT_NAMES


def test_actuator_limits_from_spec(model):
    lim = SPEC["actuators"]["torque_limit"]
    expected = np.array([lim[n.split("_")[1]] for n in JOINT_NAMES])
    assert np.allclose(model.actuator_ctrlrange[:, 1], expected)
    assert np.allclose(model.actuator_ctrlrange[:, 0], -expected)


def test_feet_touch_ground_in_home_pose(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("home").id)
    mujoco.mj_forward(model, d)
    r = SPEC["legs"]["foot"]["radius"]
    for leg in LEG_ORDER:
        z = d.site(f"{leg}_foot_site").xpos[2]
        assert z == pytest.approx(r, abs=1e-6)


def test_foot_positions_mirror_symmetric(model):
    """Left/right feet mirror in y and front/rear legs share the same stance geometry."""
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("home").id)
    mujoco.mj_forward(model, d)
    p = {leg: d.site(f"{leg}_foot_site").xpos.copy() for leg in LEG_ORDER}
    assert np.allclose(p["FL"] * [1, -1, 1], p["FR"], atol=1e-9)
    assert np.allclose(p["RL"] * [1, -1, 1], p["RR"], atol=1e-9)


def test_generator_is_deterministic():
    assert to_xml(robot_mjcf(SPEC)) == to_xml(robot_mjcf(SPEC))


def test_pd_stand_holds_upright(model):
    """Smoke test: joint PD holding the home pose keeps the trunk up and level for 5 s."""
    d = run_stand(model, 5.0)
    z0 = stance_hip_height(SPEC)
    assert d.qpos[2] > z0 - 0.03            # sags < 3 cm under joint PD
    tilt = 2 * math.acos(min(1.0, abs(d.qpos[3])))
    assert tilt < math.radians(2)
    assert np.linalg.norm(d.qvel) < 0.05     # at rest


def test_simulation_bitwise_deterministic(model):
    a = run_stand(model, 1.0)
    b = run_stand(model, 1.0)
    assert np.array_equal(a.qpos, b.qpos) and np.array_equal(a.qvel, b.qvel)


def test_stance_torques_within_limits(model):
    """Static stance: gravity-compensating joint torques are well inside actuator limits (> 3x margin)."""
    d = run_stand(model, 3.0)
    tau = np.abs(d.ctrl)
    assert np.all(tau < model.actuator_ctrlrange[:, 1] / 3)


# ---- reference robot ----

@pytest.fixture(scope="module")
def go2() -> mujoco.MjModel:
    from model.fetch_reference import go2_scene_path
    try:
        path = go2_scene_path()
    except OSError as e:  # offline: say so rather than pass silently
        pytest.skip(f"Go2 reference unavailable (network): {e}")
    return mujoco.MjModel.from_xml_path(str(path))


def test_go2_joint_order_matches_adr_0001(go2):
    names = [go2.joint(i).name for i in range(1, go2.njnt)]
    # Go2 calls abd/hip/knee "hip/thigh/calf"; map ours onto theirs leg by leg.
    mapping = {"abd": "hip", "hip": "thigh", "knee": "calf"}
    ours = [f"{n.split('_')[0]}_{mapping[n.split('_')[1]]}_joint" for n in JOINT_NAMES]
    assert names == ours


def test_go2_stands_with_same_pd(go2):
    d = run_stand(go2, 3.0, kp=60.0, kd=2.0)
    assert d.qpos[2] > 0.2
    assert 2 * math.acos(min(1.0, abs(d.qpos[3]))) < math.radians(3)
    assert d.qpos[NQ_FREE:].shape == (12,)
