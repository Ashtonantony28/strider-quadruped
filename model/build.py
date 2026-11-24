"""Generate the Strider MJCF from spec/robot.yaml + CAD outputs (ADR 0002, ADR 0003). Never hand-edit.

    python -m model.build            # writes build/strider.xml and build/strider_scene.xml

Bodies carry CAD-derived mass properties (spec/massprops.yaml). Each body has two geometry sets:
  * visual: the CAD part meshes (group 2, no contact), from `python -m cad.build`
  * collision: primitives fitted to the CAD parts (group 3); see docs/AUDIT-geometry.md for coverage
Self-collision is ON between non-adjacent bodies (parent/child pairs are excluded by MuJoCo).
"""
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from model.spec import JOINT_ORDER, LEG_ORDER, LEG_SIGNS, REPO_ROOT, load_massprops, load_spec

BUILD_DIR = REPO_ROOT / "build"
MESH_DIR = BUILD_DIR / "meshes"

# Contact bitmasks: robot geoms collide with each other (non-adjacent bodies) and with the world;
# world geoms (floor, terrain) collide with the robot only.
ROBOT_CONTACT = {"contype": "2", "conaffinity": "3"}
WORLD_CONTACT = {"contype": "1", "conaffinity": "2"}
VISUAL = {"contype": "0", "conaffinity": "0", "group": "2", "density": "0"}
COLLISION_GROUP = "3"
JOINT_AXES = {"abd": "1 0 0", "hip": "0 1 0", "knee": "0 1 0"}
QUAT_Z_TO_Y = "0.70710678 0.70710678 0 0"   # cylinder axis z -> y
QUAT_Z_TO_X = "0.70710678 0 0.70710678 0"   # cylinder axis z -> x
HIP_SIDE_PLATE_X1 = 0.035                     # side plate reaches past the flexion axis (hip frame x)
THIGH_BEAM_Z_TOP = -0.035                     # beam spigot depth into the knee actuator (thigh frame z)


def _fmt(*values: float) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def layout(spec: dict[str, Any]) -> dict[str, float]:
    """Same derived layout the CAD uses (single implementation lives in cad.parts, mirrored here
    without importing build123d so the sim does not need the CAD stack)."""
    t, a, legs = spec["trunk"], spec["actuator"], spec["legs"]
    gap = spec["clearance"]["joint_gap"]
    hx, hy, _ = legs["hip_offset"]
    end_outer = t["size"][0] / 2
    lateral = legs["hip"]["lateral"]
    flex_y_out = lateral - a["width"] / 2 - gap
    boss_half = legs["calf"]["hub_width"] / 2
    return {
        "hx": hx, "hy": hy,
        "end_outer_hip": end_outer - hx,
        "end_inner_hip": end_outer - t["wall"] - hx,
        "flex_y_out": flex_y_out,
        "flex_y_c": flex_y_out - a["width"] / 2,
        "flex_y_in": flex_y_out - a["width"],
        "calf_lateral": boss_half + gap + legs["calf"]["hub_width"] / 2,
        "boss_half": boss_half,
    }


def stance_hip_height(spec: dict[str, Any]) -> float:
    """Height of the hip axis above ground with all feet down in the nominal stance."""
    legs, st = spec["legs"], spec["stance"]
    lt, lc = legs["thigh"]["length"], legs["calf"]["length"]
    drop = lt * math.cos(st["hip"]) + lc * math.cos(st["hip"] + st["knee"])
    return drop + legs["foot"]["radius"]


def _inertial(parent: ET.Element, mp: dict[str, Any]) -> None:
    ET.SubElement(parent, "inertial", pos=_fmt(*mp["com"]), mass=_fmt(mp["mass"]),
                  fullinertia=_fmt(*mp["fullinertia"]))


def _col(parent: ET.Element, name: str, **attrs: str) -> None:
    ET.SubElement(parent, "geom", name=name, group=COLLISION_GROUP, rgba="0.9 0.2 0.2 0.35",
                  **ROBOT_CONTACT, **attrs)


def _visuals(parent: ET.Element, key: str, parts: dict[str, float], spec_parts: dict[str, str]) -> None:
    for part in parts:
        ET.SubElement(parent, "geom", name=f"{parent.get('name')}__{part}_vis", type="mesh",
                      mesh=f"{key}__{part}", material=spec_parts.get(part, "alu"), **VISUAL)


# part name -> material used for rendering (colours live in the <asset> section)
PART_MATERIALS = {
    "shell": "shell", "battery": "battery", "compute": "pcb", "harness": "dark", "head": "accent",
    "lens_glass": "glass", "handle": "dark", "belly_plate": "carbon", "flex_actuator": "actuator", "bracket": "alu",
    "knee_actuator": "actuator", "boss": "alu", "housing": "accent", "belt": "dark",
    "foot": "rubber", "hub": "alu", "tube": "carbon",
    **{f"{leg}_abd_actuator": "actuator" for leg in LEG_ORDER},
}
MATERIALS = {
    "shell": ("0.86 0.87 0.89 1", "0.5", "0.6"), "actuator": ("0.16 0.17 0.19 1", "0.6", "0.7"),
    "alu": ("0.66 0.68 0.72 1", "0.7", "0.8"), "accent": ("0.93 0.47 0.10 1", "0.4", "0.5"),
    "carbon": ("0.09 0.09 0.10 1", "0.5", "0.8"), "rubber": ("0.05 0.05 0.05 1", "0.05", "0.1"),
    "glass": ("0.05 0.15 0.25 1", "1.0", "1.0"), "pcb": ("0.10 0.35 0.18 1", "0.3", "0.3"),
    "battery": ("0.20 0.25 0.45 1", "0.3", "0.3"), "dark": ("0.12 0.12 0.13 1", "0.3", "0.4"),
}


def _mesh_key(body: str) -> str:
    for kind in ("thigh", "calf"):
        if body.endswith(f"_{kind}"):
            return kind
    return body


def _leg(base: ET.Element, leg: str, spec: dict[str, Any], mp: dict[str, Any], visuals: bool) -> None:
    legs, a, lay = spec["legs"], spec["actuator"], layout(spec)
    sx, sy = LEG_SIGNS[leg]
    gap, hp = spec["clearance"]["joint_gap"], legs["hip"]
    tp, ph = hp["plate_thickness"], hp["plate_height"] / 2
    th, cf = legs["thigh"], legs["calf"]
    r_act, hw_act = a["diameter"] / 2, a["width"] / 2

    hip = ET.SubElement(base, "body", name=f"{leg}_hip", pos=_fmt(sx * lay["hx"], sy * lay["hy"], 0))
    _inertial(hip, mp[f"{leg}_hip"])
    ET.SubElement(hip, "joint", name=f"{leg}_abd", **{"class": "abd"})
    if visuals:
        _visuals(hip, _mesh_key(f"{leg}_hip"), mp[f"{leg}_hip"]["parts"], PART_MATERIALS)
    _col(hip, f"{leg}_flex_act_col", type="cylinder", size=_fmt(r_act, hw_act),
         pos=_fmt(0, sy * lay["flex_y_c"], 0), quat=QUAT_Z_TO_Y)
    x0 = lay["end_outer_hip"] + gap
    y_lo, y_hi = -ph, lay["flex_y_in"]
    _col(hip, f"{leg}_bracket_col", type="box", size=_fmt(tp / 2, (y_hi - y_lo) / 2, ph),
         pos=_fmt(sx * (x0 + tp / 2), sy * (y_hi + y_lo) / 2, 0))
    hub_x0 = lay["end_inner_hip"] + gap
    _col(hip, f"{leg}_abd_hub_col", type="cylinder", size=_fmt(a["output_hub_diameter"] / 2, (x0 - hub_x0) / 2),
         pos=_fmt(sx * (x0 + hub_x0) / 2, 0, 0), quat=QUAT_Z_TO_X)
    x1 = HIP_SIDE_PLATE_X1
    _col(hip, f"{leg}_side_plate_col", type="box", size=_fmt((x1 - x0) / 2, tp / 2, ph),
         pos=_fmt(sx * (x0 + x1) / 2, sy * (lay["flex_y_in"] - tp / 2), 0))

    thigh = ET.SubElement(hip, "body", name=f"{leg}_thigh", pos=_fmt(0, sy * hp["lateral"], 0))
    _inertial(thigh, mp[f"{leg}_thigh"])
    ET.SubElement(thigh, "joint", name=f"{leg}_hip", **{"class": "hip"})
    if visuals:
        _visuals(thigh, "thigh", mp[f"{leg}_thigh"]["parts"], PART_MATERIALS)
    L = th["length"]
    z_top = THIGH_BEAM_Z_TOP
    w0, w1 = th["beam_width"]
    w_top = w0 + (w1 - w0) * (z_top / -L)
    _col(thigh, f"{leg}_knee_act_col", type="cylinder", size=_fmt(r_act, hw_act), quat=QUAT_Z_TO_Y)
    _col(thigh, f"{leg}_beam_col", type="box", size=_fmt(w_top / 2, th["beam_thickness"] / 2, (L + z_top) / 2),
         pos=_fmt(0, 0, (z_top - L) / 2))
    _col(thigh, f"{leg}_boss_col", type="cylinder", size=_fmt(th["knee_boss_diameter"] / 2, lay["boss_half"]),
         pos=_fmt(0, 0, -L), quat=QUAT_Z_TO_Y)

    calf = ET.SubElement(thigh, "body", name=f"{leg}_calf", pos=_fmt(0, sy * lay["calf_lateral"], -L))
    _inertial(calf, mp[f"{leg}_calf"])
    ET.SubElement(calf, "joint", name=f"{leg}_knee", **{"class": "knee"})
    if visuals:
        _visuals(calf, "calf", mp[f"{leg}_calf"]["parts"], PART_MATERIALS)
    Lc = cf["length"]
    _col(calf, f"{leg}_hub_col", type="cylinder", size=_fmt(cf["hub_diameter"] / 2, cf["hub_width"] / 2),
         quat=QUAT_Z_TO_Y)
    _col(calf, f"{leg}_tube_col", type="capsule", fromto=_fmt(0, 0, 0, 0, 0, -Lc),
         size=_fmt(cf["tube_diameter"][0] / 2))
    ET.SubElement(calf, "geom", name=f"{leg}_foot", type="sphere", pos=_fmt(0, 0, -Lc),
                  size=_fmt(legs["foot"]["radius"]), condim="3", priority="1",
                  friction=_fmt(*spec["ground"]["friction"]), group=COLLISION_GROUP,
                  rgba="0.9 0.2 0.2 0.35", **ROBOT_CONTACT)
    ET.SubElement(calf, "site", name=f"{leg}_foot_site", pos=_fmt(0, 0, -Lc), size="0.005", group="4")


def robot_mjcf(spec: dict[str, Any] | None = None, visuals: bool = True) -> ET.Element:
    spec = spec or load_spec()
    mp = load_massprops()
    sim, joints, act, t = spec["sim"], spec["joints"], spec["actuators"], spec["trunk"]
    root = ET.Element("mujoco", model=spec["name"])
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", meshdir=str(MESH_DIR))
    ET.SubElement(root, "option", timestep=_fmt(sim["timestep"]), integrator=sim["integrator"],
                  gravity=_fmt(0, 0, -sim["gravity"]), cone=sim["cone"], impratio=_fmt(sim["impratio"]))

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping=_fmt(joints["damping"]),
                  frictionloss=_fmt(joints["frictionloss"]), armature=_fmt(joints["armature"]))
    for j in JOINT_ORDER:
        cls = ET.SubElement(default, "default", **{"class": j})
        ET.SubElement(cls, "joint", axis=JOINT_AXES[j], range=_fmt(*joints["range"][j]))
        lim = act["torque_limit"][j]
        ET.SubElement(cls, "motor", ctrlrange=_fmt(-lim, lim))

    if visuals:
        asset = ET.SubElement(root, "asset")
        for name, (rgba, spec_, shin) in MATERIALS.items():
            ET.SubElement(asset, "material", name=name, rgba=rgba, specular=spec_, shininess=shin)
        for stl in sorted(MESH_DIR.glob("*.stl")):
            ET.SubElement(asset, "mesh", name=stl.stem, file=stl.name)

    world = ET.SubElement(root, "worldbody")
    z0 = stance_hip_height(spec)
    base = ET.SubElement(world, "body", name="base", pos=_fmt(0, 0, z0))
    ET.SubElement(base, "freejoint", name="root")
    _inertial(base, mp["base"])
    if visuals:
        _visuals(base, "base", mp["base"]["parts"], PART_MATERIALS)
    L, W, H = t["size"]
    ET.SubElement(base, "geom", name="trunk", type="box", size=_fmt(L / 2, W / 2, H / 2),
                  group=COLLISION_GROUP, rgba="0.9 0.2 0.2 0.35", **ROBOT_CONTACT)
    hd = t["head"]
    hx, hy, hz = hd["size"]
    _col(base, "head_col", type="box", size=_fmt(hx / 2, hy / 2, hz / 2),
         pos=_fmt(hd["x_front"] - hx / 2, 0, H / 2 + hz / 2))
    hn = t["handle"]
    _col(base, "handle_col", type="capsule", size=_fmt(hn["bar_diameter"] / 2),
         fromto=_fmt(hn["x_center"] - hn["span"] / 2, 0, H / 2 + hn["height"],
                     hn["x_center"] + hn["span"] / 2, 0, H / 2 + hn["height"]))
    bp = t["belly_plate"]
    _col(base, "belly_col", type="box",
         size=_fmt(L / 2 - bp["inset"][0], W / 2 - bp["inset"][1], bp["thickness"] / 2),
         pos=_fmt(0, 0, -H / 2 - bp["thickness"] / 2))
    ET.SubElement(base, "site", name="imu", pos="0 0 0", size="0.01", group="4")
    for leg in LEG_ORDER:
        _leg(base, leg, spec, mp, visuals)

    actuator = ET.SubElement(root, "actuator")
    for leg in LEG_ORDER:
        for j in JOINT_ORDER:
            ET.SubElement(actuator, "motor", name=f"{leg}_{j}_motor", joint=f"{leg}_{j}", **{"class": j})

    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "framequat", name="imu_quat", objtype="site", objname="imu")
    ET.SubElement(sensor, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sensor, "accelerometer", name="imu_acc", site="imu")

    st = spec["stance"]
    qpos = [0, 0, z0, 1, 0, 0, 0] + [st[j] for _ in LEG_ORDER for j in JOINT_ORDER]
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="home", qpos=_fmt(*qpos))
    return root


def scene_mjcf(spec: dict[str, Any] | None = None, visuals: bool = True) -> ET.Element:
    """Robot plus floor, lights, sky and a framebuffer large enough for 1080p renders."""
    spec = spec or load_spec()
    root = robot_mjcf(spec, visuals)
    visual = ET.Element("visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="8")
    ET.SubElement(visual, "map", znear="0.02", zfar="30")
    ET.SubElement(visual, "headlight", ambient="0.25 0.25 0.25", diffuse="0.3 0.3 0.3", specular="0.1 0.1 0.1")
    root.insert(2, visual)
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(3, asset)
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.75 0.82 0.9",
                  rgb2="0.95 0.96 0.98", width="512", height="512")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1="0.80 0.82 0.84",
                  rgb2="0.72 0.74 0.77", width="512", height="512", mark="edge", markrgb="0.6 0.62 0.65")
    ET.SubElement(asset, "material", name="floor", texture="grid", texrepeat="8 8", texuniform="true",
                  reflectance="0", specular="0", shininess="0")
    world = root.find("worldbody")
    assert world is not None
    # Lighting: a distant wide spot as the shadow-casting key (near-parallel shadows) plus a shadow-free
    # directional fill. A directional key speckled the floor at its shadow-map edge (found by eye in the
    # audit renders); a spot's perspective shadow map does not.
    world.insert(0, ET.Element("light", pos="1.5 -1.0 2.5", dir="-0.5 0.35 -1", directional="true",
                               diffuse="0.32 0.32 0.34", castshadow="false"))
    world.insert(0, ET.Element("light", pos="-5 5 8", dir="0.5 -0.5 -0.8", directional="false",
                               cutoff="60", exponent="0", attenuation="1 0 0",
                               diffuse="0.55 0.55 0.55", castshadow="true"))
    world.insert(0, ET.Element("geom", name="floor", type="plane", size="0 0 0.05", material="floor",
                               friction=_fmt(*spec["ground"]["friction"]), **WORLD_CONTACT))
    return root


def to_xml(root: ET.Element) -> str:
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def load_model(scene: bool = True, spec: dict[str, Any] | None = None, visuals: bool = False) -> mujoco.MjModel:
    """Compile the model. visuals=False (default) skips the CAD meshes: physics is identical
    (mass properties are explicit) and it does not need `python -m cad.build` to have run."""
    root = scene_mjcf(spec, visuals) if scene else robot_mjcf(spec, visuals)
    return mujoco.MjModel.from_xml_string(to_xml(root))


def home_body_poses() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return {model.body(b).name: (data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy())
            for b in range(1, model.nbody)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=BUILD_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "strider.xml").write_text(to_xml(robot_mjcf()))
    (args.out / "strider_scene.xml").write_text(to_xml(scene_mjcf()))
    model = load_model(visuals=True)
    print(f"wrote {args.out}/strider*.xml: nq={model.nq} nu={model.nu} nmesh={model.nmesh} "
          f"mass={sum(model.body_mass):.3f} kg hip height={stance_hip_height(load_spec()):.3f} m")


if __name__ == "__main__":
    main()
