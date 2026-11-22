"""Generate the Strider MJCF from spec/robot.yaml (ADR 0002). Never hand-edit the output.

    python -m model.build            # writes build/strider.xml and build/strider_scene.xml
"""
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco

from model import inertia
from model.spec import JOINT_ORDER, LEG_ORDER, LEG_SIGNS, REPO_ROOT, load_spec

BUILD_DIR = REPO_ROOT / "build"

# Collision filtering: robot geoms collide with the world (floor, terrain) but not with each other.
# World geoms use contype=1/conaffinity=0, robot geoms contype=0/conaffinity=1.
ROBOT_CONTACT = {"contype": "0", "conaffinity": "1"}
WORLD_CONTACT = {"contype": "1", "conaffinity": "0"}
JOINT_AXES = {"abd": "1 0 0", "hip": "0 1 0", "knee": "0 1 0"}


def _fmt(*values: float) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _inertial(parent: ET.Element, pos: tuple[float, float, float], mass: float,
              diag: tuple[float, float, float]) -> None:
    ET.SubElement(parent, "inertial", pos=_fmt(*pos), mass=_fmt(mass), diaginertia=_fmt(*diag))


def stance_hip_height(spec: dict[str, Any]) -> float:
    """Height of the hip axis above ground with all feet down in the nominal stance."""
    legs, st = spec["legs"], spec["stance"]
    lt, lc = legs["thigh"]["length"], legs["calf"]["length"]
    drop = lt * math.cos(st["hip"]) + lc * math.cos(st["hip"] + st["knee"])
    return drop + legs["foot"]["radius"]


def _leg(base: ET.Element, leg: str, spec: dict[str, Any]) -> None:
    legs = spec["legs"]
    sx, sy = LEG_SIGNS[leg]
    hx, hy, hz = legs["hip_offset"]
    hip_spec, thigh, calf = legs["hip"], legs["thigh"], legs["calf"]

    hip = ET.SubElement(base, "body", name=f"{leg}_hip", pos=_fmt(sx * hx, sy * hy, hz))
    _inertial(hip, (0, 0, 0), hip_spec["mass"],
              inertia.cylinder(hip_spec["mass"], hip_spec["radius"], hip_spec["length"], "x"))
    ET.SubElement(hip, "joint", name=f"{leg}_abd", **{"class": "abd"})
    ET.SubElement(hip, "geom", name=f"{leg}_hip_geom", type="cylinder",
                  size=_fmt(hip_spec["radius"], hip_spec["length"] / 2), quat="0.70710678 0 0.70710678 0",
                  rgba="0.25 0.25 0.28 1", **ROBOT_CONTACT)

    th = ET.SubElement(hip, "body", name=f"{leg}_thigh", pos=_fmt(0, sy * hip_spec["lateral"], 0))
    _inertial(th, (0, 0, -thigh["length"] / 2), thigh["mass"],
              inertia.cylinder(thigh["mass"], thigh["radius"], thigh["length"], "z"))
    ET.SubElement(th, "joint", name=f"{leg}_hip", **{"class": "hip"})
    ET.SubElement(th, "geom", name=f"{leg}_thigh_geom", type="capsule",
                  fromto=_fmt(0, 0, 0, 0, 0, -thigh["length"]), size=_fmt(thigh["radius"]),
                  rgba="0.85 0.45 0.10 1", **ROBOT_CONTACT)

    cf = ET.SubElement(th, "body", name=f"{leg}_calf", pos=_fmt(0, 0, -thigh["length"]))
    _inertial(cf, (0, 0, -calf["length"] / 2), calf["mass"],
              inertia.cylinder(calf["mass"], calf["radius"], calf["length"], "z"))
    ET.SubElement(cf, "joint", name=f"{leg}_knee", **{"class": "knee"})
    ET.SubElement(cf, "geom", name=f"{leg}_calf_geom", type="capsule",
                  fromto=_fmt(0, 0, 0, 0, 0, -calf["length"]), size=_fmt(calf["radius"]),
                  rgba="0.30 0.30 0.32 1", **ROBOT_CONTACT)
    ET.SubElement(cf, "geom", name=f"{leg}_foot", type="sphere", pos=_fmt(0, 0, -calf["length"]),
                  size=_fmt(legs["foot"]["radius"]), condim="3", priority="1",
                  friction=_fmt(*spec["ground"]["friction"]), rgba="0.1 0.1 0.1 1", **ROBOT_CONTACT)
    ET.SubElement(cf, "site", name=f"{leg}_foot_site", pos=_fmt(0, 0, -calf["length"]), size="0.005")


def robot_mjcf(spec: dict[str, Any] | None = None) -> ET.Element:
    spec = spec or load_spec()
    sim, joints, act = spec["sim"], spec["joints"], spec["actuators"]
    root = ET.Element("mujoco", model=spec["name"])
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
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

    world = ET.SubElement(root, "worldbody")
    z0 = stance_hip_height(spec)
    base = ET.SubElement(world, "body", name="base", pos=_fmt(0, 0, z0))
    ET.SubElement(base, "freejoint", name="root")
    body = spec["body"]
    _inertial(base, tuple(body["com_offset"]), body["mass"], inertia.box(body["mass"], *body["size"]))
    ET.SubElement(base, "geom", name="trunk", type="box", size=_fmt(*(s / 2 for s in body["size"])),
                  rgba="0.85 0.85 0.88 1", **ROBOT_CONTACT)
    ET.SubElement(base, "site", name="imu", pos="0 0 0", size="0.01")
    for leg in LEG_ORDER:
        _leg(base, leg, spec)

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


def scene_mjcf(spec: dict[str, Any] | None = None) -> ET.Element:
    """Robot plus a flat floor, a light and an offscreen framebuffer large enough for 1080p renders."""
    spec = spec or load_spec()
    root = robot_mjcf(spec)
    visual = ET.Element("visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    root.insert(2, visual)
    world = root.find("worldbody")
    assert world is not None
    world.insert(0, ET.Element("light", pos="0 0 3", dir="0 0 -1", directional="true"))
    world.insert(0, ET.Element("geom", name="floor", type="plane", size="0 0 0.05",
                               friction=_fmt(*spec["ground"]["friction"]), rgba="0.55 0.6 0.65 1",
                               **WORLD_CONTACT))
    return root


def to_xml(root: ET.Element) -> str:
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def load_model(scene: bool = True, spec: dict[str, Any] | None = None) -> mujoco.MjModel:
    root = scene_mjcf(spec) if scene else robot_mjcf(spec)
    return mujoco.MjModel.from_xml_string(to_xml(root))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=BUILD_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "strider.xml").write_text(to_xml(robot_mjcf()))
    (args.out / "strider_scene.xml").write_text(to_xml(scene_mjcf()))
    model = load_model()
    print(f"wrote {args.out}/strider*.xml: nq={model.nq} nu={model.nu} "
          f"mass={sum(model.body_mass):.3f} kg hip height={stance_hip_height(load_spec()):.3f} m")


if __name__ == "__main__":
    main()
