"""Render Strider stills from the CAD-meshed model (needs `python -m cad.build` first).

    MUJOCO_GL=egl python -m viz.render --out renders/

Writes hero.png (3/4 view), views.png (side / front / top / leg close-up), poses.png (articulation),
and collision.png (CAD meshes beside the fitted collision primitives).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from model.build import load_model
from model.spec import JOINT_NAMES
from sim.stand import NQ_FREE, run_stand


def _cam(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def _shot(r: mujoco.Renderer, model, data, cam, groups=(2,)) -> np.ndarray:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1  # floor
    for g in groups:
        opt.geomgroup[g] = 1
    r.update_scene(data, cam, opt)
    return r.render()


def _pose(model, q_leg: dict[str, float], base_z: float | None = None) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for name, v in q_leg.items():
        data.qpos[NQ_FREE + JOINT_NAMES.index(name)] = v
    if base_z is not None:
        data.qpos[2] = base_z
    mujoco.mj_forward(model, data)
    # rest the robot on its lowest foot so no pose is shown floating
    r = model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FL_foot"), 0]
    lowest = min(data.site(f"{lg}_foot_site").xpos[2] for lg in ("FL", "FR", "RL", "RR")) - r
    data.qpos[2] -= lowest
    mujoco.mj_forward(model, data)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("renders"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_model(visuals=True)
    stand = run_stand(model, 1.0)
    look = [0.0, 0.0, 0.24]

    big = mujoco.Renderer(model, 1080, 1620)
    Image.fromarray(_shot(big, model, stand, _cam(look, 1.35, 140, -16))).save(args.out / "hero.png")

    r = mujoco.Renderer(model, 600, 900)
    views = [
        _shot(r, model, stand, _cam(look, 1.25, 90, -2)),             # side
        _shot(r, model, stand, _cam(look, 1.05, 180, -6)),            # front
        _shot(r, model, stand, _cam(look, 1.35, 320, -28)),           # rear three-quarter
        _shot(r, model, stand, _cam([0.22, 0.15, 0.24], 0.75, 215, -12)),  # front-left leg close-up
    ]
    Image.fromarray(np.vstack([np.hstack(views[:2]), np.hstack(views[2:])])).save(args.out / "views.png")

    poses = [
        _pose(model, {f"{lg}_{j}": v for lg in ("FL", "FR", "RL", "RR") for j, v in (("hip", 1.25), ("knee", -2.5))},
              ),                                                                  # crouch
        _pose(model, {"FL_hip": 0.35, "FL_knee": -1.95, "RR_hip": 0.95, "RR_knee": -1.95}),  # trot: FL+RR swing
        _pose(model, {"FL_abd": 0.6, "FL_hip": -0.2, "FL_knee": -2.4}),                     # paw raised out
        _pose(model, {"FL_hip": 2.6, "FL_knee": -2.6, "FR_hip": 2.6, "FR_knee": -2.6}),  # legs folded up
    ]
    shots = [_shot(r, model, d, _cam(look, 1.35, 140, -14)) for d in poses]
    Image.fromarray(np.vstack([np.hstack(shots[:2]), np.hstack(shots[2:])])).save(args.out / "poses.png")

    col = [_shot(r, model, stand, _cam(look, 1.35, 140, -16), groups=(2,)),
           _shot(r, model, stand, _cam(look, 1.35, 140, -16), groups=(3,))]
    Image.fromarray(np.hstack(col)).save(args.out / "collision.png")
    print(f"wrote renders to {args.out}")


if __name__ == "__main__":
    main()
