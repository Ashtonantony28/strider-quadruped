"""Physically simulated demo reel (needs `python -m cad.build` and `python -m model.build` first).

    MUJOCO_GL=egl python -m viz.demo_video --out docs/media/

The CAD-meshed robot is dropped onto the floor and then driven through a sequence of joint-space poses by the
joint PD hold of sim/stand.py, with the targets interpolated smoothly. Every frame is MuJoCo physics (contacts,
actuator torque limits from spec/robot.yaml); there is NO gait controller yet (CPG/MPC/RL are M2+), so the
robot never walks in this reel. Writes strider-demo.{mp4,gif} and strider-hero.png.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from model.build import load_model
from model.spec import JOINT_NAMES
from sim.stand import NQ_FREE, pd_hold, run_stand

KP, KD = run_stand.__defaults__[:2]            # the same gains as the sim/stand.py smoke test

LEGS = ("FL", "FR", "RL", "RR")


def pose(home: np.ndarray, **kw) -> np.ndarray:
    q = home.copy()
    for name, v in kw.items():
        if name in JOINT_NAMES:
            q[JOINT_NAMES.index(name)] = v
        else:                                   # "hip" -> all four legs
            for lg in LEGS:
                q[JOINT_NAMES.index(f"{lg}_{name}")] = v
    return q


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/media"))
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model = load_model(visuals=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    home = data.qpos[NQ_FREE:].copy()
    data.qpos[2] += 0.15                         # drop from 15 cm above the home height
    mujoco.mj_forward(model, data)
    # (label, target, seconds to blend, seconds to hold)
    script = [
        ("drop from 15 cm, PD hold", home, 0.0, 1.6),
        ("crouch", pose(home, hip=1.2, knee=-2.45), 0.9, 0.6),
        ("stand", home, 0.9, 0.5),
        ("raise front-left paw (no balance control yet, so it tips)", pose(home, FL_abd=0.6, FL_hip=-0.2, FL_knee=-2.4), 0.8, 1.0),
        ("stand", home, 0.8, 0.5),
        ("crouch", pose(home, hip=1.2, knee=-2.45), 0.9, 0.5),
        ("stand", home, 0.9, 1.5),
    ]
    W, H = 1280, 720
    r = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    opt.geomgroup[2] = 1
    dt = model.opt.timestep
    spf = max(1, int(round(1.0 / a.fps / dt)))
    big, small = font(30), font(18)
    tmp = Path(tempfile.mkdtemp())
    n, t, q_prev = 0, 0.0, home.copy()
    total = sum(b + h for _, _, b, h in script)
    for label, target, blend, hold in script:
        q0 = q_prev.copy()
        steps = int(round((blend + hold) / dt))
        for k in range(steps):
            s = min(1.0, (k * dt) / blend) if blend > 0 else 1.0
            s = 0.5 - 0.5 * np.cos(np.pi * s)          # smooth blend
            pd_hold(model, data, q0 + (target - q0) * s, kp=KP, kd=KD)
            mujoco.mj_step(model, data)
            t += dt
            if k % spf == 0:
                cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.22]
                cam.distance, cam.elevation = 1.45, -14
                cam.azimuth = 125 + 50 * t / total
                r.update_scene(data, cam, opt)
                img = Image.fromarray(r.render())
                d = ImageDraw.Draw(img, "RGBA")
                d.rectangle((0, 0, W, 96), fill=(13, 17, 23, 200))
                d.rectangle((0, H - 64, W, H), fill=(13, 17, 23, 200))
                d.text((28, 22), "Strider", font=big, fill=(240, 240, 240))
                d.text((28, 62), "SIMULATION (MuJoCo): CAD-meshed 12.4 kg model, joint PD hold only. No gait controller yet.",
                       font=small, fill=(215, 215, 215))
                d.text((28, H - 50), f"{label}", font=big, fill=(255, 196, 120))
                d.text((W - 150, H - 40), f"t = {t:4.1f} s", font=small, fill=(215, 215, 215))
                img.save(tmp / f"f{n:04d}.png")
                n += 1
        q_prev = target
    cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.24]
    cam.distance, cam.azimuth, cam.elevation = 1.35, 140, -16
    hero = mujoco.Renderer(model, 1080, 1620)
    hero.update_scene(data, cam, opt)
    Image.fromarray(hero.render()).save(a.out / "strider-hero.png")
    ff = ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.fps), "-i", str(tmp / "f%04d.png")]
    subprocess.run(ff + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(a.out / "strider-demo.mp4")], check=True)
    subprocess.run(ff + ["-vf", "fps=12,scale=640:-1:flags=lanczos,split[x][y];[x]palettegen=max_colors=64:stats_mode=diff[p];"
                         "[y][p]paletteuse=dither=none:diff_mode=rectangle", str(a.out / "strider-demo.gif")], check=True)
    shutil.rmtree(tmp)
    print(f"{n} frames, {t:.1f} s simulated")


if __name__ == "__main__":
    main()
