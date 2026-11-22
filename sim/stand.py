"""Joint-space PD hold: the simplest controller, used as a model smoke test (not a gait controller)."""
from __future__ import annotations

import mujoco
import numpy as np

NQ_FREE = 7  # free joint: 3 position + 4 quaternion
NV_FREE = 6


def pd_hold(model: mujoco.MjModel, data: mujoco.MjData, q_target: np.ndarray,
            kp: float, kd: float) -> None:
    """Write clipped PD torques for the 12 leg joints into data.ctrl (ADR 0001 order)."""
    q = data.qpos[NQ_FREE:]
    dq = data.qvel[NV_FREE:]
    tau = kp * (q_target - q) - kd * dq
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    data.ctrl[:] = np.clip(tau, lo, hi)


def run_stand(model: mujoco.MjModel, seconds: float, kp: float = 60.0, kd: float = 2.0,
              key: str = "home") -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
    q_target = data.qpos[NQ_FREE:].copy()
    for _ in range(int(round(seconds / model.opt.timestep))):
        pd_hold(model, data, q_target, kp, kd)
        mujoco.mj_step(model, data)
    return data
