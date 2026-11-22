# 0002: MuJoCo as primary simulator; the robot model is generated from the spec

Status: Accepted
Date: 2026-11-22

## Context
The plan (sec. 5) needs a CPU-friendly, contact-capable simulator that runs in the cloud container
(4 cores, no GPU). The robot is our own design, and its parameters will change as CAD matures (M1).

## Decision
- MuJoCo (pip `mujoco`, CPU) is the primary simulator. A second engine is used only for the
  sim-to-sim check (M7).
- The Strider MJCF is **generated** by `model/build.py` from `spec/robot.yaml`; it is never edited by
  hand and is not committed (it is a build output).
- The Unitree Go2 model from MuJoCo Menagerie (BSD-3) is the reference robot. It is **fetched**, not
  vendored, by `python -m model.fetch_reference` at a pinned commit and checked against committed
  SHA-256 sums (`model/reference/go2.sha256`), because the meshes are 30 MB.

## Consequences
Changing the robot means editing the spec, never XML. Tests that need Go2 fetch it first (CI does this).
