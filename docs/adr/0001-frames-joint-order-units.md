# 0001: Frames, joint order and units

Status: Accepted
Date: 2026-11-22

## Context
Controllers (CPG, MPC, RL), the state estimator, the int8 HIL path and the reference Go2 model must
all agree on how state and commands are laid out. NeuroFlight's INTERFACES.md showed that fixing this
on day one avoids silent sign bugs later.

## Decision
- **World frame**: z up, x and y horizontal (ENU-style), origin on the ground plane. Gravity -z.
- **Body frame**: FLU (x forward, y left, z up), origin at the trunk geometric centre.
- **Quaternions**: scalar-first `[w, x, y, z]` (MuJoCo convention), body-to-world.
- **Leg order**: FL, FR, RL, RR. **Joint order within a leg**: abduction, hip, knee. The 12-vector of
  joint positions/velocities/torques is `[FL_abd, FL_hip, FL_knee, FR_abd, ..., RR_knee]`.
  This matches the Unitree Go2 model in MuJoCo Menagerie (hip, thigh, calf), so controllers run on
  both robots unchanged.
- **Joint signs**: abduction positive about body +x; hip and knee positive about body +y, so a positive
  hip angle swings the foot backwards and the knee is negative when bent (same as Go2).
- **Units**: SI (m, kg, s, rad, N, N m). No degrees anywhere in code.
- **Rates**: physics 500 Hz, control 50 Hz (`spec/robot.yaml` `sim`).

## Consequences
Any new interface (observation vector, HIL packet) is defined in terms of this ordering and frame, and
tests assert the generated model's joint order against this list.
