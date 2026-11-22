# Strider: walking robot-dog simulation (quadruped design + locomotion control)

Status: DRAFT plan, 2026-11-22. Working name "Strider", suggested repo `Ashtonantony28/strider-quadruped`.
Plan only. Nothing is built until you approve.

## 1. What "walking dog simulation" could mean

| | Option | What it shows | Fit with existing work | Risk |
|---|---|---|---|---|
| **A (recommended)** | **Robot dog: design a 12-DoF quadruped and make it walk in a physics sim, with a classical MPC controller and a learned (RL) policy compared head to head** | Legged-robot dynamics, contact-rich simulation, convex MPC + whole-body control, PPO with domain randomisation, sim-to-sim transfer, CAD | Reuses NeuroFlight's PPO + int8 pipeline, Helix Arm's actuator model and build123d/Zener toolchain, and runs the policy bit-exact on **Vortex-V**; stretch link to vla-arm (arm on the back) | Medium: well-trodden research area with published baselines; CPU-only training is the main constraint (sec. 7) |
| B | Animal dog: physics-based character animation that imitates real dog motion capture (DeepMimic/AMP style) | Character animation, imitation learning, biomechanics | Weak links to other projects; closer to Hollow Signal than to the hardware projects | Medium-high: needs a licensed dog mocap set, heavier RL |
| C | Procedural animated dog (inverse kinematics + gait cycle) in the browser | Animation and IK only | Little reuse, no physics | Low, but shallow as a portfolio piece |

**Recommendation: A**, with B folded in as a stretch: an AMP "style reward" from public dog motion data makes the robot walk in a more dog-like way (M9). A covers the "walking dog" ask, has real engineering depth, and links to four existing projects instead of standing alone.

## 2. Goal and headline deliverables

A medium-size robot dog (~12 kg, ~0.4 m hip height, 3 actuated joints per leg) designed in code and walking in a validated MuJoCo simulation:

1. **Robot design package**: `spec/robot.yaml` as single source of truth; build123d parametric CAD (body, legs, feet, battery bay, head mount); mass and inertia exported from CAD; MJCF model generated from the spec, never hand-edited.
2. **Actuators**: quasi-direct-drive BLDC + planetary (or the cycloidal unit from Helix Arm) modelled with torque-speed limits, current limits, gear backlash, friction and thermal model; same actuator model library shared with Helix Arm.
3. **Simulation**: MuJoCo 3.x (CPU), 500 Hz physics, 50 Hz policy, terrain generator (flat, slopes, stairs, rough heightfield, low-friction patches), push disturbances, sensor models (IMU noise/bias, joint encoder quantisation, foot contact).
4. **Controllers, from simple to strong**:
   - Open-loop gait baseline: central pattern generator (CPG) trot with Raibert foot placement.
   - **Classical**: convex MPC on single-rigid-body dynamics (MIT Cheetah 3 style, solved with OSQP) + whole-body controller (QP) for joint torques. Gaits: stand, walk, trot, pace, bound, plus gait transitions.
   - **Learned**: PPO policy (blind, proprioception only) trained with domain randomisation and a terrain curriculum; then a teacher-student policy with a heightmap for stairs.
   - State estimator: kinematics + IMU EKF (body velocity and height), used by both controllers.
5. **Evaluation**: velocity-tracking error, cost of transport, fall rate under pushes, max slope and step height, gait-transition speeds compared against dog biomechanics (walk-to-trot near Froude number 0.5, Alexander 1984) as a "does it walk like a dog" check.
6. **Sim-to-sim transfer**: the trained policy run unmodified in a second engine (Genesis or PyBullet) to check it is not exploiting MuJoCo quirks.
7. **Hardware-in-the-loop on Vortex-V**: the policy quantised to int8 with NeuroFlight's toolchain (same requant rule as Vortex-V `mlacc.rqs`), run in Vortex-V Verilator co-sim in lockstep with the MuJoCo plant; report cycles per 50 Hz control step.
8. **Visualisation**: MuJoCo offscreen MP4 renders plus a browser 3D replay (three.js) of the dog walking over the terrain course, and a short report.

## 3. Architecture

```
strider-quadruped/
  spec/robot.yaml        single source of truth (geometry, masses, actuators, limits)
  cad/                   build123d models -> STEP/STL + mass properties
  model/                 MJCF generator from spec (+ reference Unitree Go2 from MuJoCo Menagerie, BSD-3)
  actuators/             motor + gearbox models (shared code with helix-arm)
  sim/                   env, terrain, sensors, disturbances
  est/                   state estimator (EKF)
  ctrl/cpg/              open-loop baseline
  ctrl/mpc/              convex MPC + WBC
  rl/                    gymnasium env, PPO, curriculum, teacher-student
  eval/                  benchmark course + metrics
  transfer/              second-engine runner
  hil/                   int8 export + Vortex-V co-sim bridge
  viz/                   renders + three.js replay
  tests/                 pytest: analytic cases, golden fixtures
  docs/adr/              decisions (ADR 0001: frames FLU body / ENU world, joint order, units SI)
```

## 4. Milestones

| M | Scope | Verify command | Exit criteria |
|---|---|---|---|
| M0 | Repo scaffold, spec, MJCF generator, reference Go2 model loads, analytic physics tests | `pytest -q tests/` | Pendulum period and free fall match analytic within 0.1%; energy drift bounded in frictionless tests |
| M1 | Own robot: build123d CAD, mass props from CAD, actuator models | `python -m cad.build && pytest -q tests/test_robot.py` | CAD mass within 3% of budget; joint torque margin > 1.5x at a 2 g stance load |
| M2 | Sim env, sensors, terrain generator, CPG trot baseline | `python -m eval.run --ctrl cpg --course flat` | Walks 10 m on flat without falling |
| M3 | State estimator (EKF) | `pytest -q tests/test_est.py` | Body velocity error < 0.1 m/s RMS on scripted trot |
| M4 | Convex MPC + WBC, five gaits, transitions | `python -m eval.run --ctrl mpc --course standard` | Tracks 0 to 1.5 m/s with < 0.1 m/s error; recovers from 50 N·0.1 s pushes |
| M5 | PPO blind policy, flat then rough, domain randomisation | `python -m rl.train --cfg blind && python -m eval.run --ctrl rl` | Matches MPC tracking on flat; beats MPC on rough terrain fall rate |
| M6 | Teacher-student with heightmap: stairs and slopes | `python -m eval.run --ctrl rl_student --course stairs` | Climbs 12 cm steps and 20° slopes, ≥ 95% success |
| M7 | Sim-to-sim transfer + dog-gait comparison | `python -m transfer.run --engine genesis` | ≤ 20% tracking degradation in the second engine; gait-transition Froude numbers reported |
| M8 | int8 policy + Vortex-V HIL | `make -C hil cosim` | Policy output bit-exact vs host golden; cycles per step reported vs 20 ms budget |
| M9 | Report, renders, three.js replay; stretch: AMP dog-style reward, arm on the back (vla-arm), head camera (Sentinel) | `python -m viz.replay --out site/` | Published replay + report |

## 5. Tooling (all cloud, no purchase)

Python 3.11, MuJoCo 3.x (pip, CPU), numpy, scipy, OSQP / qpSWIFT, Pinocchio (kinematics/dynamics for WBC), gymnasium, PyTorch (CPU), build123d, Genesis or PyBullet, matplotlib, ffmpeg + EGL/OSMesa for renders. Verilator + RISC-V GCC for M8 (already used by Vortex-V). Nothing runs on your laptop.

## 6. Links to your other projects

- **NeuroFlight**: PPO training scaffold, checkpoint/resume discipline, int8 quantise/golden/codegen toolchain (reused directly).
- **Vortex-V**: M8 HIL, same path as the NeuroFlight and Redline HIL workloads.
- **Helix Arm**: shared actuator model and build123d/Zener toolchain; the dog's legs use the same joint-module family.
- **vla-arm** (stretch): a small arm mounted on the back for loco-manipulation, commanded over vla-arm protocol v1.
- **Sentinel camera** (stretch): head camera feed for the student policy or object following.

## 7. Risks

- **CPU-only RL training.** The container has 4 cores and no GPU. Estimate: ~30k physics steps/s, so a flat-ground blind policy (~30-50M steps) takes 1-3 h and rough terrain (~200M) 8 h+. Mitigations: residual policy on top of the CPG to cut sample needs, short episodes, checkpoint every 5 min with resume (container restarts kill long runs), train overnight. If that is too slow, a free GPU notebook (Colab/Kaggle) with MuJoCo MJX is the fallback (decision D3).
- **Contact simulation artefacts** (policy exploits foot sliding or penetration): mitigated by the sim-to-sim check in M7 and conservative contact parameters.
- **MPC tuning and QP conditioning**: start from the published MIT Cheetah 3 formulation on the reference Go2 model before switching to the own robot. Fable 5.2 reserved for the MPC/WBC derivation and any hard sim-to-sim debugging.
- **HIL cycle budget**: a ~200 k-MAC MLP at 50 Hz should fit; if not, shrink the policy (distillation) before blaming the core.
- **Scope creep**: option B and the stretch items stay out until M8 passes.

## 8. Decisions for you

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Direction | A robot dog / B animated animal dog / C procedural animation | **A**, with B's dog-style reward as a stretch |
| D2 | Robot | (a) Own design from `spec/robot.yaml`, validated against the Go2 reference first; (b) reference Go2 model only | **(a)**: the design work is half the portfolio value; Go2 stays as the benchmark |
| D3 | Training compute | (a) Cloud container CPU, overnight runs; (b) add a free GPU notebook for the rough-terrain stage | **(a)** to start; decide on (b) only if M5 rough-terrain training is too slow |
| D4 | Physical build | (a) Sim-only; (b) plan a later build with a measured-parameter phase | **(a)**, outputs stay fab-ready so (b) remains open |
| D5 | Repo | New public repo `strider-quadruped`, or your own name | **New public repo**; you create it (sec. 9) |

## 9. First steps after approval

1. You create the empty repo: `gh repo create Ashtonantony28/strider-quadruped --public` (no README). Until then work lives in the container with a bundle backup at `/mnt/project-files/project-plans/walking-dog/strider.bundle`.
2. M0 + M1 in one thread (ADR 0001 fixes frames, joint order and units).
3. After M3, MPC (M4) and RL (M5) run as two parallel threads; M6-M9 follow.
