# Strider: a walking robot dog in simulation

Strider is a 12-joint, ~12 kg quadruped designed in code and made to walk in MuJoCo, with a classical
convex-MPC controller and a learned (PPO) policy compared head to head, a sim-to-sim transfer check, and the
int8 policy running bit-exact on the [Vortex-V](https://github.com/Ashtonantony28/vortex-v) RISC-V core.
Full plan: [docs/PLAN.md](docs/PLAN.md).

## Status

| Milestone | Scope | State |
|---|---|---|
| M0 | Scaffold, spec, MJCF generator, Go2 reference, analytic physics tests | done |
| M1 | build123d CAD, mass properties from CAD, actuator models | next |
| M2-M9 | CPG, EKF, MPC, RL, stairs, sim-to-sim, Vortex-V HIL, replay | planned |

## Quick start

```bash
pip install -r requirements.txt
python -m model.fetch_reference   # Unitree Go2 from MuJoCo Menagerie, pinned + checksummed
python -m model.build             # build/strider.xml from spec/robot.yaml
pytest -q tests/                  # Verify command for M0
```

## Layout

- `spec/robot.yaml`: every robot parameter (single source of truth)
- `model/`: spec loader, MJCF generator, primitive inertias, Go2 fetcher
- `sim/`: simulation helpers (M0: joint PD hold used as a smoke test)
- `tests/`: analytic physics references (free fall, exact large-amplitude pendulum period, energy and
  angular-momentum conservation, Coulomb friction) and model tests
- `docs/adr/`: decisions (0001 frames/joint order/units, 0002 MuJoCo + generated model)

The Unitree Go2 model is © Unitree Robotics, BSD-3-Clause, fetched at run time from MuJoCo Menagerie and not
redistributed here.
