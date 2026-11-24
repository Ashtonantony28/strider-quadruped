# 0003: Geometry and mass properties come from the CAD; collision uses fitted primitives

Status: Accepted
Date: 2026-11-24

## Context
The M0 model used one primitive per body with inertias from textbook formulas. A review of the renders
found real geometric faults: the abduction actuators were buried 40 mm inside the trunk, the thighs
floated 20 mm away from the hip actuators with nothing joining them, thigh and calf capsules overlapped
40 mm at the knee, and self-collision was switched off, so none of this showed up in simulation.

## Decision
- Every body is built as parametric CAD parts (`cad/parts.py`, build123d) from `spec/robot.yaml`.
  Joint interfaces carry an explicit bearing gap (`clearance.joint_gap`, 0.5 mm).
- Mass, COM and full inertia of each body are computed exactly from the B-rep solids (OCC GProp) with
  material densities, or catalogue masses for bought-in parts, and written to the committed, generated
  `spec/massprops.yaml`. The MJCF uses them as `fullinertia`. Parts inside a body are made disjoint so
  no volume is counted twice.
- Visuals are the CAD part meshes. Collision uses primitives fitted to the CAD parts; the audit reports
  how much CAD lies outside them.
- Self-collision is on between all non-adjacent bodies.
- `python -m cad.audit` (also a test) must pass: no overlaps between bodies, no floating parts,
  valid inertias, total mass within the budget tolerance, and no self-contact inside the locomotion
  envelope. Contacts outside it are listed with a reason and stay live in the simulation.

## Consequences
The sim needs no CAD stack at run time (mass properties are committed; meshes only for rendering).
Changing geometry means editing the spec or `cad/parts.py`, rerunning `python -m cad.build`, and
reviewing the massprops and audit diffs.
