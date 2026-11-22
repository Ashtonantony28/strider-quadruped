# CLAUDE.md: conventions for Strider

1. **Every change ships with a Verify command** in its PR description (pytest invocation, eval run, training run).
2. **Analytic before numeric.** Every physics or controller model has an analytic or published reference test in
   `tests/` before it is relied on. Tolerances are stated next to the test and never loosened silently.
3. **`spec/robot.yaml` is the single source of truth** for robot parameters; no magic numbers in sim/ctrl/rl code.
   The MJCF is generated (`python -m model.build`), never hand-edited (ADR 0002).
4. Frames, joint order and units follow ADR 0001. Hard-to-reverse decisions get an ADR in `docs/adr/`.
5. Deterministic: seeded RNG only; simulations must be bit-reproducible on the same machine.
6. Long training runs checkpoint at least every 5 minutes and resume with `--resume`.
7. Never commit to `main` directly; one logical change per PR; red CI never merges; commit messages explain why.
