"""Geometry audit for Strider (ADR 0003). Writes docs/AUDIT-geometry.md and build/audit.json.

    python -m cad.audit            # exact CAD checks + MuJoCo self-collision sweeps

Checks
  A  inside each body: part overlaps (only listed, designed embeds allowed) and connectivity
  B  assembly at the home pose (exact B-rep): no overlap between any two bodies; every child body is
     attached to its parent (gap <= attach_tol); clearances between non-adjacent bodies
  C  mass properties: committed massprops == fresh CAD; MJCF == massprops; inertia physically valid;
     total within budget tolerance; implied densities of bought-in parts plausible
  D  collision primitives vs CAD meshes: how much of each visual mesh lies outside its collision shapes
  E  self-collision sweeps over each leg's full joint range, and front/rear legs against each other
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from build123d import Compound, Location
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

from cad.build import MESH_DIR, compute_massprops, mesh_key, part_density, _transform
from cad.massprops import volume
from cad.parts import all_bodies
from model.build import home_body_poses, load_model
from model.spec import JOINT_NAMES, LEG_ORDER, REPO_ROOT, load_massprops, load_spec

REPORT = REPO_ROOT / "docs" / "AUDIT-geometry.md"
JSON_OUT = REPO_ROOT / "build" / "audit.json"
VOL_EPS = 1e-10      # m^3 (0.1 mm^3): below this an intersection is numerical noise

# Designed embeds inside one body (same rigid part assembly, mass counted once by cad.parts._disjoint).
ALLOWED_EMBEDS = {
    ("knee_actuator", "housing"): "belt housing spigot clamped into the knee actuator stator (9-13 mm)",
    ("boss", "housing"): "knee boss is welded into the lower end of the belt housing",
    ("foot", "tube"): "CF tube end is bonded into the moulded rubber foot",
    ("hub", "tube"): "CF tube is bonded into the aluminium knee hub",
}

# Self-collisions found by the sweep that are accepted, with the reason (filled in from the results).
# Keys are generic: "same" = parts of one leg, "L/R" = left vs right leg, "F/R" = front vs rear leg.
# Half stride of a 1.5 m/s trot with 0.25 s stance: feet stay within this of their hip, along x.
STRIDE_HALF = 0.19
GAIT_MIN_FOOT_DROP = 0.15   # feet at least this far below the hips (legs under the body, crouch included)

# Every accepted contact stays live in the simulation (MuJoCo resolves it like a real hard contact);
# none can occur inside the locomotion envelope, which the audit checks separately.
ACCEPTED_SELF_CONTACT = {
    "thigh~trunk": "needs |abduction| >= 0.5 rad together with the hip swung back past ~0.9 rad "
                   "(thigh under the belly or up against the side)",
    "calf~calf (L/R)": "front calves cross under the belly only at full inward abduction (-0.8 rad)",
    "calf~thigh (F/R)": "one leg's foot swung far under the body (or the leg near horizontal) into the "
                        "other leg's thigh",
    "calf~calf (F/R)": "front and rear feet brought together under the middle of the belly",
    "thigh~thigh (F/R)": "legs swung towards each other with at least one leg near horizontal "
                         "(feet not under the body)",
    "calf~thigh (L/R)": "both legs abducted inwards under the belly",
    "thigh~thigh (L/R)": "both legs abducted inwards under the belly",
}


def _generic(pair: str) -> str:
    """'FL_calf~RL_thigh' -> 'calf~thigh (F/R)'; 'FL_thigh~trunk' -> 'thigh~trunk'."""
    a, b = pair.split("~")
    la, lb = (x.split("_")[0] if "_" in x else None for x in (a, b))
    ka, kb = (x.split("_")[-1] if "_" in x else x for x in (a, b))
    tag = ""
    if la and lb and la != lb:
        tag = " (L/R)" if la[0] == lb[0] else " (F/R)" if la[1] == lb[1] else " (diag)"
    return "~".join(sorted((ka, kb))) + tag


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def h(self, s: str) -> None:
        self.lines += ["", f"## {s}", ""]

    def p(self, s: str = "") -> None:
        self.lines.append(s)

    def fail(self, s: str) -> None:
        self.failures.append(s)
        self.p(f"**FAIL**: {s}")


def _dist(a, b) -> float:
    d = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped)
    d.Perform()
    return float(d.Value())


def _posed(shape, pose):
    pos, rot = pose
    return shape.moved(Location(_transform(pos, rot)))


# ------------------------------------------------------------------ A

def check_parts(rep: Report, spec) -> None:
    rep.h("A. Parts inside each body")
    raw = all_bodies(spec, disjoint=False)
    rows = []
    seen = set()
    for body, parts in raw.items():
        key = mesh_key(body) if not body.endswith("_hip") else "hip"
        if key in seen:
            continue
        seen.add(key)
        for a, b in itertools.combinations(parts, 2):
            v = volume(a.shape & b.shape)
            if v <= VOL_EPS:
                continue
            reason = ALLOWED_EMBEDS.get((a.name, b.name)) or ALLOWED_EMBEDS.get((b.name, a.name))
            rows.append((key, a.name, b.name, v * 1e9, reason or "NOT ALLOWED"))
            if reason is None:
                rep.fail(f"{key}: {a.name} overlaps {b.name} by {v*1e9:.1f} mm^3")
        # connectivity: every part touches another part of the body
        for p in parts:
            others = [q for q in parts if q is not p]
            if not others:
                continue
            gap = min(_dist(p.shape, q.shape) for q in others)
            if gap > spec["clearance"]["attach_tol"]:
                rep.fail(f"{key}: part {p.name} is floating ({gap*1e3:.2f} mm from the rest of its body)")
    rep.p("Every part touches at least one other part of its body (no floating parts).")
    rep.p("")
    rep.p("| body | part | part | overlap (mm^3) | status |")
    rep.p("|---|---|---|---|---|")
    for key, a, b, v, reason in rows:
        rep.p(f"| {key} | {a} | {b} | {v:.0f} | {reason} |")
    rep.data["intra_body_overlaps"] = [r[:4] for r in rows]


# ------------------------------------------------------------------ B

def _bbox(shape) -> tuple[tuple[float, ...], tuple[float, ...]]:
    bb = shape.bounding_box()
    return (bb.min.X, bb.min.Y, bb.min.Z), (bb.max.X, bb.max.Y, bb.max.Z)


def _near(a, b, margin: float) -> bool:
    return all(a[0][i] - margin <= b[1][i] and b[0][i] - margin <= a[1][i] for i in range(3))


def check_assembly(rep: Report, spec, model) -> None:
    """Part-against-part across every pair of bodies (booleans between whole-body compounds are not
    robust in OCC when faces sit 0.5 mm apart, so each part pair is intersected on its own)."""
    rep.h("B. Assembly at the home pose (exact B-rep)")
    bodies = all_bodies(spec)
    poses = home_body_poses()
    posed = {b: [(p.name, _posed(p.shape, poses[b])) for p in parts] for b, parts in bodies.items()}
    boxes = {b: [(n, sh, _bbox(sh)) for n, sh in parts] for b, parts in posed.items()}
    parent = {model.body(i).name: model.body(model.body_parentid[i]).name for i in range(2, model.nbody)}
    tol = spec["clearance"]["attach_tol"]
    rows = []
    for a, b in itertools.combinations(boxes, 2):
        dmin, vsum, closest = math.inf, 0.0, ("", "")
        for (na, sa, ba), (nb, sb, bb) in itertools.product(boxes[a], boxes[b]):
            if not _near(ba, bb, 0.02):
                continue
            d = _dist(sa, sb)
            if d < dmin:
                dmin, closest = d, (na, nb)
            if d < 1e-7:
                v = volume(sa & sb)
                if v > VOL_EPS:
                    vsum += v
                    rep.fail(f"{a}/{na} overlaps {b}/{nb} by {v*1e9:.1f} mm^3")
        if dmin == math.inf:
            continue
        adjacent = parent.get(a) == b or parent.get(b) == a
        rows.append((a, b, adjacent, dmin, vsum, closest))
        if adjacent and dmin > tol:
            rep.fail(f"{b if parent.get(b) == a else a} is floating: {dmin*1e3:.2f} mm from its parent")
    rep.p(f"Every part is checked against every part of every other body within 20 mm. Overlap must be 0; a child "
          f"body must sit within {tol*1e3:.1f} mm of its parent (bearing gap {spec['clearance']['joint_gap']*1e3:.1f} mm).")
    rep.p("")
    rep.p("| body | body | adjacent | min gap (mm) | closest parts | overlap (mm^3) |")
    rep.p("|---|---|---|---|---|---|")
    for a, b, adj, d, v, (na, nb) in sorted(rows, key=lambda r: (not r[2], r[3])):
        rep.p(f"| {a} | {b} | {'yes' if adj else 'no'} | {d*1e3:.2f} | {na} / {nb} | {v*1e9:.1f} |")
    rep.data["home_pairs"] = [r[:5] for r in rows]


# ------------------------------------------------------------------ C

def check_mass(rep: Report, spec, model) -> None:
    rep.h("C. Mass properties")
    fresh = compute_massprops(all_bodies(spec), spec)
    committed = load_massprops()
    worst = 0.0
    for body, v in fresh.items():
        if body.startswith("_"):
            continue
        c = committed[body]
        worst = max(worst, abs(c["mass"] - v["mass"]) / v["mass"],
                    float(np.max(np.abs(np.array(c["fullinertia"]) - np.array(v["fullinertia"])))
                          / max(abs(x) for x in v["fullinertia"][:3])))
    if worst > 1e-5:
        rep.fail(f"spec/massprops.yaml is stale (max rel diff {worst:.2e}); run python -m cad.build")
    rep.p(f"Committed massprops vs fresh CAD: max relative difference {worst:.1e}.")
    total = fresh["_total_mass"]
    budget, tol = spec["budget"]["total_mass"], spec["budget"]["tolerance"]
    err = (total - budget) / budget
    rep.p(f"Total mass {total:.3f} kg vs budget {budget} kg: {err*100:+.1f}% (limit ±{tol*100:.0f}%).")
    if abs(err) > tol:
        rep.fail(f"total mass {total:.3f} kg is outside ±{tol*100:.0f}% of the {budget} kg budget")
    rep.p("")
    rep.p("| body | mass (kg) | MJCF mass (kg) | principal inertia (kg m^2) | triangle inequality |")
    rep.p("|---|---|---|---|---|")
    for b in range(1, model.nbody):
        name = model.body(b).name
        mp = committed[name]
        ixx, iyy, izz, ixy, ixz, iyz = mp["fullinertia"]
        I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        ev = np.linalg.eigvalsh(I)
        tri = ev[0] + ev[1] >= ev[2] * (1 - 1e-9)
        if not (np.all(ev > 0) and tri):
            rep.fail(f"{name}: inertia not physically valid (eigs {ev})")
        if abs(model.body_mass[b] - mp["mass"]) > 1e-6:
            rep.fail(f"{name}: MJCF mass {model.body_mass[b]} != CAD {mp['mass']}")
        if name.startswith(("FR", "RL", "RR")):
            continue  # mirror images of FL
        rep.p(f"| {name} | {mp['mass']:.4f} | {model.body_mass[b]:.4f} | "
              f"{', '.join(f'{e:.2e}' for e in ev)} | {'ok' if tri else 'FAIL'} |")
    rep.p("")
    rep.p("Bought-in parts carry catalogue masses; their implied uniform density is checked for plausibility:")
    rep.p("")
    rep.p("| part | mass (kg) | implied density (kg/m^3) | plausible range |")
    rep.p("|---|---|---|---|")
    ranges = {"actuator": (1200, 3500), "battery": (1200, 2600), "compute": (800, 3000),
              "harness": (800, 3000), "belt": (800, 3000)}
    seen = set()
    for body, parts in all_bodies(spec).items():
        for p in parts:
            if p.mass is None:
                continue
            kind = "actuator" if "actuator" in p.name else p.name
            if kind in seen:
                continue
            seen.add(kind)
            rho = part_density(p, spec)
            lo, hi = ranges[kind]
            if not lo <= rho <= hi:
                rep.fail(f"{p.name}: implied density {rho:.0f} kg/m^3 outside {lo}-{hi}")
            rep.p(f"| {kind} | {p.mass:.3f} | {rho:.0f} | {lo}-{hi} |")


# ------------------------------------------------------------------ D

def _sdf(points: np.ndarray, gtype: int, size: np.ndarray) -> np.ndarray:
    """Signed distance of points (geom frame) to a MuJoCo primitive (negative inside)."""
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.linalg.norm(points, axis=1) - size[0]
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        z = np.clip(points[:, 2], -size[1], size[1])
        return np.linalg.norm(points - np.c_[0 * z, 0 * z, z], axis=1) - size[0]
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        dr = np.linalg.norm(points[:, :2], axis=1) - size[0]
        dz = np.abs(points[:, 2]) - size[1]
        out = np.linalg.norm(np.c_[np.maximum(dr, 0), np.maximum(dz, 0)], axis=1)
        return out + np.minimum(np.maximum(dr, dz), 0)
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        q = np.abs(points) - size
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    raise ValueError(gtype)


def check_coverage(rep: Report, model) -> None:
    rep.h("D. Collision shapes vs CAD meshes")
    rep.p("Collision uses primitives fitted to the CAD (fast, stable contacts). For each body, the share of CAD "
          "mesh vertices lying outside every collision primitive, and how far out the worst one is:")
    rep.p("")
    rep.p("| body | vertices | outside (>1 mm) | worst (mm) | where |")
    rep.p("|---|---|---|---|---|")
    import trimesh
    vis = load_model(visuals=True)
    rows = {}
    for b in range(1, vis.nbody):
        name = vis.body(b).name
        if name.startswith(("FR", "RL", "RR")):
            continue
        cols = [g for g in range(vis.ngeom) if vis.geom_bodyid[g] == b and vis.geom_group[g] == 3]
        pts, owners = [], []
        for g in range(vis.ngeom):
            if vis.geom_bodyid[g] != b or vis.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_name = vis.mesh(vis.geom_dataid[g]).name
            m = trimesh.load(MESH_DIR / f"{mesh_name}.stl")
            pts.append(np.asarray(m.vertices))
            owners += [mesh_name.split("__")[1]] * len(m.vertices)
        P = np.vstack(pts)
        best = np.full(len(P), np.inf)
        for g in cols:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, vis.geom_quat[g])
            local = (P - vis.geom_pos[g]) @ R.reshape(3, 3)
            best = np.minimum(best, _sdf(local, vis.geom_type[g], vis.geom_size[g]))
        out = best > 1e-3
        worst_i = int(np.argmax(best))
        where = owners[worst_i] if out.any() else "-"
        rows[name] = (len(P), float(out.mean()), float(best.max()), where)
        rep.p(f"| {name} | {len(P)} | {out.mean()*100:.1f}% | {max(best.max(), 0)*1e3:.1f} | {where} |")
    rep.data["coverage"] = rows


# ------------------------------------------------------------------ E

def _contacts(model, data) -> set[str]:
    pairs = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if c.dist > 0:
            continue
        n1, n2 = sorted(_kind(model.body(b1).name, model.body(b2).name))
        pairs.add(f"{n1}~{n2}")
    return pairs


def _kind(*names: str) -> tuple[str, ...]:
    return tuple("trunk" if n == "base" else n for n in names)


def check_sweeps(rep: Report, spec, model) -> None:
    rep.h("E. Self-collision across joint ranges")
    data = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    model.geom_conaffinity[floor] = 0
    model.geom_contype[floor] = 0
    rng = spec["joints"]["range"]
    grids = {j: np.linspace(*rng[j], n) for j, n in (("abd", 9), ("hip", 19), ("knee", 11))}
    home = model.key("home").qpos.copy()
    home[2] = 2.0
    qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES}

    # E1: single leg over its full 3-D joint range (other legs at home)
    leg = "FL"
    hits: dict[str, list[tuple[float, float, float]]] = {}
    total = 0
    for qa, qh, qk in itertools.product(grids["abd"], grids["hip"], grids["knee"]):
        q = home.copy()
        q[qadr[f"{leg}_abd"]], q[qadr[f"{leg}_hip"]], q[qadr[f"{leg}_knee"]] = qa, qh, qk
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        total += 1
        for pair in _contacts(model, data):
            hits.setdefault(pair, []).append((qa, qh, qk))
    rep.p(f"E1. {leg} leg over a {len(grids['abd'])}x{len(grids['hip'])}x{len(grids['knee'])} grid of its joint "
          f"range ({total} poses), other legs at home. Contact pairs found:")
    rep.p("")
    rep.p("| pair | poses in contact | share | min abd, max abd | hip range | knee range | why accepted |")
    rep.p("|---|---|---|---|---|---|---|")
    if not hits:
        rep.p("| none | 0 | 0% | - | - | - | - |")
    for pair, qs in sorted(hits.items()):
        qs_a = np.array(qs)
        region = " | ".join(f"{qs_a[:, i].min():+.2f} to {qs_a[:, i].max():+.2f}" for i in range(3))
        generic = _generic(pair)
        status = ACCEPTED_SELF_CONTACT.get(generic, "UNEXPLAINED")
        if status == "UNEXPLAINED":
            rep.fail(f"unexplained self-contact {pair} in {len(qs)} poses, region {region}")
        rep.p(f"| {pair} | {len(qs)} | {len(qs)/total*100:.1f}% | {region} | {status} |")
    rep.data["sweep_single"] = {k: len(v) for k, v in hits.items()}

    # gait-relevant envelope must be completely free
    # gait / stand-up envelope: stance hip 0.67 rad +-0.85, abduction +-0.4, any knee angle
    env = {"abd": (-0.4, 0.4), "hip": (-0.2, 1.5), "knee": rng["knee"]}
    bad = [(p, q) for p, qs in hits.items() for q in qs
           if all(env[j][0] <= v <= env[j][1] for j, v in zip(("abd", "hip", "knee"), q))]
    rep.p("")
    rep.p(f"Locomotion envelope abd {env['abd']}, hip {env['hip']}, knee full range: "
          f"{'no self-contact' if not bad else f'{len(bad)} contacts'}.")
    if bad:
        rep.fail(f"self-contact inside the locomotion envelope: {bad[:3]}")

    # E2: front vs rear leg on the same side (hip and knee of both legs), abduction 0
    hits2: dict[str, list] = {}
    hip_x = spec["legs"]["hip_offset"][0]
    stride_half = STRIDE_HALF
    total2 = 0
    hg, kg = np.linspace(*rng["hip"], 13), np.linspace(*rng["knee"], 6)
    for hf, kf, hr, kr in itertools.product(hg, kg, hg, kg):
        q = home.copy()
        q[qadr["FL_hip"]], q[qadr["FL_knee"]], q[qadr["RL_hip"]], q[qadr["RL_knee"]] = hf, kf, hr, kr
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        total2 += 1
        for pair in _contacts(model, data):
            if "FL" in pair and "RL" in pair:
                # how far each foot has travelled from under its own hip, along x (body frame)
                f_back = hip_x - (data.site("FL_foot_site").xpos[0] - data.qpos[0])
                r_fwd = (data.site("RL_foot_site").xpos[0] - data.qpos[0]) + hip_x
                below = min(data.qpos[2] - data.site(f"{lg}_foot_site").xpos[2] for lg in ("FL", "RL"))
                hits2.setdefault(pair, []).append((f_back, r_fwd, below))
    rep.p("")
    rep.p(f"E2. Front-left vs rear-left leg, hips and knees of both swept together ({total2} poses):")
    rep.p("")
    rep.p("For contacts with both feet under the body, 'closest to gait' is the smallest foot excursion from "
          "under its own hip (along x) that still produced the contact; a 1.5 m/s trot needs at most "
          f"{STRIDE_HALF:.2f} m.")
    rep.p("")
    rep.p("| pair | poses | share | closest to gait (m) | why accepted |")
    rep.p("|---|---|---|---|---|")
    for pair, qs in sorted(hits2.items()):
        qa_ = np.array(qs)
        reason = ACCEPTED_SELF_CONTACT.get(_generic(pair), "UNEXPLAINED")
        if reason == "UNEXPLAINED":
            rep.fail(f"unexplained front/rear contact {pair}")
        under = qa_[qa_[:, 2] >= GAIT_MIN_FOOT_DROP]
        closest = f"{np.max(np.abs(under[:, :2]), axis=1).min():.3f}" if len(under) else "legs not under body"
        rep.p(f"| {pair} | {len(qs)} | {len(qs)/total2*100:.2f}% | {closest} | {reason} |")
    gait2 = [q for qs in hits2.values() for q in qs
             if abs(q[0]) <= stride_half and abs(q[1]) <= stride_half and q[2] >= GAIT_MIN_FOOT_DROP]
    rep.p("")
    rep.p(f"With both feet under the body (at least {GAIT_MIN_FOOT_DROP:.2f} m below the hips) and within "
          f"±{stride_half:.2f} m of their hips along x (a 1.5 m/s trot's stride): {len(gait2)} contacts.")
    for q in gait2[:5]:
        rep.p(f"- front foot back {q[0]:+.3f} m, rear foot forward {q[1]:+.3f} m, lowest foot {q[2]:.3f} m below hip")
    if gait2:
        rep.fail(f"front/rear leg contact inside the gait envelope ({len(gait2)} poses)")
    rep.data["sweep_front_rear"] = {k: len(v) for k, v in hits2.items()}

    # E3: left vs right leg (abduction inward on both)
    hits3 = 0
    in_env3 = 0
    ag = np.linspace(*rng["abd"], 9)
    for af, ar in itertools.product(ag, ag):
        for hp in np.linspace(*rng["hip"], 7):
            q = home.copy()
            q[qadr["FL_abd"]], q[qadr["FR_abd"]] = af, ar
            q[qadr["FL_hip"]] = q[qadr["FR_hip"]] = hp
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            if any("FL" in p and "FR" in p for p in _contacts(model, data)):
                hits3 += 1
                if max(abs(af), abs(ar)) <= 0.4 and -0.2 <= hp <= 1.5:
                    in_env3 += 1
    rep.p("")
    rep.p(f"E3. Front-left vs front-right leg, abductions and hips swept together: {hits3} of {9*9*7} poses in "
          f"contact, {in_env3} of them inside the locomotion envelope (|abd| <= 0.4, hip -0.2 to 1.5). The rest need "
          f"large abduction with the legs swung under the belly or up over the back.")
    if in_env3:
        rep.fail(f"left/right leg contact inside the locomotion envelope ({in_env3} poses)")
    rep.data["sweep_left_right"] = hits3


def main() -> int:
    spec = load_spec()
    model = load_model()
    rep = Report()
    rep.p("# Strider geometry audit")
    rep.p("")
    rep.p("Generated by `python -m cad.audit` (ADR 0003). Do not edit by hand.")
    check_parts(rep, spec)
    check_assembly(rep, spec, model)
    check_mass(rep, spec, model)
    check_coverage(rep, model)
    check_sweeps(rep, spec, load_model())
    rep.h("Result")
    rep.p("PASS: no unexplained findings." if not rep.failures else
          f"FAIL: {len(rep.failures)} finding(s):\n\n" + "\n".join(f"- {f}" for f in rep.failures))
    REPORT.write_text("\n".join(rep.lines) + "\n")
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps({"failures": rep.failures, **{k: str(v) for k, v in rep.data.items()}}, indent=1))
    print("\n".join(rep.failures) or "PASS")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
