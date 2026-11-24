"""Parametric CAD for Strider (build123d). Every dimension comes from spec/robot.yaml (ADR 0002).

Each function returns the parts of ONE simulation body, expressed in that body's MJCF frame
(ADR 0001: FLU, joint axes abd=+x, hip/knee=+y). Parts inside one body are made disjoint here
(each later part has the earlier ones subtracted) so mass properties never double-count volume.

Layout (left-front leg, mirrored for the others):
  trunk end wall  |gap| abd output hub+flange plate -- side plate -- hip-flexion actuator |gap|
  knee actuator (thigh top, also the flexion output) -- hollow belt-housing beam -- knee boss |gap|
  calf hub (beside the thigh, outboard) -- hollow CF tube -- rubber foot
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from build123d import (Axis, Box, Cylinder, Part, Plane, Pos, Rot, Sphere, Solid, fillet, loft,
                       mirror, Circle, Rectangle, RectangleRounded, Sketch, Location, extrude, Compound)

from model.build import HIP_SIDE_PLATE_X1, THIGH_BEAM_Z_TOP, layout  # shared with the MJCF generator
from model.spec import LEG_ORDER, LEG_SIGNS, load_spec

MM = 1e-3


@dataclass
class PartSpec:
    """One manufactured or bought-in part of a body."""
    name: str
    shape: Part
    material: str | None = None   # key into spec['materials'] (density) ...
    mass: float | None = None     # ... or a catalogue mass (uniform density inferred from volume)
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    # Optional render-only shape. Thin closed shells (2 mm trunk wall) leak light through MuJoCo's
    # shadow map, so the trunk renders from its filled outer envelope; mass always uses `shape`.
    visual: Part | None = None


COLORS = {
    "shell": (0.86, 0.87, 0.89, 1.0),
    "actuator": (0.16, 0.17, 0.19, 1.0),
    "alu": (0.62, 0.64, 0.68, 1.0),
    "accent": (0.93, 0.47, 0.10, 1.0),
    "cf": (0.10, 0.10, 0.11, 1.0),
    "rubber": (0.06, 0.06, 0.06, 1.0),
    "glass": (0.05, 0.12, 0.20, 1.0),
    "pcb": (0.10, 0.35, 0.18, 1.0),
    "battery": (0.20, 0.25, 0.45, 1.0),
}


# ---------------------------------------------------------------- derived layout

# ---------------------------------------------------------------- primitives

def _cyl_y(r: float, w: float, at=(0.0, 0.0, 0.0)) -> Part:
    return Pos(*at) * Rot(X=90) * Cylinder(r, w)


def _cyl_x(r: float, w: float, at=(0.0, 0.0, 0.0)) -> Part:
    return Pos(*at) * Rot(Y=90) * Cylinder(r, w)


def actuator_shape(spec: dict[str, Any]) -> Part:
    """QDD actuator housing, axis +z, centred at the origin; stays inside its Ø x width envelope."""
    a = spec["actuator"]
    r, w = a["diameter"] / 2, a["width"]
    body = Cylinder(r, w)
    body = fillet(body.edges().filter_by(lambda e: True), radius=3.0 * MM)
    # cooling grooves around the rim
    groove_d = 1.6 * MM
    for i in range(a["fin_count"]):
        ang = 360.0 * i / a["fin_count"]
        body -= Rot(Z=ang) * Pos(r - groove_d / 2 + 0.2 * MM, 0, 0) * Box(groove_d + 0.4 * MM, 2.0 * MM, w - 12 * MM)
    # recessed face ring and counterbored bolt circle on both faces
    for side in (1, -1):
        z_face = side * w / 2
        body -= Pos(0, 0, z_face) * (Cylinder(r - 9 * MM, 2.0 * MM) - Cylinder(r - 13 * MM, 2.0 * MM))
        for i in range(8):
            ang = 2 * math.pi * i / 8 + math.pi / 8
            x, y = a["bolt_circle"] / 2 * math.cos(ang), a["bolt_circle"] / 2 * math.sin(ang)
            body -= Pos(x, y, z_face) * Cylinder(2.8 * MM, 3.0 * MM)
    return body


def _disjoint(parts: list[PartSpec]) -> list[PartSpec]:
    """Subtract earlier parts from later ones (no shared volume inside a body)."""
    out: list[PartSpec] = []
    acc: Part | None = None
    for p in parts:
        shape = p.shape if acc is None else p.shape - acc
        out.append(PartSpec(p.name, shape, p.material, p.mass, p.rgba, p.visual))
        acc = p.shape if acc is None else acc + p.shape
    return out


# ---------------------------------------------------------------- bodies

def trunk_parts(spec: dict[str, Any], disjoint: bool = True) -> list[PartSpec]:
    t, a, lay = spec["trunk"], spec["actuator"], layout(spec)
    L, W, H = t["size"]
    f, wall = t["edge_fillet"], t["wall"]

    def rounded_box(lx: float, ly: float, lz: float, shrink: float) -> Part:
        """Intersection of three extruded rounded rectangles: plan (vertical corners), side (end
        top/bottom edges) and front (long edges). Robust where OCC can't blend mixed fillet radii."""
        plan = extrude(RectangleRounded(lx, ly, f["end_vertical"] - shrink), amount=lz / 2, both=True)
        side = Plane.XZ * RectangleRounded(lx, lz, f["end_top"] - shrink)
        side = extrude(side, amount=ly / 2, both=True)
        front = Plane.YZ * RectangleRounded(ly, lz, f["long"] - shrink)
        front = extrude(front, amount=lx / 2, both=True)
        return plan & side & front

    outer = rounded_box(L, W, H, 0.0)
    inner = rounded_box(L - 2 * wall, W - 2 * wall, H - 2 * wall, wall)
    shell = outer - inner
    shell_visual = outer
    hole_r = t["hub_hole_diameter"] / 2
    for sx in (1, -1):
        for sy in (1, -1):
            shell -= _cyl_x(hole_r, 4 * wall, (sx * L / 2, sy * lay["hy"], 0))
    # shallow battery-lid shut line on top (visual), 0.6 mm deep groove
    lid = Pos(0, 0, H / 2) * (Box(0.24, 0.16, 1.2 * MM) - Box(0.24 - 2.4 * MM, 0.16 - 2.4 * MM, 1.2 * MM))
    shell -= lid
    shell_visual -= lid
    # recessed cooling-vent grille on both side faces (1 mm deep, does not break through the wall)
    vt = t["vents"]
    for sy in (1, -1):
        for i in range(vt["count"]):
            z = (i - (vt["count"] - 1) / 2) * vt["pitch"]
            slot = fillet(Box(vt["size"][0], vt["depth"] * 2, vt["size"][1]).edges().filter_by(Axis.Y), 2.4 * MM)
            shell -= Pos(vt["x_center"], sy * (W / 2), z) * slot
            shell_visual -= Pos(vt["x_center"], sy * (W / 2), z) * slot
    bp = t["belly_plate"]
    belly = Pos(0, 0, -H / 2 - bp["thickness"] / 2) * fillet(
        Box(L - 2 * bp["inset"][0], W - 2 * bp["inset"][1], bp["thickness"]).edges().filter_by(Axis.Z), 0.02)

    floor_z = -H / 2 + wall
    bx, by, bz = t["battery"]["size"]
    battery = Pos(0, 0, floor_z + bz / 2) * fillet(Box(bx, by, bz).edges().filter_by(Axis.Z), 4 * MM)
    cx, cy, cz = t["compute"]["size"]
    compute = Pos(0, 0, floor_z + bz + cz / 2) * Box(cx, cy, cz)

    hx_, hy_, hz_ = t["harness"]["size"]
    harness = Pos(cx / 2 + hx_ / 2 + 0.005, 0, floor_z + bz + hz_ / 2) * Box(hx_, hy_, hz_)

    abd = []
    act = actuator_shape(spec)
    x_c = L / 2 - wall - a["width"] / 2           # housing face bolted against the end wall inner face
    for leg in LEG_ORDER:
        sx, sy = LEG_SIGNS[leg]
        abd.append(PartSpec(f"{leg}_abd_actuator", Pos(sx * x_c, sy * lay["hy"], 0) * Rot(Y=90) * act,
                            mass=a["mass"], rgba=COLORS["actuator"]))

    hd = t["head"]
    hx, hy, hz = hd["size"]
    hw = hd["wall"]
    head_outer = fillet(Box(hx, hy, hz).edges(), hd["fillet"])
    head_inner = Pos(0, 0, -hw / 2) * fillet(Box(hx - 2 * hw, hy - 2 * hw, hz - hw).edges(), hd["fillet"] - hw)
    head_shape = Pos(hd["x_front"] - hx / 2, 0, H / 2 + hz / 2) * (head_outer - head_inner)
    lens_r = hd["lens_diameter"] / 2
    lenses = Part()
    for sy in (1, -1):
        lenses += _cyl_x(lens_r, 6 * MM, (hd["x_front"], sy * 0.035, H / 2 + hz / 2))
    head_shape -= lenses
    glass = Part()
    for sy in (1, -1):
        # fills the lens bore through the 2 mm head wall, recessed 1 mm from the face
        glass += _cyl_x(lens_r, 2 * MM, (hd["x_front"] - 2 * MM, sy * 0.035, H / 2 + hz / 2))

    hn = t["handle"]
    rb = hn["bar_diameter"] / 2
    handle = Part()
    for sx in (1, -1):
        handle += Pos(hn["x_center"] + sx * hn["span"] / 2, 0, H / 2 + hn["height"] / 2) * Cylinder(rb, hn["height"])
        handle += Pos(hn["x_center"] + sx * hn["span"] / 2, 0, H / 2 + hn["height"]) * Sphere(rb)
    handle += _cyl_x(rb, hn["span"], (hn["x_center"], 0, H / 2 + hn["height"]))

    parts = [PartSpec("shell", shell, "aluminium_6061", rgba=COLORS["shell"], visual=shell_visual)] + abd + [
        PartSpec("battery", battery, mass=t["battery"]["mass"], rgba=COLORS["battery"]),
        PartSpec("compute", compute, mass=t["compute"]["mass"], rgba=COLORS["pcb"]),
        PartSpec("harness", harness, mass=t["harness"]["mass"], rgba=COLORS["cf"]),
        PartSpec("head", head_shape, "pa12_print", rgba=COLORS["accent"]),
        PartSpec("lens_glass", glass, "pa12_print", rgba=COLORS["glass"]),
        PartSpec("handle", handle, "pa12_print", rgba=COLORS["cf"]),
        PartSpec("belly_plate", belly, "carbon_fibre", rgba=COLORS["cf"]),
    ]
    return _disjoint(parts) if disjoint else parts


def hip_parts(spec: dict[str, Any], leg: str, disjoint: bool = True) -> list[PartSpec]:
    """Abduction output hub + bracket + hip-flexion actuator, in the hip (abduction) frame."""
    a, hp, lay = spec["actuator"], spec["legs"]["hip"], layout(spec)
    gap, tp = spec["clearance"]["joint_gap"], hp["plate_thickness"]
    ph = hp["plate_height"] / 2
    x0 = lay["end_outer_hip"] + gap                  # flange inboard face (clears the end wall)
    hub_x0 = lay["end_inner_hip"] + gap              # hub face (clears the abd actuator output face)
    hub = _cyl_x(a["output_hub_diameter"] / 2, x0 - hub_x0 + tp / 2, ((hub_x0 + x0 + tp / 2) / 2, 0, 0))
    # flange plate: disc round the hub, running out to the side plate
    y_side_in = lay["flex_y_in"] - tp
    disc = _cyl_x(ph, tp, (x0 + tp / 2, 0, 0))
    web = Pos(x0 + tp / 2, lay["flex_y_in"] / 2, 0) * Box(tp, lay["flex_y_in"], 2 * ph)
    # trim the disc where it would poke out past the side plate towards the flexion actuator
    keep = Pos(x0 + tp / 2, (lay["flex_y_in"] - 0.2) / 2, 0) * Box(4 * tp, 0.2 + lay["flex_y_in"], 4 * ph)
    flange = (disc + web) & keep
    # side plate bolted to the flexion actuator's inboard face
    x1 = HIP_SIDE_PLATE_X1
    side = Pos((x0 + x1) / 2, (y_side_in + lay["flex_y_in"]) / 2, 0) * Box(x1 - x0, tp, 2 * ph)
    side = fillet(side.edges().filter_by(Axis.Y), 6 * MM)
    actuator = Pos(0, lay["flex_y_c"], 0) * Rot(X=90) * actuator_shape(spec)
    parts = [
        PartSpec("flex_actuator", actuator, mass=a["mass"], rgba=COLORS["actuator"]),
        PartSpec("bracket", hub + flange + side, "aluminium_6061", rgba=COLORS["alu"]),
    ]
    return _mirror(_disjoint(parts) if disjoint else parts, leg)


def thigh_parts(spec: dict[str, Any], disjoint: bool = True) -> list[PartSpec]:
    """Knee actuator (also the hip-flexion output) + hollow belt housing + knee boss, thigh frame."""
    a, th, lay = spec["actuator"], spec["legs"]["thigh"], layout(spec)
    L = th["length"]
    w0, w1 = th["beam_width"]
    t, wall = th["beam_thickness"], th["wall"]
    actuator = Rot(X=90) * actuator_shape(spec)

    def section(z: float, width: float, thick: float) -> Sketch:
        return Plane.XY.offset(z) * Rectangle(width, thick)

    # Beam starts 35 mm below the axis: a 9-13 mm spigot into the actuator housing (designed embed),
    # and stops on the knee axis where the boss takes over.
    z_top = THIGH_BEAM_Z_TOP
    w_top = w0 + (w1 - w0) * (z_top / -L)
    outer = loft([section(z_top, w_top, t), section(-L, w1, t)])
    inner = loft([section(z_top - wall, w_top - 2 * wall, t - 2 * wall), section(-L + wall, w1 - 2 * wall, t - 2 * wall)])
    beam = outer - inner
    boss = _cyl_y(th["knee_boss_diameter"] / 2, 2 * lay["boss_half"], (0, 0, -L)) \
        - _cyl_y(th["bearing_bore"] / 2, 2 * lay["boss_half"], (0, 0, -L))
    # belt + pulleys: modelled as a thin strip inside the housing (catalogue mass)
    # belt runs from the actuator pulley (tangent to its rim) to the knee boss (tangent to its rim)
    belt_top, belt_bot = -a["diameter"] / 2, -L + th["knee_boss_diameter"] / 2
    belt = Pos(0, 0, (belt_top + belt_bot) / 2) * Box(w1 - 2 * wall - 4 * MM, t - 2 * wall - 4 * MM, belt_top - belt_bot)
    parts = [
        PartSpec("knee_actuator", actuator, mass=a["mass"], rgba=COLORS["actuator"]),
        PartSpec("boss", boss, "aluminium_6061", rgba=COLORS["alu"]),
        PartSpec("housing", beam, "aluminium_6061", rgba=COLORS["accent"]),
        PartSpec("belt", belt, mass=th["belt_and_pulleys_mass"], rgba=COLORS["cf"]),
    ]
    return _disjoint(parts) if disjoint else parts


def calf_parts(spec: dict[str, Any], disjoint: bool = True) -> list[PartSpec]:
    """Knee hub + hollow tapered CF tube + rubber foot, calf frame (origin on the knee axis)."""
    c, ft = spec["legs"]["calf"], spec["legs"]["foot"]
    L = c["length"]
    d0, d1 = c["tube_diameter"]
    bore = spec["legs"]["thigh"]["bearing_bore"]
    hub = _cyl_y(c["hub_diameter"] / 2, c["hub_width"]) - _cyl_y(bore / 2, c["hub_width"])
    tube_outer = Solid.make_cone(d0 / 2, d1 / 2, L, Plane.XY.offset(0).rotated((180, 0, 0)))
    tube_inner = Solid.make_cone(d0 / 2 - c["tube_wall"], d1 / 2 - c["tube_wall"], L,
                                 Plane.XY.rotated((180, 0, 0)))
    tube = tube_outer - tube_inner
    foot = Pos(0, 0, -L) * Sphere(ft["radius"])
    parts = [
        PartSpec("foot", foot, "polyurethane_rubber", rgba=COLORS["rubber"]),
        PartSpec("hub", hub, "aluminium_6061", rgba=COLORS["alu"]),
        PartSpec("tube", tube, "carbon_fibre", rgba=COLORS["cf"]),
    ]
    return _disjoint(parts) if disjoint else parts


def _mirror(parts: list[PartSpec], leg: str) -> list[PartSpec]:
    sx, sy = LEG_SIGNS[leg]
    out = []
    for p in parts:
        s = p.shape
        if sx < 0:
            s = mirror(s, about=Plane.YZ)
        if sy < 0:
            s = mirror(s, about=Plane.XZ)
        out.append(PartSpec(p.name, s, p.material, p.mass, p.rgba, p.visual))
    return out


def all_bodies(spec: dict[str, Any] | None = None, disjoint: bool = True) -> dict[str, list[PartSpec]]:
    """Body name (as in the MJCF) -> its parts in the body frame."""
    spec = spec or load_spec()
    bodies = {"base": trunk_parts(spec, disjoint)}
    thigh = thigh_parts(spec, disjoint)
    calf = calf_parts(spec, disjoint)
    for leg in LEG_ORDER:
        bodies[f"{leg}_hip"] = hip_parts(spec, leg, disjoint)
        bodies[f"{leg}_thigh"] = thigh
        bodies[f"{leg}_calf"] = calf
    return bodies
