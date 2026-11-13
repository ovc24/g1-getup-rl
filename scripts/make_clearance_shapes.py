#!/usr/bin/env python3
"""Make getup/clearance_shapes.py: stand-ins of the collision shapes of Unitree's deploy
scene (scene_g1_23dof.xml), cheap enough for training, so that training and evaluation
measure the link conflicts the self-collision benchmark measures in that scene.

What the deploy scene collides: per geom, the convex hull of its STL mesh (or a cylinder,
a sphere). Here, per geom, in its body's frame:
  - a 26-DOP: the hull's extent along 26 fixed directions (the faces, edges and corners of
    a cube), i.e. the hull wrapped in 26 planes;
  - 32 points spread over the hull (farthest-point sampling of its vertices).
Clearance of a pair of links (getup/clearance.py): for the points of one shape, the
largest signed distance to the planes of the other (negative inside: how deep), smallest
over the points, both ways, smallest over the shapes of the two links. Negative = one
link that far into the other. The check below compares it with the exact signed
distance of the hulls (the minimum translation that separates them, what MuJoCo's
mj_geomDistance gives) over random poses.

Pairs: an arm link (shoulder yaw, elbow, hand) and the torso, the pelvis, a leg link
(hip pitch, roll, yaw, knee, ankle pitch, ankle roll, either side) or an arm link of the
other arm. The benchmark's conflicts of v5 and v6 were all such pairs.

    python scripts/make_clearance_shapes.py \\
        --scene <unitree_rl_mjlab>/src/assets/robots/unitree_g1/xmls/scene_g1_23dof.xml \\
        --check 120
Needs numpy and scipy (convex hulls)."""
from __future__ import annotations

import argparse
import hashlib
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "getup" / "clearance_shapes.py"
# The walking pose (unitree_rl_mjlab's HOME_KEYFRAME): the getup policy's default pose.
WALK = {r".*_hip_pitch_joint": -0.1, r".*_knee_joint": 0.3, r".*_ankle_pitch_joint": -0.2,
        r".*_shoulder_pitch_joint": 0.35, r".*_elbow_joint": 0.87,
        r"left_shoulder_roll_joint": 0.18, r"right_shoulder_roll_joint": -0.18}
SIDES = ("left", "right")
ARMS = [f"{s}_{link}" for s in SIDES
        for link in ("shoulder_yaw_link", "elbow_link", "wrist_roll_rubber_hand")]
BODY = ["torso_link", "pelvis"] + [f"{s}_{link}" for s in SIDES for link in (
    "hip_pitch_link", "hip_roll_link", "hip_yaw_link", "knee_link", "ankle_pitch_link",
    "ankle_roll_link")]
PAIRS = ([(a, b) for a in ARMS for b in BODY]
         + [(a, b) for a in ARMS if a.startswith("left") for b in ARMS if b.startswith("right")])
N_POINTS = 32


def dirs26() -> np.ndarray:
    d = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                  if (x, y, z) != (0, 0, 0)], float)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def quat_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def axis_mat(a, t):
    a = np.asarray(a, float) / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def vec(s, default):
    return np.array([float(x) for x in s.split()]) if s is not None else np.array(default, float)


def fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def load_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            tri = np.frombuffer(data[84:], dtype=np.dtype(
                [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
            return np.unique(tri["v"].reshape(-1, 3).astype(float), axis=0)
    pts = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)
    return np.unique(np.array(pts, dtype=float), axis=0)


class Robot:
    """Bodies, hinge joints and the collision shapes (convex hull vertices per geom, in the
    body frame) of a MuJoCo XML, with forward kinematics (pelvis at the origin)."""

    def __init__(self, xml: Path):
        from scipy.spatial import ConvexHull
        root = ET.parse(xml).getroot()
        comp = root.find("compiler")
        meshdir = Path(xml).parent / ((comp.get("meshdir") if comp is not None else None) or "")
        files = {m.get("name"): meshdir / m.get("file") for m in root.iter("mesh")}
        self.bodies, self.shapes, cache = [], {}, {}

        def walk(el, parent):
            for b in el.findall("body"):
                j = b.find("joint")
                jn = axis = None
                if j is not None and j.get("type", "hinge") == "hinge":
                    jn, axis = j.get("name"), vec(j.get("axis"), (0, 0, 1))
                idx = len(self.bodies)
                self.bodies.append((b.get("name"), parent, vec(b.get("pos"), (0, 0, 0)),
                                    quat_mat(vec(b.get("quat"), (1, 0, 0, 0))), jn, axis))
                shapes = []
                for g in b.findall("geom"):
                    if g.get("contype") == "0" and g.get("conaffinity") == "0":
                        continue                          # visual only
                    gp = vec(g.get("pos"), (0, 0, 0))
                    gr = quat_mat(vec(g.get("quat"), (1, 0, 0, 0)))
                    typ = g.get("type", "sphere")
                    if typ == "mesh":
                        name = g.get("mesh")
                        if name not in cache:
                            v = load_stl(files[name])
                            cache[name] = v[ConvexHull(v).vertices]
                        local = cache[name]
                    elif typ == "cylinder":
                        r, h = vec(g.get("size"), (0, 0))[:2]
                        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
                        ring = np.c_[r * np.cos(t), r * np.sin(t)]
                        local = np.r_[np.c_[ring, np.full(48, -h)], np.c_[ring, np.full(48, h)]]
                    elif typ == "sphere":
                        local = float(vec(g.get("size"), (0,))[0]) * fib(26)
                    else:
                        continue
                    shapes.append(local @ gr.T + gp)
                if shapes:
                    self.shapes[b.get("name")] = shapes
                walk(b, idx)

        walk(root.find("worldbody"), -1)
        self.index = {b[0]: i for i, b in enumerate(self.bodies)}
        self.joints = [b[4] for b in self.bodies if b[4]]

    def pose(self, q: dict | None = None):
        ang = {}
        for jn in self.joints:
            ang[jn] = next((v for pat, v in WALK.items() if re.fullmatch(pat, jn)), 0.0)
        ang.update(q or {})
        pos, rot = [], []
        for _, parent, p, rq, jn, axis in self.bodies:
            p0, r0 = (np.zeros(3), np.eye(3)) if parent < 0 else (pos[parent], rot[parent])
            rb = r0 @ rq
            if jn:
                rb = rb @ axis_mat(axis, ang[jn])
            pos.append(p0 + r0 @ p)
            rot.append(rb)
        return pos, rot

    def world(self, body, pos, rot):
        i = self.index[body]
        return [s @ rot[i].T + pos[i] for s in self.shapes[body]]


def exact_distance(a: np.ndarray, b: np.ndarray, n=3000) -> float:
    """Signed distance of the convex hulls of two point sets: -min over unit u of
    (max a.u + max -b.u) (the support function of the Minkowski difference): the gap if
    > 0, minus the depth (minimum translation) if < 0."""
    def f(u):
        return (a @ u.T).max(axis=0) + (-(b @ u.T)).max(axis=0)
    d = fib(n)
    v = f(d)
    best = float(v.min())
    rng = np.random.default_rng(0)
    for u0 in d[np.argsort(v)[:4]]:
        u, fu, width = u0, float(f(u0[None])[0]), 0.06
        for _ in range(6):
            c = u + width * rng.normal(size=(300, 3))
            c /= np.linalg.norm(c, axis=1, keepdims=True)
            vc = f(c)
            k = int(np.argmin(vc))
            if vc[k] < fu:
                u, fu = c[k], float(vc[k])
            width *= 0.5
        best = min(best, fu)
    return -best


def farthest_points(pts: np.ndarray, n: int) -> np.ndarray:
    if len(pts) <= n:
        return pts[list(range(len(pts))) + [0] * (n - len(pts))]
    chosen = [int(np.argmax(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))]
    d = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(n - 1):
        chosen.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(pts - pts[chosen[-1]], axis=1))
    return pts[chosen]


MIRROR = np.array([1.0, -1.0, 1.0])      # left <-> right: y -> -y in the body frames


def fit(robot: Robot):
    """Left and right the same by construction (the policy is trained with mirrored
    episodes): a right shape is the mirror image of the left one (the meshes are mirror
    images, checked to 1 mm), a middle one (torso, pelvis) is made symmetric (half of its
    points and their mirror images, each plane as far out as its mirror plane)."""
    dirs = dirs26()
    mirror_dir = [int(np.argmin(np.linalg.norm(dirs - d * MIRROR, axis=1))) for d in dirs]
    bodies = sorted({x for p in PAIRS for x in p})
    missing = [b for b in bodies if b not in robot.shapes]
    if missing:
        raise SystemExit(f"no collision shapes for {missing} in this scene")
    shapes = []
    for b in bodies:
        if b.startswith("right_"):
            continue
        for pts in robot.shapes[b]:
            off = (pts @ dirs.T).max(axis=0)
            if b.startswith("left_"):
                shapes.append({"body": b, "points": farthest_points(pts, N_POINTS), "offsets": off})
            else:
                half = farthest_points(pts, N_POINTS // 2)
                shapes.append({"body": b, "points": np.r_[half, half * MIRROR],
                               "offsets": np.maximum(off, off[mirror_dir])})
    for b in bodies:
        if not b.startswith("right_"):
            continue
        left = [s for s in shapes if s["body"] == b.replace("right_", "left_", 1)]
        mirrored = [{"body": b, "points": s["points"] * MIRROR, "offsets": s["offsets"][mirror_dir]}
                    for s in left]
        real = [(pts @ dirs.T).max(axis=0) for pts in robot.shapes[b]]
        worst = max(min(np.abs(m["offsets"] - r).max() for m in mirrored) for r in real)
        if len(real) != len(mirrored) or worst > 0.001:
            raise SystemExit(f"{b} is not the mirror image of the left one ({1000 * worst:.1f} mm)")
        shapes += mirrored
    return dirs, shapes


def proxy_distance(dirs, shapes, a, b, pos, rot, index):
    """numpy version of getup/clearance.py (the same rule, one pair)."""
    best = np.inf
    for sa in (s for s in shapes if s["body"] == a):
        for sb in (s for s in shapes if s["body"] == b):
            ia, ib = index[a], index[b]
            wa = sa["points"] @ rot[ia].T + pos[ia]
            wb = sb["points"] @ rot[ib].T + pos[ib]
            in_b = ((wa - pos[ib]) @ rot[ib] @ dirs.T - sb["offsets"]).max(axis=1).min()
            in_a = ((wb - pos[ia]) @ rot[ia] @ dirs.T - sa["offsets"]).max(axis=1).min()
            best = min(best, in_b, in_a)
    return float(best)


def write(dirs, shapes, walk, scene: Path, out: Path):
    digest = hashlib.sha1(scene.read_bytes()).hexdigest()[:12]
    lines = ['"""Stand-ins of the deploy scene\'s collision shapes for getup/clearance.py.',
             f"Made by scripts/make_clearance_shapes.py from {scene.name} (sha1 {digest}):",
             "per collision geom, in its body's frame: the hull's extent along DIRS (offsets, m)",
             f"and {N_POINTS} points spread over the hull (m). PAIRS: the pairs of links measured;",
             "WALK: their exact clearance in the walking pose (m). Do not edit by hand.\"\"\"",
             "",
             f"SCENE_SHA1 = {digest!r}",
             f"N_POINTS = {N_POINTS}",
             "",
             "DIRS = (" + ", ".join("(" + ", ".join(f"{x:.6f}" for x in d) + ")" for d in dirs) + ")",
             "",
             "PAIRS = (",
             *[f"    ({a!r}, {b!r})," for a, b in PAIRS],
             ")",
             "",
             "WALK = {",
             *[f"    ({a!r}, {b!r}): {walk[(a, b)]:.4f}," for a, b in PAIRS],
             "}",
             "",
             "SHAPES = ("]
    for s in shapes:
        lines.append("    {")
        lines.append(f"        \"body\": {s['body']!r},")
        lines.append("        \"offsets\": (" + ", ".join(f"{x:.5f}" for x in s["offsets"]) + "),")
        lines.append("        \"points\": (")
        for p in s["points"]:
            lines.append("            (" + ", ".join(f"{x:.5f}" for x in p) + "),")
        lines.append("        ),")
        lines.append("    },")
    lines.append(")")
    out.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="Unitree's scene_g1_23dof.xml (with assets/)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", type=int, default=0, metavar="N",
                    help="compare with the exact distance in N random poses (slow: ~1 s each)")
    args = ap.parse_args(argv)
    scene = Path(args.scene).expanduser()
    robot = Robot(scene)
    dirs, shapes = fit(robot)
    pos, rot = robot.pose()
    walk, rows = {}, []
    for a, b in PAIRS:
        ex = min(exact_distance(x, y) for x in robot.world(a, pos, rot) for y in robot.world(b, pos, rot))
        px = proxy_distance(dirs, shapes, a, b, pos, rot, robot.index)
        walk[(a, b)] = ex
        rows.append((ex, px, a, b))
    write(dirs, shapes, walk, scene, Path(args.out))
    print(f"wrote {args.out}: {len(shapes)} shapes, {len(PAIRS)} pairs")
    print("closest pairs in the walking pose (exact, stand-in):")
    for ex, px, a, b in sorted(rows)[:8]:
        print(f"  {a:30s} {b:24s} {100 * ex:6.2f} cm  {100 * px:6.2f} cm")
    print(f"smallest stand-in clearance in the walking pose: {100 * min(r[1] for r in rows):.2f} cm")
    if args.check:
        rng = np.random.default_rng(1)
        ex_all, px_all = [], []
        for i in range(args.check):
            sd_arm, sd_leg = (0.9, 0.6) if i % 2 else (0.25, 0.12)
            base = dict(zip(robot.joints, [0.0] * len(robot.joints)))
            base.update({jn: next((v for pat, v in WALK.items() if re.fullmatch(pat, jn)), 0.0)
                         for jn in robot.joints})
            q = {jn: base[jn] + float(rng.normal(0, sd_arm if ("shoulder" in jn or "elbow" in jn)
                                                 else sd_leg)) for jn in robot.joints}
            pos, rot = robot.pose(q)
            for a, b in PAIRS:
                wa, wb = robot.world(a, pos, rot), robot.world(b, pos, rot)
                ca, cb = np.concatenate(wa), np.concatenate(wb)
                gap = (np.linalg.norm(ca.mean(0) - cb.mean(0)) - np.linalg.norm(ca - ca.mean(0), axis=1).max()
                       - np.linalg.norm(cb - cb.mean(0), axis=1).max())
                if gap > 0.06:
                    continue
                ex = min(exact_distance(x, y, 1500) for x in wa for y in wb)
                if ex > 0.06:
                    continue
                ex_all.append(ex)
                px_all.append(proxy_distance(dirs, shapes, a, b, pos, rot, robot.index))
        ex_all, px_all = np.array(ex_all), np.array(px_all)
        err = px_all - ex_all
        print(f"check: {len(ex_all)} pairs closer than 6 cm in {args.check} random poses; "
              f"stand-in minus exact (cm):")
        for lo, hi in ((-0.3, -0.03), (-0.03, -0.01), (-0.01, 0.01), (0.01, 0.03), (0.03, 0.06)):
            m = (ex_all >= lo) & (ex_all < hi)
            if m.any():
                print(f"  exact {100 * lo:5.1f} .. {100 * hi:4.1f} cm: n {int(m.sum()):4d}, median "
                      f"{100 * np.median(err[m]):5.2f}, p10 {100 * np.percentile(err[m], 10):5.2f}, "
                      f"p90 {100 * np.percentile(err[m], 90):5.2f}")
        real = ex_all < -0.01
        flag = px_all < -0.01
        print(f"  conflicts (> 1 cm deep): {int(real.sum())}; the stand-in finds "
              f"{int((real & flag).sum())}, misses {int((real & ~flag).sum())}, "
              f"flags {int((~real & flag).sum())} that are not")
    return 0


if __name__ == "__main__":
    sys.exit(main())
