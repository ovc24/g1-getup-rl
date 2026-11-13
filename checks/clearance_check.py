#!/usr/bin/env python3
"""Checks of the link clearance (getup/clearance.py):  python3 checks/clearance_check.py

Runs the real code with the numpy stand-in for torch (checks/faketorch.py) against a plain
loop of the same rule, on random poses. With scipy and a unitree_rl_mjlab checkout
(UNITREE_RL_MJLAB=<path>), also in the robot's own poses: the walking pose (no pair closer
than 2 cm), a right arm turned in (the right hand into the right hip) and against the
exact distances of the convex hulls."""
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import faketorch  # noqa: E402

sys.modules["torch"] = faketorch
import torch  # noqa: E402

from getup import clearance_shapes as cs  # noqa: E402
from getup.clearance import CONFLICT_M, LinkClearance, check_scene, quat_to_mat  # noqa: E402

OK = True


def check(cond, msg):
    global OK
    print(("  ok   " if cond else "  FAIL ") + msg)
    OK &= bool(cond)


def mat_np(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def loop_clearance(bodies, pos, rot, a, b):
    """The rule written as a plain loop (one env)."""
    dirs = np.array(cs.DIRS)
    best = math.inf
    for sa in (s for s in cs.SHAPES if s["body"] == a):
        for sb in (s for s in cs.SHAPES if s["body"] == b):
            ia, ib = bodies.index(a), bodies.index(b)
            wa = np.array(sa["points"]) @ rot[ia].T + pos[ia]
            wb = np.array(sb["points"]) @ rot[ib].T + pos[ib]
            in_b = ((wa - pos[ib]) @ rot[ib] @ dirs.T - np.array(sb["offsets"])).max(axis=1).min()
            in_a = ((wb - pos[ia]) @ rot[ia] @ dirs.T - np.array(sa["offsets"])).max(axis=1).min()
            best = min(best, in_b, in_a)
    return best


print("stand-in data")
check(len(cs.DIRS) == 26 and all(abs(np.linalg.norm(d) - 1) < 1e-5 for d in cs.DIRS),
      "26 unit directions")
check(all(len(s["points"]) == cs.N_POINTS and len(s["offsets"]) == 26 for s in cs.SHAPES),
      f"{len(cs.SHAPES)} shapes of {cs.N_POINTS} points and 26 offsets")
check(min(cs.WALK.values()) > 0.02, f"walking pose: every pair apart (closest "
      f"{100 * min(cs.WALK.values()):.2f} cm, exact)")

print("rotations")
rng = np.random.default_rng(3)
qs = rng.normal(size=(5, 4))
r = quat_to_mat(torch.tensor(qs)).numpy()
check(all(np.allclose(r[i], mat_np(qs[i]), atol=1e-6) for i in range(5)), "quat_to_mat = the usual matrix")

print("clearance = the rule, written as a loop")
lc = LinkClearance()
n_env = 3
nb = len(lc.bodies)
pos = rng.normal(scale=0.12, size=(n_env, nb, 3))       # bodies crowded together
qq = rng.normal(size=(n_env, nb, 4))
got = lc(torch.tensor(pos, dtype=torch.float32), torch.tensor(qq, dtype=torch.float32)).numpy()
rot = np.array([[mat_np(qq[e, i]) for i in range(nb)] for e in range(n_env)])
want = np.array([[loop_clearance(lc.bodies, pos[e], rot[e], a, b) for a, b in lc.pairs]
                 for e in range(n_env)])
check(got.shape == (n_env, len(lc.pairs)) and np.allclose(got, want, atol=2e-5),
      f"{n_env} envs x {len(lc.pairs)} pairs, largest difference {np.abs(got - want).max():.1e} m; "
      f"{int((want < -CONFLICT_M).sum())} conflicts among them")
far = pos.copy()
far[:, :, 0] += np.arange(nb)[None, :] * 3.0             # every body 3 m from the next
check(bool((lc(torch.tensor(far, dtype=torch.float32), torch.tensor(qq, dtype=torch.float32)).numpy()
            > 1.0).all()), "bodies far apart: every pair clear")

print("in the robot's poses (Unitree's scene)")
um = os.environ.get("UNITREE_RL_MJLAB")
scene = (Path(um).expanduser() / "src/assets/robots/unitree_g1/xmls/scene_g1_23dof.xml"
         if um else None)
try:
    import scipy  # noqa: F401
    have_scipy = True
except ImportError:
    have_scipy = False
if not (scene and scene.is_file() and have_scipy):
    print("  (no UNITREE_RL_MJLAB checkout or no scipy: skipped)")
else:
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_clearance_shapes as mk  # noqa: E402
    check(check_scene(scene) is None, f"stand-ins made from this scene ({cs.SCENE_SHA1})")
    robot = mk.Robot(scene)

    def measure(q):
        P, R = robot.pose(q)
        p = np.array([P[robot.index[b]] for b in lc.bodies])[None]
        m = np.array([R[robot.index[b]] for b in lc.bodies])[None]
        c = lc.measure(torch.tensor(p, dtype=torch.float32), torch.tensor(m, dtype=torch.float32))
        return dict(zip(lc.pairs, c.numpy()[0])), P, R

    walk, _, _ = measure(None)
    low = min(walk, key=walk.get)
    check(walk[low] > 0.015, f"walking pose: closest {low[0]} - {low[1]} {100 * walk[low]:.2f} cm "
          f"(exact {100 * cs.WALK[low]:.2f})")
    left = {p: v for p, v in walk.items() if "left" in p[0] and "right" not in p[1]}
    mirror = all(abs(v - walk[(a.replace("left", "right"), b.replace("left", "right"))]) < 0.003
                 for (a, b), v in left.items() if (a.replace("left", "right"),
                                                   b.replace("left", "right")) in walk)
    check(mirror, "walking pose: left and right the same (within 3 mm)")
    inward, P, R = measure({"right_shoulder_roll_joint": -0.18 + math.radians(10)})
    hip = min((p for p in inward if p[0] == "right_wrist_roll_rubber_hand" and "hip" in p[1]),
              key=inward.get)
    ex = min(mk.exact_distance(x, y) for x in robot.world(hip[0], P, R) for y in robot.world(hip[1], P, R))
    check(inward[hip] < -CONFLICT_M and ex < -CONFLICT_M and abs(inward[hip] - ex) < 0.015,
          f"right arm 10 deg in (shoulder roll): right hand - {hip[1]} {100 * inward[hip]:.1f} cm "
          f"(exact {100 * ex:.1f})")
    check(all(v > 0.015 for p, v in inward.items() if "left" in p[0] and "right" not in p[1]),
          "...the left arm untouched")

print()
print("ALL OK" if OK else "SOME CHECKS FAILED")
sys.exit(0 if OK else 1)
