#!/usr/bin/env python3
"""Checks of the numpy parts of sim2sim_getup.py:  python3 checks/sim2sim_core_check.py

The policy input with history (mjlab's term-major layout, oldest step first, backfilled
after a reset), the motor model, the lowest point of a mesh, the standing test, the
report, and which policy file a run folder / checkpoint points to. MuJoCo and ONNX are
checked by the script itself on the PC (it stops with a message if the joints, actuators,
inputs or the ONNX metadata do not match)."""
import ast
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from getup import poses as ps  # noqa: E402
from getup.sim2sim_core import (Episode, ObsHistory, ObsTerm, is_standing,  # noqa: E402
                                mesh_lowest_z, motor_torque, report)

OK = True


def check(cond, msg):
    global OK
    print(("  ok   " if cond else "  FAIL ") + msg)
    OK = OK and bool(cond)


print("policy input with history")
terms = [ObsTerm("a", 2, None), ObsTerm("b", 3, None)]
h = ObsHistory(terms, history=3, noise=False, rng=np.random.default_rng(0))
f = [{"a": np.array([k, k + 0.5]), "b": np.array([10 * k, 10 * k + 1, 10 * k + 2])}
     for k in (1, 2, 3, 4)]
x1 = h(f[0])[0]
check(np.allclose(x1, [1, 1.5] * 3 + [10, 11, 12] * 3), "after a reset the first frame fills the history")
h(f[1])
x3 = h(f[2])[0]
want = [1, 1.5, 2, 2.5, 3, 3.5, 10, 11, 12, 20, 21, 22, 30, 31, 32]
check(np.allclose(x3, want), "term-major, oldest step first: [a(t-2) a(t-1) a(t)  b(t-2) b(t-1) b(t)]")
x4 = h(f[3])[0]
check(np.allclose(x4[:6], [2, 2.5, 3, 3.5, 4, 4.5]), "the oldest step drops out")
h.reset()
check(np.allclose(h(f[3])[0][:6], [4, 4.5] * 3), "reset: backfill again")
check(h.size == 15 and x4.dtype == np.float32, "size and dtype")
hn = ObsHistory([ObsTerm("a", 4, (-0.2, 0.2))], history=5, noise=True, rng=np.random.default_rng(1))
xs = np.array([hn({"a": np.zeros(4)})[0] for _ in range(200)])
last = xs[:, -4:]
check(np.all(np.abs(last) <= 0.2) and abs(last.mean()) < 0.03 and last.std() > 0.08,
      "uniform noise within +-0.2 on every step")
check(np.allclose(xs[1:, -8:-4], xs[:-1, -4:]), "the noisy frame is stored as it was seen (noise before history)")

print("motor model")
tau = motor_torque(np.array([100.0, -100.0, 100.0, 100.0]), np.array([0.0, 0.0, 10.0, -30.0]),
                   np.array([50.0] * 4), np.array([20.0] * 4), curve=True)
check(np.allclose(tau, [50, -50, 25, 50]), f"curve: at rest +-50, at 10 of 20 rad/s 25, braking 50: {tau}")
check(np.allclose(motor_torque(np.array([100.0]), np.array([10.0]), np.array([50.0]),
                               np.array([20.0]), curve=False), [50]), "no curve: +-limit")

print("geometry and standing")
cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float) * 0.1
r45 = ps.quat_to_mat(ps.quat_from_rpy(math.pi / 4, 0, 0)).reshape(9)
check(abs(mesh_lowest_z(np.array([0, 0, 1.0]), np.eye(3).reshape(9), cube) - 0.9) < 1e-9
      and abs(mesh_lowest_z(np.array([0, 0, 1.0]), r45, cube) - (1 - 0.1 * math.sqrt(2))) < 1e-9,
      "lowest vertex of a mesh, also rotated")
check(is_standing(0.76, 1.23, 0.99, False) and not is_standing(0.76, 1.23, 0.99, True)
      and not is_standing(0.45, 0.95, 0.99, False) and not is_standing(0.76, 1.23, 0.80, False),
      "standing test as in training")

print("report")
eps = [Episode(start=0, settle_s=1.2, success=True, stood=True, t_stand=2.1, head_s=0.1,
               head_peak=30.0, peak_qd=np.full(23, 8.0), end_pelvis=0.76, end_pose_err=0.05,
               peak_vz=0.5, peak_w=1.0),
       Episode(start=1, settle_s=0.9, success=False, stood=True, t_stand=3.0, head_s=0.0,
               head_peak=0.0, peak_qd=np.full(23, 12.0), air_s=0.12, peak_vz=2.0, peak_w=6.0),
       Episode(start=1, settle_s=3.0, peak_qd=np.full(23, 25.0), blown=True, peak_vz=1.0,
               peak_w=2.0)]
groups = [("arms", np.arange(13, 23), 37.0), ("legs", np.arange(0, 12), 20.0)]
lines = report(eps, groups, [("arms", 0.01, "left_elbow_joint", 0.02)], "test", "mode")
text = "\n".join(lines)
check("supine" in text and "prone" in text and "all" in text and "blew up" in text,
      "one row per starting posture + all, blow-ups flagged")
row = next(ln for ln in lines if ln.startswith("prone"))
check(" 0.0%" in row and "50.0%" in row, f"prone: success 0 %, stood 50 %, fell again 50 %: {row}")
check(any("above 90 %:  33.3 %" in ln for ln in lines if "legs" in ln),
      "legs above 90 % of their no-load speed (18 of 20 rad/s) in 1 of 3 episodes")
air = next(ln for ln in lines if ln.startswith("nothing on the ground"))
pel = next(ln for ln in lines if ln.startswith("pelvis per episode"))
check("33.3 % of the episodes" in air and "max 0.12 s" in air,
      f"flight: 1 of 3 episodes, 0.12 s: {air}")
check("median 1.00 m/s" in pel and "median 2.0 rad/s" in pel, f"pelvis speeds (medians): {pel}")

print("policy file")
src = (ROOT / "sim2sim_getup.py").read_text()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "resolve_policy_file")
ns = {"re": re, "Path": Path}
exec(compile(ast.Module([fn], []), "x", "exec"), ns)
resolve = ns["resolve_policy_file"]
d = Path(tempfile.mkdtemp())
for k in (900, 1300):
    (d / f"model_{k}.pt").write_bytes(b"x")
(d / "policy.onnx").write_bytes(b"y")
now = time.time()
os.utime(d / "model_900.pt", (now - 3000, now - 3000))
os.utime(d / "model_1300.pt", (now - 1, now - 1))
os.utime(d / "policy.onnx", (now, now))
check(resolve(str(d)).name == "policy.onnx" and resolve(str(d / "model_1300.pt")).name == "policy.onnx",
      "run folder and the last checkpoint -> policy.onnx")
try:
    resolve(str(d / "model_900.pt"))
    check(False, "an older checkpoint is refused")
except SystemExit as e:
    check("model_1300.pt" in str(e), "an older checkpoint is refused, the message names the last one")

print()
print("ALL OK" if OK else "SOME CHECKS FAILED")
sys.exit(0 if OK else 1)
