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
from getup.sim2sim_core import (Episode, ObsHistory, ObsTerm,  # noqa: E402
                                add_conflicts, add_proxy, add_self_contacts, conflict_report,
                                group_report, is_standing, link_groups, mesh_lowest_z,
                                motor_torque, pose_report, proxy_report, report)

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

print("self-contacts")
se = Episode(start=0, settle_s=1.0, peak_qd=np.zeros(23))
EK = "left_elbow_link - left_knee_link"
PH = "left_hip_roll_link - pelvis"
for _ in range(10):        # 10 physics steps of 2 ms: an elbow on a knee, 30 N then 80 N
    add_self_contacts(se, {EK: 30.0, "a - b": 2.0}, 0.002)
for _ in range(5):
    add_self_contacts(se, {EK: 80.0, PH: 900.0}, 0.002, ignore={PH})
for _ in range(50):        # standing: the hand resting on the hip, 20 N
    add_self_contacts(se, {"left_hip_roll_link - left_wrist_roll_rubber_hand": 20.0}, 0.002,
                      standing=True)
check(abs(se.self_s - 0.03) < 1e-9 and abs(se.self_hard_s - 0.01) < 1e-9 and se.self_peak == 80.0,
      f"getting up: time above 5 N and 50 N, hardest: {se.self_s:.3f} s, {se.self_hard_s:.3f} s, {se.self_peak} N")
check("a - b" not in se.self_pairs and se.self_pairs[EK][:2] == [0.03, 0.01] or
      (abs(se.self_pairs[EK][0] - 0.03) < 1e-9 and abs(se.self_pairs[EK][1] - 0.01) < 1e-9),
      "2 N is touching, not pressing: not counted; per pair s > 5 N and s > 50 N")
check(PH in se.self_pairs and se.self_pairs[PH][2] == 900.0,
      "a pair near a joint is listed (900 N) but left out of the totals (hardest stays 80 N)")
check(abs(se.stand_self_s - 0.1) < 1e-9 and se.stand_self_hard_s == 0.0 and se.self_s < 0.04,
      "standing contacts are counted apart")
se2 = Episode(start=1, settle_s=1.0, peak_qd=np.zeros(23), self_at_start=40.0)
lines = report([se, se2], groups, [("arms", 0.01, "left_elbow_joint", 0.02)], "test", "mode",
               {PH: "near a joint", "left_hip_roll_link - left_wrist_roll_rubber_hand":
                "touches in the walking pose"})
text = "\n".join(lines)
gl = next(ln for ln in lines if "getting up (not standing)" in ln)
check("already pressing when the policy started: 50.0 %" in text and "[near a joint]" in text
      and "[touches in the walking pose]" in text and EK in text,
      "report: tangled starts, pairs with their notes")
check("> 50 N median 0.01 s" in gl, f"getting up line (medians of 2 episodes): {gl.strip()}")
pl = next(ln for ln in lines if ln.strip().startswith(EK))
check("0.01 s  0.01 s" in pl or "0.01 s 0.01 s" in pl or " 0.01 s" in pl, f"pair line, per episode: {pl.strip()}")

print("link conflicts")
NJ = "left_hip_roll_link - pelvis"
AT = "right_shoulder_yaw_link - torso_link"
HH = "left_hip_roll_link - left_wrist_roll_rubber_hand"
base = {HH: 0.004}             # the hand already 4 mm into the hip in the walking pose
c1 = Episode(start=0, settle_s=1.0, peak_qd=np.zeros(23), scenario="supine")
add_conflicts(c1, {AT: 0.006, HH: 0.010}, base)
add_conflicts(c1, {AT: 0.018, NJ: 0.05}, base)
c2 = Episode(start=1, settle_s=1.0, peak_qd=np.zeros(23), scenario="prone")
add_conflicts(c2, {HH: 0.013, AT: 0.004}, base)
check(abs(c1.conflicts[AT] - 0.018) < 1e-12 and abs(c1.conflicts[HH] - 0.006) < 1e-12
      and abs(c2.conflicts[HH] - 0.009) < 1e-12,
      "deepest per pair, beyond the walking pose (a hand 13 mm into the hip that is 4 mm in "
      "at rest: 9 mm)")
cl = conflict_report([c1, c2], {NJ})
ct = "\n".join(cl)
check("1 of 2" in ct and "prone 0/1" in ct and "supine 1/1" in ct and AT in ct and "1.8 cm" in ct
      and "near a joint, not counted" in ct and NJ in ct and HH not in ct.split("near a joint")[0].split("pairs, most")[-1],
      "conflict report: 1 of 2 scenarios (the arm 1.8 cm into the torso), the 9 mm hand not "
      "counted, the pair near a joint listed apart: " + cl[1].strip())

print("link conflicts: getting up and standing")
p1 = Episode(start=1, settle_s=1.0, peak_qd=np.zeros(23), scenario="prone")
over = add_conflicts(p1, {AT: 0.030, NJ: 0.05}, base, standing=False, dt=0.02, limit_m=0.01,
                     ignore={NJ})
over2 = add_conflicts(p1, {AT: 0.005}, base, standing=False, dt=0.02, limit_m=0.01, ignore={NJ})
for _ in range(10):                           # standing: the hand 8 cm into the hip, 0.2 s
    add_conflicts(p1, {HH: 0.084}, base, standing=True, dt=0.02, limit_m=0.01, ignore={NJ})
check(over == {AT: 0.030} and over2 == {} and abs(p1.conflict_up_s - 0.02) < 1e-12
      and abs(p1.conflict_stand_s - 0.2) < 1e-12 and abs(p1.conflicts_stand[HH] - 0.08) < 1e-12
      and HH not in p1.conflicts_up and abs(p1.conflicts[HH] - 0.08) < 1e-12,
      "per phase: pairs over the limit returned (near a joint left out), time per phase, "
      "deepest per phase and overall")
p2 = Episode(start=0, settle_s=1.0, peak_qd=np.zeros(23), scenario="supine")
for _ in range(5):
    add_conflicts(p2, {HH: 0.06}, base, standing=True, dt=0.02, limit_m=0.01, ignore={NJ})
p2.conflict_at_start = {AT: 0.02}
p3 = Episode(start=0, settle_s=1.0, peak_qd=np.zeros(23), scenario="supine")
add_conflicts(p3, {AT: 0.004}, base, standing=False, dt=0.02, limit_m=0.01, ignore={NJ})
pl = conflict_report([p1, p2, p3], {NJ})
pt = "\n".join(pl)
up_line = next(ln for ln in pl if "getting up (not standing)" in ln)
st_line = next(ln for ln in pl if ln.strip().startswith("standing "))
check("1 of 3" in up_line and "prone 1/1" in up_line and "supine 0/2" in up_line,
      "getting up: " + up_line.strip())
check("2 of 3" in st_line and "supine 1/2" in st_line, "standing: " + st_line.strip())
check("already in a conflict when the policy started: 1 of 3" in pt, "a start already in a conflict")
e1 = next(ln for ln in pl if "#1 " in ln)
e2 = next(ln for ln in pl if "#2 " in ln)
check(f"worst {HH} 8.0 cm (standing); getting up: {AT} 3.0 cm" in e1
      and f"worst {HH} 5.6 cm (standing); getting up: none" in e2,
      "per scenario: the worst, its phase, and the other phase")
row = next(ln for ln in pl if ln.strip().startswith(HH))
check("2/3  getting up   0/3  standing   2/3" in row, "pairs: any time, getting up, standing: "
      + row.strip())

print("link conflicts by part of the body, standing pose, the training's measure")
RH = "right_hip_roll_link - right_wrist_roll_rubber_hand"
RK = "right_knee_link - right_wrist_roll_rubber_hand"
LH = "left_hip_pitch_link - left_wrist_roll_rubber_hand"
EL = "left_elbow_link - torso_link"
LL = "left_knee_link - right_knee_link"
parts = link_groups([RH, RK, LH, EL, LL, NJ, AT])
check(parts["right hand - hips and legs"] == {RH, RK} and parts["left hand - hips and legs"] == {LH}
      and parts["arms - torso"] == {EL, AT},
      "parts: each hand against the hips and legs, the arms against the torso (legs against "
      "legs in none)")
q1 = Episode(start=1, settle_s=1.0, peak_qd=np.zeros(23), scenario="prone")
for _ in range(5):
    add_conflicts(q1, {AT: 0.03}, {}, standing=False, dt=0.02, limit_m=0.01, groups=parts)
for _ in range(50):
    add_conflicts(q1, {RH: 0.08, RK: 0.02}, {}, standing=True, dt=0.02, limit_m=0.01, groups=parts)
check(abs(q1.group_s["arms - torso"][0] - 0.1) < 1e-9 and abs(q1.group_s["right hand - hips and legs"][1] - 1.0) < 1e-9
      and abs(q1.group_deep["right hand - hips and legs"][1] - 0.08) < 1e-12
      and q1.group_s["right hand - hips and legs"][0] == 0.0,
      "per part: time with one of its pairs over the limit (two pairs at once count once), deepest")
q2 = Episode(start=0, settle_s=1.0, peak_qd=np.zeros(23), scenario="supine")
gl = group_report([q1, q2])
rh = next(ln for ln in gl if "right hand" in ln)
check(gl and "standing  1/2" in rh and "8.0 cm" in rh and "getting up  0/2" in rh
      and not any("left hand" in ln for ln in gl),
      "part report: " + rh.strip())
names = ["left_shoulder_roll_joint", "right_shoulder_roll_joint", "right_elbow_joint", "waist_yaw_joint"]
for e_, dev in ((q1, [0.05, 0.25, -0.02, 0.0]), (q2, [0.03, 0.21, -0.04, 0.01])):
    e_.success, e_.end_dev = True, np.array(dev)
pl = pose_report([q1, q2], names, top=2)
check(pl and "right_shoulder_roll_joint +13.2 (arm in)" in pl[0]
      and "left_shoulder_roll_joint +2.3 (arm out)" in pl[0] and "elbow" not in pl[0],
      "pose: the joints furthest from the walking pose, a shoulder roll marked in/out: " + pl[0][-70:])
r1 = Episode(start=1, settle_s=1.0, peak_qd=np.zeros(23), scenario="prone")
both = {RH: 0.07}
add_conflicts(r1, both, {}, standing=True, dt=0.02, limit_m=0.01)
add_proxy(r1, {RH: -0.06, LH: 0.03}, True, 0.01, {RH: 0.07})        # both see it
add_proxy(r1, {RH: 0.004, LH: 0.03}, True, 0.01, {RH: 0.012})       # only the exact shapes
add_proxy(r1, {RH: -0.02, LH: 0.03}, False, 0.01, {})               # only the stand-ins
add_proxy(r1, {RH: 0.02, LH: 0.03}, False, 0.01, {})                # neither
check(r1.agree == [1, 1, 1, 1] and abs(r1.proxy_stand[RH] - 0.06) < 1e-12
      and abs(r1.proxy_up[RH] - 0.02) < 1e-12,
      "the training's measure next to the exact one, step by step")
pr = proxy_report([r1], [RH, LH], {NJ})
check(len(pr) == 3 and "standing 1/1 (exact 1/1)" in pr[1] and "getting up 1/1 (exact 0/1)" in pr[1]
      and "both 33 %" in pr[2],
      "comparison report: " + pr[1].strip())
check(proxy_report([q2], [RH], {NJ}) == [], "no comparison when it was not measured")
s1 = Episode(start=1, settle_s=1.0, peak_qd=np.zeros(23), scenario="facedown")
s1.conflict_at_start = {AT: 0.015}
s1.inherited = {AT}
for dep in (0.015, 0.012):        # the start's own conflict, still there
    add_conflicts(s1, {AT: dep}, {}, standing=False, dt=0.02, limit_m=0.01)
inherited_new = dict(s1.new_up)
add_conflicts(s1, {AT: 0.005}, {}, standing=False, dt=0.02, limit_m=0.01)    # it clears
add_conflicts(s1, {AT: 0.03}, {}, standing=False, dt=0.02, limit_m=0.01)     # and comes back
add_conflicts(s1, {RH: 0.08}, {}, standing=True, dt=0.02, limit_m=0.01)
s2 = Episode(start=1, settle_s=1.0, peak_qd=np.zeros(23), scenario="facedown")
s2.conflict_at_start, s2.inherited = {AT: 0.02}, {AT}
add_conflicts(s2, {AT: 0.02}, {}, standing=False, dt=0.02, limit_m=0.01)
add_conflicts(s2, {}, {}, standing=True, dt=0.02, limit_m=0.01)
sl = conflict_report([s1, s2], {NJ})
wl = next(ln for ln in sl if "without those" in ln)
check(inherited_new == {} and abs(s1.new_up[AT] - 0.03) < 1e-12 and abs(s1.new_stand[RH] - 0.08) < 1e-12
      and s1.inherited == set() and "getting up 1 of 2" in wl and "standing 1 of 2" in wl,
      "the start's own conflicts: not counted until they first clear, then they count again: "
      + wl.strip())

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
