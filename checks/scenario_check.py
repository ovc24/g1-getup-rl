#!/usr/bin/env python3
"""Offline checks of getup/scenario_core.py (numpy + PyYAML, no MuJoCo, no GPU): the
scenario script, g1_ctrl's policy loop for a deploy.yaml, the state machine, the forces.

    python3 checks/scenario_check.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from getup import scenario_core as sc  # noqa: E402

FAILS = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


# Unitree's walking deploy.yaml (g1_23dof), as g1_ctrl reads it
WALK_YAML = """
joint_ids_map: [0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,22,23,24,25,26]
step_dt: 0.02
stiffness: [40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2,
            14.3, 14.3, 14.3, 14.3, 14.3, 14.3, 14.3, 14.3, 14.3, 14.3]
damping:   [2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6,
            0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
default_joint_pos: [-0.1,0,0,0.3,-0.2,0, -0.1,0,0,0.3,-0.2,0,  0,  0.35,0.18,0,0.87,0, 0.35,-0.18,0,0.87,0]
commands:
  base_velocity:
    ranges:
      lin_vel_x: [-0.5, 1.0]
      lin_vel_y: [-0.5, 0.5]
      ang_vel_z: [-1.0, 1.0]
      heading: null
actions:
  JointPositionAction:
    clip: null
    joint_names: [.*]
    scale: [0.55, 0.35, 0.55, 0.35, 0.44, 0.44, 0.55, 0.35, 0.55, 0.35, 0.44, 0.44, 0.55,
            0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44]
    offset: [-0.1,0,0,0.3,-0.2,0, -0.1,0,0,0.3,-0.2,0,  0,  0.35,0.18,0,0.87,0, 0.35,-0.18,0,0.87,0]
    joint_ids: null
observations:
  base_ang_vel: {params: {}, clip: null, scale: [1.0, 1.0, 1.0], history_length: 1}
  projected_gravity: {params: {}, clip: null, scale: [1.0, 1.0, 1.0], history_length: 1}
  velocity_commands: {params: {command_name: base_velocity}, clip: null, scale: [1.0, 1.0, 1.0], history_length: 1}
  gait_phase: {params: {period: 0.6}, clip: null, scale: [1.0, 1.0], history_length: 1}
  joint_pos_rel: {params: {}, clip: null, scale: null, history_length: 1}
  joint_vel_rel: {params: {}, clip: null, scale: null, history_length: 1}
  last_action: {params: {}, clip: null, scale: null, history_length: 1}
"""
DEF = [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0.35, 0.18, 0, 0.87, 0, 0.35, -0.18, 0, 0.87, 0]
GETUP_CFG = yaml.safe_load(WALK_YAML)
GETUP_CFG.pop("commands")
GETUP_CFG["actions"]["JointPositionAction"]["scale"] = [1.0] * 23
GETUP_CFG["observations"] = {k: {"params": {}, "clip": None, "scale": None, "history_length": 5}
                             for k in ("base_ang_vel", "projected_gravity", "joint_pos_rel",
                                       "joint_vel_rel", "last_action")}
CONFIG_YAML = """
FSM:
  Passive:
    kd: [3,3,3,3,3,3, 3,3,3,3,3,3, 3,3,3, 3,3,3,3,3,3,3, 3,3,3,3,3,3,3,]
  Fallen: {auto_getup: true, settle_s: 1.0, still_gyro: 0.5, still_joint_vel: 1.0}
  GetUp: {settle_s: 0.5, max_settle_s: 3.0, min_time_s: 2.0, hold_s: 0.5, max_time_s: 10.0,
          max_tilt: 0.35, max_gyro: 0.8, max_leg_dev: 0.5, on_success: Velocity, on_timeout: Fallen}
"""


class Net:
    def __init__(self, n, value=0.1):
        self.n, self.value, self.seen = n, value, []

    def __call__(self, obs):
        self.seen.append(np.array(obs))
        return np.full((1, self.n), self.value)


def state(q=None, qd=0.0, gyro=(0.0, 0.0, 0.0), tilt=0.0):
    g = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
    qd = np.full(23, qd)
    return SimpleNamespace(q=np.array(DEF if q is None else q, float), qd=qd, gyro=np.array(gyro, float),
                           gravity=g, tilt=tilt, gyro_norm=float(np.linalg.norm(gyro)),
                           qd_max=float(np.abs(qd).max()))


print("scenario script")
ev = sc.parse_script(sc.DEFAULT_SCRIPT)
kinds = [e.kind for e in ev]
check(kinds[0] == "walk" and kinds[-1] == "end" and kinds.count("push") == 6 and "band" in kinds
      and "release" in kinds and kinds.count("getup") == 2, f"default scenario: {len(ev)} events")
p = [e for e in ev if e.kind == "push"]
check(p[0].args == [0.0, 300.0, 0.1] and p[1].args[0] == 90.0 and p[3].args[0] == -90.0
      and p[-1].args == [0.0, 500.0, 0.25], "push: direction (deg), force, duration")
check(sc.parse_script("5 push 45 200\n1 walk # x\n")[0].kind == "walk", "events sorted by time, comments")
for bad in ("1 jump", "x walk", "2 cmd 1 2", "3 push up 100", "4 push left -5", "5"):
    try:
        sc.parse_script(bad)
        check(False, f"refused: '{bad}'")
    except ValueError as e:
        check("line 1" in str(e), f"refused: '{bad}' ({e})")

print("walking policy, as g1_ctrl runs it")
net = Net(23)
walk = sc.DeployPolicy(yaml.safe_load(WALK_YAML), net, "walking")
check(walk.size == 80, f"inputs: {walk.size} (3+3+3+2+3x23 = 80)")
s0 = state(q=np.array(DEF) + 0.01, qd=0.1, gyro=(0.1, -0.2, 0.3))
walk.reset(s0, [0.5, 0.0, 0.0])
d = 0.02 / 0.6
check(abs(walk.phase - d) < 1e-12, "reset: gait phase advanced once (computed once at reset)")
walk.step(s0, [0.5, 0.0, 0.0])
o = net.seen[-1][0]
ph = 2 * d
check(np.allclose(o[:3], [0.1, -0.2, 0.3]) and np.allclose(o[3:6], s0.gravity)
      and np.allclose(o[6:9], [0.5, 0, 0]) and np.allclose(o[9:11], [math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)])
      and np.allclose(o[11:34], 0.01) and np.allclose(o[34:57], 0.1) and np.allclose(o[57:80], 0.0),
      "first step: gyro, gravity, command, phase (2 steps), joint pos - default, speeds, last action 0")
q = walk.step(s0, [0.5, 0.0, 0.0])
o = net.seen[-1][0]
check(np.allclose(o[57:80], 0.1), "second step: last action = the previous raw action")
check(np.allclose(q, 0.1 * walk.scale + np.array(DEF)), "joint targets = action x scale + offset")
check(np.allclose(walk.command([2.0, -1.0, 3.0]), [1.0, -0.5, 1.0]), "joystick clamped to the ranges")
walk.step(s0, [0.05, 0.0, 0.0])
check(np.allclose(net.seen[-1][0][9:11], 0.0), "gait phase input 0 when the command is below 0.1")

print("get-up policy (history 5 per term)")
gnet = Net(23, 0.2)
gu = sc.DeployPolicy(GETUP_CFG, gnet, "get-up")
check(gu.size == 375, f"inputs: {gu.size} (75 x 5)")
gu.reset(state(gyro=(1.0, 0.0, 0.0)), np.zeros(3))
gu.step(state(gyro=(2.0, 0.0, 0.0)), np.zeros(3))
o = gnet.seen[-1][0]
check(np.allclose(o[0:12:3], 1.0) and o[12] == 2.0 and np.allclose(o[15:30].reshape(5, 3), state().gravity),
      "term after term, oldest first, the reset value copied into the whole history")

print("states (config.yaml as installed with --auto)")
cfg = sc.fsm_cfg_from_config(yaml.safe_load(CONFIG_YAML), walk.ids_map)
check(cfg.auto_getup and cfg.on_timeout == "fallen" and np.allclose(cfg.passive_kd, 3.0)
      and cfg.passive_kd.size == 23 and cfg.settle_s == 0.5, "settings read; kd mapped to the 23 joints")
logs = []
walk2 = sc.DeployPolicy(yaml.safe_load(WALK_YAML), Net(23), "walking")
gu2 = sc.DeployPolicy(GETUP_CFG, Net(23, 0.0), "get-up")
ctl = sc.Controller(cfg, walk2, gu2, lambda t, x: logs.append((round(t, 2), x)))
t = 0.0


def run(seconds, s):
    global t
    out = None
    for _ in range(int(round(seconds / sc.DT))):
        out = ctl.tick(t, s)
        t += sc.DT
    return out


ctl.request("walk", t)
out = run(1.0, state())
check(ctl.state == "velocity" and out.mode == "pd" and np.allclose(out.kp, walk2.kp), "walking: PD with its gains")
ctl.request("getup", t)
check(ctl.state == "velocity" and "ignored" in logs[-1][1], "get-up buttons ignored while walking")
out = run(0.02, state(tilt=1.2, gyro=(2.0, 0, 0), qd=3.0))
check(ctl.state == "fallen" and out.mode == "limp" and "fall detected" in logs[-1][1], "tilt > 1 rad -> Fallen, limp")
run(0.5, state(tilt=1.5, gyro=(1.0, 0, 0), qd=2.0))
check(ctl.state == "fallen", "Fallen while still moving")
lying = state(q=np.zeros(23), tilt=1.5, gyro=(0.1, 0, 0), qd=0.2)
run(1.0, lying)
check(ctl.state == "getup_wait", "still for 1 s -> GetUp (limp, waiting)")
out = run(0.5, lying)
check(ctl.state == "getup" and out.mode == "pd" and ctl.attempts[-1]["policy"] is not None,
      "still for 0.5 s more -> get-up policy")
up = state(gyro=(0.1, 0, 0), qd=0.1)
run(1.0, up)
check(ctl.state == "getup", "standing, but before min_time_s: keeps the get-up policy")
run(1.2, up)
check(ctl.state == "velocity" and ctl.attempts[-1]["result"] == "standing", "standing for hold_s after 2 s -> Velocity")
ctl.request("limp", t)
run(0.2, lying)
ctl.request("getup", t)
run(0.6, lying)
check(ctl.state == "getup", "Passive + get-up buttons -> GetUp -> policy once still")
run(10.0, lying)
check(ctl.state == "fallen" and ctl.attempts[-1]["result"].startswith("not up"), "not up in 10 s -> Fallen (on_timeout)")
moving = state(q=np.zeros(23), tilt=1.5, gyro=(1.0, 0, 0), qd=2.0)
ctl.request("limp", t)
ctl.request("getup", t)
run(3.0, moving)
check(ctl.state == "getup" and "not still after 3" in logs[-1][1], "never still: the policy starts after max_settle_s")

print("forces")
f = sc.Forces(mass=32.0)
f.push_at(0.0, math.pi / 2, 0.0, 300.0, 0.1)          # robot faces +y; push forward
check(np.allclose(f.force(0.05, np.zeros(3), np.zeros(3)), [0.0, 300.0, 0.0]), "push forward = where it faces")
check(np.allclose(f.force(0.11, np.zeros(3), np.zeros(3)), 0.0), "push over after its duration")
f.push_at(0.0, 0.0, 90.0, 200.0, 0.1)
check(np.allclose(f.force(0.0, np.zeros(3), np.zeros(3)), [0.0, 200.0, 0.0]), "push left (heading 0) = +y")
f = sc.Forces(mass=32.0)
pos = np.array([1.0, 2.0, 0.75])
f.band_on(10.0, pos, 1.1)
f0 = f.force(10.0, pos, np.zeros(3))
check(np.allclose(f0, [0, 0, 32.0 * 9.81]), "band on: carries the weight, no jerk")
hang = np.array([1.0, 2.0, 1.1])
check(np.allclose(f.force(12.0, hang, np.zeros(3)), [0, 0, 32.0 * 9.81]), "after the ramp: holds the pelvis at 1.1 m")
f.band_off()
check(np.allclose(f.force(13.0, hang, np.zeros(3)), 0.0), "band off: no force")

print("\nALL OK" if not FAILS else f"\n{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
