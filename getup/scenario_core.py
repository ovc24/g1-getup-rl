"""The parts of sim2sim_scenario.py without MuJoCo, so checks/ can test them:

- DeployPolicy: a policy run the way g1_ctrl runs it, from its deploy.yaml (a Python port
  of unitree_rl_mjlab/deploy/include/isaaclab: ManagerBasedRLEnv reset/step, the
  observation terms, clip-then-scale, the history per term, JointPositionAction);
- Controller: g1_ctrl's state machine with the get-up states of deploy/install_getup.py
  (Passive, Velocity, Fallen, GetUp), one tick per policy step;
- the scenario script (time, event, arguments) and the external forces (pushes on the
  torso, the elastic band of Unitree's simulator).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

DT = 0.02

DEFAULT_SCRIPT = """\
# time (s)  event   arguments        # what it is with g1_ctrl
0     walk                           # FixStand, then R2+A: the walking policy, sticks released
3     push  forward 300              # a shove on the torso for 0.1 s (forward = where it faces)
6     push  left 300
9     push  backward 300
12    push  right 300
15    cmd   0.5 0 0                  # left stick forward: walk at 0.5 m/s
18    push  left 300                 # a shove while walking
21    cmd   0 0 0.6                  # right stick: turn on the spot
24    cmd   0 0 0                    # sticks released: stand
26    push  forward 500 0.25         # too strong: it falls -> Fallen -> GetUp -> walking (by itself)
42    limp                           # L2+B: motors off, it collapses where it stands
46    getup                          # R2+X: get up
60    band  1.1                      # hung from the elastic band, motors off
64    release                        # key 9: band off, it drops
67    getup                          # R2+X
82    end
"""

EVENTS = {          # name: (min args, max args, help)
    "walk": (0, 0, "the walking policy (FixStand, then R2+A)"),
    "cmd": (3, 3, "VX VY WZ: joystick, m/s forward, m/s to the left, rad/s to the left"),
    "push": (2, 3, "DIR N [S]: horizontal force on the torso for S s (0.1); DIR: forward, "
                   "backward, left, right (where the robot faces) or degrees (90 = left)"),
    "limp": (0, 0, "Passive: motors limp (L2+B)"),
    "getup": (0, 0, "the get-up buttons (R2+X), from Passive or Fallen"),
    "band": (1, 1, "H: hang the robot from an elastic band, pelvis at about H m, motors limp"),
    "release": (0, 0, "band off (key 9): it drops"),
    "end": (0, 0, "stop"),
}
DIRECTIONS = {"forward": 0.0, "left": 90.0, "backward": 180.0, "right": -90.0}


@dataclass
class Event:
    t: float
    kind: str
    args: list
    line: int


def parse_script(text: str) -> list[Event]:
    """Lines 'time event args...' (# comments), sorted by time (stable)."""
    events = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            t = float(parts[0])
        except ValueError:
            raise ValueError(f"line {i}: '{parts[0]}' is not a time in seconds") from None
        if len(parts) < 2:
            raise ValueError(f"line {i}: no event after the time")
        kind, args = parts[1].lower(), parts[2:]
        if kind not in EVENTS:
            raise ValueError(f"line {i}: unknown event '{kind}' (events: {', '.join(EVENTS)})")
        lo, hi, _ = EVENTS[kind]
        if not lo <= len(args) <= hi:
            raise ValueError(f"line {i}: '{kind}' takes {lo}..{hi} arguments, got {len(args)}")
        if kind in ("cmd", "band"):
            args = [_num(a, i) for a in args]
        elif kind == "push":
            d = args[0].lower()
            ang = DIRECTIONS[d] if d in DIRECTIONS else _num(d, i)
            args = [ang, _num(args[1], i), _num(args[2], i) if len(args) > 2 else 0.1]
            if args[1] <= 0 or args[2] <= 0:
                raise ValueError(f"line {i}: push force and duration must be > 0")
        events.append(Event(t, kind, args, i))
    events.sort(key=lambda e: e.t)
    return events


def _num(s, line: int) -> float:
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"line {line}: '{s}' is not a number") from None


# -- a policy as g1_ctrl runs it ------------------------------------------------------------
TERM_DIMS = {"base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3, "gait_phase": 2}
JOINT_TERMS = ("joint_pos_rel", "joint_vel_rel", "last_action")


class DeployPolicy:
    """deploy.yaml + a network (obs -> action). Robot state `s` in the policy's joint order:
    s.q, s.qd (joint angles and speeds), s.gyro (pelvis angular velocity, body frame),
    s.gravity (gravity direction, body frame)."""

    def __init__(self, cfg: dict, net, name: str = "policy"):
        self.name, self.net = name, net
        self.step_dt = float(cfg["step_dt"])
        self.ids_map = [int(x) for x in cfg["joint_ids_map"]]
        n = self.n = len(self.ids_map)
        self.kp = np.asarray(cfg["stiffness"], float)
        self.kd = np.asarray(cfg["damping"], float)
        self.default = np.asarray(cfg["default_joint_pos"], float)
        for key, v in (("stiffness", self.kp), ("damping", self.kd), ("default_joint_pos", self.default)):
            if v.shape != (n,):
                raise ValueError(f"{name}: {key} has {v.size} values, joint_ids_map {n}")
        acts = cfg.get("actions") or {}
        if list(acts) != ["JointPositionAction"]:
            raise ValueError(f"{name}: only one JointPositionAction is supported, found {list(acts)}")
        a = acts["JointPositionAction"]
        if a.get("joint_ids") is not None:
            raise ValueError(f"{name}: actions with joint_ids are not supported")
        self.scale = np.ones(n) if a.get("scale") is None else np.asarray(a["scale"], float)
        self.offset = np.zeros(n) if a.get("offset") is None else np.asarray(a["offset"], float)
        self.clip = None if a.get("clip") is None else np.asarray(a["clip"], float).reshape(n, 2)
        obs = cfg["observations"]
        first = next(iter(obs.values()))
        if not (isinstance(first, dict) and "params" in first):
            raise ValueError(f"{name}: observation groups are not supported (one input 'obs')")
        self.scale_first = bool(obs.get("scale_first", False))
        self.gym_history = bool(obs.get("use_gym_history", False))
        self.terms = []
        for key, t in obs.items():
            if key in ("scale_first", "use_gym_history"):
                continue
            if key not in TERM_DIMS and key not in JOINT_TERMS:
                raise ValueError(f"{name}: observation '{key}' is not supported")
            params = t.get("params") or {}
            if key in JOINT_TERMS and (params.get("asset_cfg") or {}).get("joint_ids"):
                raise ValueError(f"{name}: '{key}' with joint_ids is not supported")
            dim = TERM_DIMS.get(key, n)
            scale = None if t.get("scale") is None else np.asarray(t["scale"], float).reshape(-1)
            if scale is not None and scale.size != dim:
                raise ValueError(f"{name}: '{key}' scale has {scale.size} values, expected {dim}")
            clip = None if t.get("clip") is None else (float(t["clip"][0]), float(t["clip"][1]))
            self.terms.append(SimpleNamespace(name=key, dim=dim, params=params, scale=scale,
                                              clip=clip, hist=int(t.get("history_length") or 1),
                                              buf=deque()))
        if self.gym_history and len({t.hist for t in self.terms}) != 1:
            raise ValueError(f"{name}: use_gym_history needs the same history for every term")
        ranges = ((cfg.get("commands") or {}).get("base_velocity") or {}).get("ranges") or {}
        self.ranges = [ranges.get(k) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")]
        self.size = sum(t.dim * t.hist for t in self.terms)
        self.phase = 0.0
        self.action = np.zeros(n)
        self.targets = self.offset.copy()

    def command(self, cmd) -> np.ndarray:
        """velocity_commands: the joystick clamped to the deploy.yaml ranges."""
        c = np.asarray(cmd, float).reshape(3).copy()
        for i, r in enumerate(self.ranges):
            if r is not None:
                c[i] = min(max(c[i], float(r[0])), float(r[1]))
        return c

    def _value(self, t, s, cmd) -> np.ndarray:
        n = t.name
        if n == "base_ang_vel":
            return np.asarray(s.gyro, float)
        if n == "projected_gravity":
            return np.asarray(s.gravity, float)
        if n == "velocity_commands":
            return self.command(cmd)
        if n == "gait_phase":                      # advances every time it is computed
            self.phase = math.fmod(self.phase + self.step_dt / float(t.params["period"]), 1.0)
            if np.linalg.norm(self.command(cmd)) < 0.1:
                return np.zeros(2)
            return np.array([math.sin(2.0 * math.pi * self.phase), math.cos(2.0 * math.pi * self.phase)])
        if n == "joint_pos_rel":
            return np.asarray(s.q, float) - self.default
        if n == "joint_vel_rel":
            return np.asarray(s.qd, float).copy()
        return self.action.copy()                  # last_action

    def _add(self, t, x) -> None:
        x = np.asarray(x, float).reshape(-1).copy()
        if x.size != t.dim:
            raise ValueError(f"{self.name}: '{t.name}' has {x.size} values, expected {t.dim}")
        if self.scale_first:
            if t.scale is not None:
                x = x * t.scale
            if t.clip is not None:
                x = np.clip(x, *t.clip)
        else:
            if t.clip is not None:
                x = np.clip(x, *t.clip)
            if t.scale is not None:
                x = x * t.scale
        t.buf.append(x)
        while len(t.buf) > t.hist:
            t.buf.popleft()

    def reset(self, s, cmd) -> None:
        """ManagerBasedRLEnv::reset: phase 0, last action 0, each term computed once and
        copied into its whole history."""
        self.phase = 0.0
        self.action = np.zeros(self.n)
        for t in self.terms:
            t.buf.clear()
            x = self._value(t, s, cmd)
            for _ in range(t.hist):
                self._add(t, x)

    def observation(self, s, cmd) -> np.ndarray:
        for t in self.terms:
            self._add(t, self._value(t, s, cmd))
        if self.gym_history:
            parts = [t.buf[h] for h in range(self.terms[0].hist) for t in self.terms]
        else:                                      # term after term, oldest first
            parts = [x for t in self.terms for x in t.buf]
        return np.concatenate(parts)

    def step(self, s, cmd) -> np.ndarray:
        """ManagerBasedRLEnv::step: observations -> network -> joint targets."""
        obs = self.observation(s, cmd)
        a = np.asarray(self.net(obs[None, :].astype(np.float32)), float).reshape(-1)
        if a.size != self.n:
            raise ValueError(f"{self.name}: the network gave {a.size} actions, expected {self.n}")
        self.action = a
        q = a * self.scale + self.offset
        if self.clip is not None:
            q = np.clip(q, self.clip[:, 0], self.clip[:, 1])
        self.targets = q
        return q


# -- g1_ctrl's states ---------------------------------------------------------------------------
@dataclass
class FsmCfg:
    fall_tilt: float = 1.0               # Velocity: bad_orientation (rad)
    passive_kd: np.ndarray | None = None  # per policy joint
    fallen_kd: np.ndarray | None = None
    auto_getup: bool = False
    fallen_settle_s: float = 1.0
    fallen_still_gyro: float = 0.5
    fallen_still_joint_vel: float = 1.0
    settle_s: float = 0.5                # GetUp: limp until still this long
    max_settle_s: float = 3.0
    still_gyro: float = 0.5
    still_joint_vel: float = 1.0
    min_time_s: float = 2.0
    hold_s: float = 0.5
    max_time_s: float = 10.0
    max_tilt: float = 0.35
    max_gyro: float = 0.8
    max_leg_dev: float = 0.5
    on_timeout: str = "passive"


def fsm_cfg_from_config(config: dict, ids_map: list[int]) -> FsmCfg:
    """The settings g1_ctrl reads from config.yaml (FSM section), defaults as in the states."""
    fsm = config.get("FSM") or {}
    pas, fal, gu = (fsm.get(k) or {} for k in ("Passive", "Fallen", "GetUp"))

    def per_joint(kd, fallback):
        if kd is None:
            return fallback
        kd = list(kd)
        if max(ids_map) >= len(kd):
            raise ValueError(f"kd has {len(kd)} values, joint_ids_map needs {max(ids_map) + 1}")
        return np.asarray([float(kd[i]) for i in ids_map])

    passive = per_joint(pas.get("kd"), np.full(len(ids_map), 3.0))
    c = FsmCfg(passive_kd=passive, fallen_kd=per_joint(fal.get("kd"), passive.copy()))
    c.auto_getup = bool(fal.get("auto_getup", False))
    c.fallen_settle_s = float(fal.get("settle_s", c.fallen_settle_s))
    c.fallen_still_gyro = float(fal.get("still_gyro", c.fallen_still_gyro))
    c.fallen_still_joint_vel = float(fal.get("still_joint_vel", c.fallen_still_joint_vel))
    for key in ("settle_s", "max_settle_s", "still_gyro", "still_joint_vel", "min_time_s",
                "hold_s", "max_time_s", "max_tilt", "max_gyro", "max_leg_dev"):
        if key in gu:
            setattr(c, key, float(gu[key]))
    c.on_timeout = "fallen" if str(gu.get("on_timeout", "Passive")) == "Fallen" else "passive"
    return c


LABELS = {
    "passive": "Passive: motors limp",
    "velocity": "Velocity: walking policy",
    "fallen": "Fallen: motors limp",
    "getup_wait": "GetUp: limp, waiting until still",
    "getup": "GetUp: get-up policy",
}


class Controller:
    """One tick per policy step (20 ms): first the automatic transitions (with the robot
    state of the last step), then what the motors do in this step:
        ("pd", joint targets, kp, kd)  or  ("limp", damping per joint).
    log(t, text) is called on every transition."""

    def __init__(self, cfg: FsmCfg, walk: DeployPolicy, getup: DeployPolicy, log=None):
        self.cfg, self.walk, self.getup = cfg, walk, getup
        self.log = log or (lambda t, text: None)
        self.state = "passive"
        self.cmd = np.zeros(3)
        self.still = self.waited = self.stand = 0.0
        self.policy_t0 = None
        self._reset_walk = False
        self.attempts = []                      # get-up attempts (dicts)

    # -- requests (buttons) ------------------------------------------------------------------
    def request(self, what: str, t: float) -> None:
        if what == "walk":
            self._enter("velocity", t, "walking policy on")
        elif what == "limp":
            self._enter("passive", t, "motors off")
        elif what == "getup":
            if self.state in ("passive", "fallen"):
                self._enter("getup_wait", t, "get-up buttons")
            else:
                self.log(t, f"get-up buttons ignored in {LABELS[self.state]}")
        else:
            raise ValueError(what)

    def _enter(self, state: str, t: float, why: str) -> None:
        if state == "getup_wait":
            self.attempts.append({"start": t, "policy": None, "end": None, "result": None,
                                  "head_N": 0.0})
        if self.state in ("getup_wait", "getup") and state not in ("getup_wait", "getup"):
            a = self.attempts[-1]
            a["end"], a["result"] = t, ("standing" if state == "velocity" else why)
        self.state = state
        self.still = self.waited = self.stand = 0.0
        self.policy_t0 = None
        self._reset_walk = state == "velocity"
        self.log(t, f"{why} -> {LABELS[state]}")

    # -- every step ----------------------------------------------------------------------------
    def tick(self, t: float, s) -> SimpleNamespace:
        c = self.cfg
        if self.state == "velocity" and s.tilt > c.fall_tilt:
            self._enter("fallen", t, f"fall detected (tilt {math.degrees(s.tilt):.0f} deg)")
        if self.state == "fallen":
            still = s.gyro_norm < c.fallen_still_gyro and s.qd_max < c.fallen_still_joint_vel
            self.still = self.still + DT if still else 0.0
            if c.auto_getup and self.still >= c.fallen_settle_s:
                self._enter("getup_wait", t, f"still for {c.fallen_settle_s:g} s")
        if self.state == "getup_wait":
            still = s.gyro_norm < c.still_gyro and s.qd_max < c.still_joint_vel
            self.still = self.still + DT if still else 0.0
            self.waited += DT
            if self.still >= c.settle_s or self.waited >= c.max_settle_s:
                why = "still" if self.still >= c.settle_s else f"not still after {c.max_settle_s:g} s"
                self.state, self.policy_t0, self.stand = "getup", t, 0.0
                self.attempts[-1]["policy"] = t
                self.getup.reset(s, np.zeros(3))
                self.log(t, f"{why} -> {LABELS['getup']}")

        if self.state == "velocity":
            if self._reset_walk:
                self.walk.reset(s, self.cmd)
                self._reset_walk = False
            q = self.walk.step(s, self.cmd)
            return SimpleNamespace(mode="pd", q=q, kp=self.walk.kp, kd=self.walk.kd)
        if self.state == "getup":
            q = self.getup.step(s, np.zeros(3))
            leg_dev = float(np.max(np.abs(np.asarray(s.q[:12]) - self.getup.default[:12])))
            up = s.tilt < c.max_tilt and s.gyro_norm < c.max_gyro and leg_dev < c.max_leg_dev
            self.stand = self.stand + DT if up else 0.0
            elapsed = t - self.policy_t0 + DT
            if elapsed >= c.min_time_s and self.stand >= c.hold_s:
                self._enter("velocity", t + DT, f"standing after {elapsed:.1f} s of get-up policy")
            elif elapsed >= c.max_time_s:
                self._enter(c.on_timeout, t + DT, f"not up after {c.max_time_s:g} s")
            return SimpleNamespace(mode="pd", q=q, kp=self.getup.kp, kd=self.getup.kd)
        kd = c.fallen_kd if self.state == "fallen" else c.passive_kd
        return SimpleNamespace(mode="limp", kd=kd)


# -- external forces --------------------------------------------------------------------------
class Forces:
    """Pushes (a horizontal force on the torso for a while, relative to where the pelvis
    faces) and the elastic band (Unitree's simulator: a spring-damper pulling the pelvis
    up, applied to the torso)."""

    def __init__(self, mass: float, g: float = 9.81, k: float = 1500.0, c: float = 400.0,
                 ramp_s: float = 1.5):
        self.mass, self.g, self.k, self.c, self.ramp_s = mass, g, k, c, ramp_s
        self.push = None                       # (force xy, until)
        self.band = None                       # dict

    def push_at(self, t: float, heading: float, angle_deg: float, newton: float, dur: float) -> None:
        a = heading + math.radians(angle_deg)
        self.push = (np.array([newton * math.cos(a), newton * math.sin(a)]), t + dur)

    def band_on(self, t: float, pos, height: float) -> None:
        sag = self.mass * self.g / self.k      # the spring stretch that carries the robot
        self.band = {"xy": np.asarray(pos[:2], float).copy(), "z0": float(pos[2]) + sag,
                     "z1": float(height) + sag, "t0": t}

    def band_off(self) -> None:
        self.band = None

    def force(self, t: float, pos, vel) -> np.ndarray:
        f = np.zeros(3)
        if self.push is not None:
            if t < self.push[1] - 1e-9:
                f[:2] += self.push[0]
            else:
                self.push = None
        if self.band is not None:
            b = self.band
            u = min(1.0, max(0.0, (t - b["t0"]) / self.ramp_s))
            anchor = np.array([b["xy"][0], b["xy"][1], b["z0"] + (b["z1"] - b["z0"]) * u])
            f += self.k * (anchor - np.asarray(pos, float)) - self.c * np.asarray(vel, float)
        return f
