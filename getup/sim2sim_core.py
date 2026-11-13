"""The numpy parts of sim2sim_getup.py (no MuJoCo, no torch), so checks/ can test them:
the policy input with its history, the motor model, the standing test, the report."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .poses import POSTURES
from .stand import STAND, StandCfg


@dataclass
class ObsTerm:
    name: str
    dim: int
    noise: tuple[float, float] | None     # uniform noise range added every step


class ObsHistory:
    """The get-up policy input exactly as mjlab's ObservationManager builds it:

    each term every step (compute -> uniform noise), then a history of H steps per term
    (oldest first) and the terms one after the other (term-major):
        [term1(t-H+1) .. term1(t), term2(t-H+1) .. term2(t), ...]
    After a reset the first frame fills the whole history (mjlab's CircularBuffer
    "backfill"; g1_ctrl's ObservationManager does the same when a state starts)."""

    def __init__(self, terms: list[ObsTerm], history: int, noise: bool,
                 rng: np.random.Generator):
        self.terms, self.h, self.noise, self.rng = terms, int(history), noise, rng
        self.buf: list[np.ndarray] | None = None

    @property
    def size(self) -> int:
        return sum(t.dim for t in self.terms) * self.h

    def reset(self) -> None:
        self.buf = None

    def __call__(self, values: dict[str, np.ndarray]) -> np.ndarray:
        frames = []
        for t in self.terms:
            x = np.asarray(values[t.name], dtype=np.float64).reshape(-1)
            if x.size != t.dim:
                raise ValueError(f"observation {t.name}: {x.size} values, expected {t.dim}")
            if self.noise and t.noise is not None:
                x = x + self.rng.uniform(t.noise[0], t.noise[1], size=x.shape)
            frames.append(x)
        if self.buf is None:
            self.buf = [np.tile(f, (self.h, 1)) for f in frames]
        else:
            self.buf = [np.vstack((b[1:], f[None, :])) for b, f in zip(self.buf, frames)]
        return np.concatenate([b.reshape(-1) for b in self.buf])[None, :].astype(np.float32)


def motor_torque(tau_pd, qd, tau_max, vmax, curve: bool):
    """Torque the motor can give: the PD torque clipped like mjlab's DcMotorActuator with
    saturation = effort limit: at joint speed v within [tau_max (-1 - v/vmax),
    tau_max (1 - v/vmax)] and +-tau_max, v clipped to +-2 vmax. curve=False: +-tau_max."""
    if not curve:
        return np.clip(tau_pd, -tau_max, tau_max)
    v = np.clip(qd, -2.0 * vmax, 2.0 * vmax)
    top = np.minimum(tau_max * (1.0 - v / vmax), tau_max)
    bottom = np.maximum(tau_max * (-1.0 - v / vmax), -tau_max)
    return np.clip(tau_pd, bottom, top)


def is_standing(pelvis_z: float, head_z: float, up: float, other_contact: bool,
                stand: StandCfg = STAND) -> bool:
    """Same test as training (getup_mdp.getup_state)."""
    return (pelvis_z > stand.pelvis_min and head_z > stand.head_min and up > stand.up_min
            and not other_contact)


def mesh_lowest_z(geom_pos: np.ndarray, geom_mat: np.ndarray, verts: np.ndarray) -> float:
    """Lowest world z of a mesh geom: vertices (n, 3) in the geom frame."""
    r = np.asarray(geom_mat, dtype=float).reshape(3, 3)
    return float(geom_pos[2] + (verts @ r[2]).min())


@dataclass
class Episode:
    start: int                      # posture class after settling (index into POSTURES)
    settle_s: float
    success: bool = False           # standing >= hold_s when the episode ends
    stood: bool = False             # stood >= hold_s at some point
    t_stand: float = math.nan       # when the first such stand began, s after the start
    head_s: float = 0.0             # head on the ground, s
    head_peak: float = 0.0          # hardest head-ground force, N
    blown: bool = False             # the simulation went wrong
    peak_qd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    end_pelvis: float = math.nan
    end_pose_err: float = math.nan  # rad RMS from the walking pose
    air_s: float = 0.0              # nothing on the ground (after the first 0.1 s), s
    peak_vz: float = 0.0            # fastest vertical pelvis speed, m/s
    peak_w: float = 0.0             # fastest pelvis angular speed, rad/s
    # Self-contacts while the policy runs: one part of the robot pressing on another.
    # Getting up = while not standing; pairs "near a joint" are left out of the totals.
    self_s: float = 0.0             # getting up: time with a self-contact > SELF_MIN_N, s
    self_hard_s: float = 0.0        # getting up: time with a self-contact > SELF_HARD_N, s
    self_peak: float = 0.0          # getting up: hardest self-contact, N
    stand_self_s: float = 0.0       # standing: the same three
    stand_self_hard_s: float = 0.0
    stand_self_peak: float = 0.0
    self_at_start: float = 0.0      # hardest self-contact when the policy starts, N
    self_pairs: dict = field(default_factory=dict)   # getting up: pair -> [s > min, s > hard, N]
    stand_pairs: dict = field(default_factory=dict)  # standing: the same
    # Link conflicts (--conflicts): one link into another, deepest per pair (m), beyond what
    # the pair already overlaps in the walking pose; and the start of a benchmark scenario.
    conflicts: dict = field(default_factory=dict)
    scenario: str = ""
    # The same split by phase (getting up = not standing), the time with a link deeper than
    # the limit into another in each phase (s), and the conflicts already there when the
    # policy started (links that went into each other while limp, before the policy).
    conflicts_up: dict = field(default_factory=dict)
    conflicts_stand: dict = field(default_factory=dict)
    conflict_up_s: float = 0.0
    conflict_stand_s: float = 0.0
    conflict_at_start: dict = field(default_factory=dict)
    # The conflicts the start already had (pairs over the limit when the policy started),
    # each until it first clears, and the conflicts without them ("new"), per phase.
    inherited: set = field(default_factory=set)
    new_up: dict = field(default_factory=dict)
    new_stand: dict = field(default_factory=dict)
    # Per part of the body (link_groups): [s getting up, s standing] with one of its pairs
    # deeper than the limit, and [deepest getting up, deepest standing] (m).
    group_s: dict = field(default_factory=dict)
    group_deep: dict = field(default_factory=dict)
    # Joints at the end minus the walking pose (rad), to see which ones hold a bad pose.
    end_dev: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # The training's measure (stand-ins of these shapes, getup/clearance.py) in the same
    # steps: deepest per pair (m) per phase, and policy steps [both see a conflict, only
    # the exact shapes, only the stand-ins, neither] (arm-body pairs).
    proxy_up: dict = field(default_factory=dict)
    proxy_stand: dict = field(default_factory=dict)
    agree: list = field(default_factory=lambda: [0, 0, 0, 0])


SELF_MIN_N = 5.0      # below: touching, not pressing (an arm resting is ~10-20 N)
SELF_HARD_N = 50.0    # above: about what an arm motor can push back with


def add_self_contacts(e: Episode, pair_forces: dict, dt: float, standing: bool = False,
                      ignore: frozenset | set = frozenset()) -> None:
    """One physics step of self-contacts: pair_forces = {"body - body": force N}. Every
    pair is kept (to list it); the pairs in `ignore` (near a joint) stay out of the totals."""
    pairs = e.stand_pairs if standing else e.self_pairs
    hardest = 0.0
    for pair, f in pair_forces.items():
        if f <= SELF_MIN_N:
            continue
        s = pairs.setdefault(pair, [0.0, 0.0, 0.0])
        s[0] += dt
        if f > SELF_HARD_N:
            s[1] += dt
        s[2] = max(s[2], f)
        if pair not in ignore:
            hardest = max(hardest, f)
    on, hard = hardest > SELF_MIN_N, hardest > SELF_HARD_N
    if standing:
        e.stand_self_s += dt * on
        e.stand_self_hard_s += dt * hard
        e.stand_self_peak = max(e.stand_self_peak, hardest)
    else:
        e.self_s += dt * on
        e.self_hard_s += dt * hard
        e.self_peak = max(e.self_peak, hardest)


def pct(x, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


def report(eps: list[Episode], groups: list[tuple[str, np.ndarray, float]],
           sat: list[tuple[str, float, str, float]], title: str, mode: str,
           self_flags: dict | None = None) -> list[str]:
    """groups: (label, joint indices, no-load speed); sat: (label, share of the 2 ms steps
    at the torque limit, worst joint, its share); self_flags: pair -> note (e.g. "near a
    joint", "touches in the walking pose")."""
    out = [f"=== {title}  |  {len(eps)} episodes ===", f"    {mode}"]
    blown = sum(e.blown for e in eps)
    if blown:
        out.append(f"WARNING: {blown} episodes ended because the simulation blew up")
    out.append(f"{'start':11s} {'n':>4s} {'success':>8s} {'stood up':>9s} {'t stand':>8s} "
               f"{'p90':>5s} {'fell again':>10s} {'head s':>7s} {'head N p90':>10s}")
    rows = [(name, [e for e in eps if e.start == i]) for i, name in enumerate(POSTURES)]
    rows.append(("all", eps))
    for name, sel in rows:
        if not sel:
            continue
        k = len(sel)
        succ = 100.0 * sum(e.success for e in sel) / k
        stood = 100.0 * sum(e.stood for e in sel) / k
        fell = 100.0 * sum(e.stood and not e.success for e in sel) / k
        ts = [e.t_stand for e in sel if e.stood]
        out.append(f"{name:11s} {k:4d} {succ:7.1f}% {stood:8.1f}% {pct(ts, 0.5):7.1f}s "
                   f"{pct(ts, 0.9):4.1f}s {fell:9.1f}% {np.mean([e.head_s for e in sel]):6.2f}s "
                   f"{pct([e.head_peak for e in sel], 0.9):9.0f}N")
    out.append("  start = how it lay after settling in this scene; success = standing (pelvis "
               "> 0.62 m, head > 1.0 m, < 32 deg tilt, only the feet on the ground) >= 1 s at "
               "the end; t stand = when the first 1 s stand began")
    ok = [e for e in eps if e.success]
    if ok:
        out.append(f"at the end, standing episodes: pelvis {pct([e.end_pelvis for e in ok], 0.5):.2f} m "
                   f"(median), joints {math.degrees(pct([e.end_pose_err for e in ok], 0.5)):.1f} deg "
                   f"RMS from the walking pose (median; p90 "
                   f"{math.degrees(pct([e.end_pose_err for e in ok], 0.9)):.1f})")
    out.append(f"settling before the policy starts: median {pct([e.settle_s for e in eps], 0.5):.1f} s")
    out.append("")
    out.append("how violent (a calm get-up: joints well below their no-load speed, no flight, "
               "motors rarely at their limit)")
    out.append("peak joint speed per episode vs motor no-load speed (real motor: 0 torque there)")
    for label, ids, vmax in groups:
        peak = np.array([e.peak_qd[ids].max() if e.peak_qd.size else np.nan for e in eps])
        over90 = 100.0 * float(np.mean(peak > 0.9 * vmax)) if peak.size else float("nan")
        out.append(f"  {label:22s} no-load {vmax:4.0f} rad/s   peak median {pct(peak, 0.5):5.1f}   "
                   f"p90 {pct(peak, 0.9):5.1f}   episodes above 90 %: {over90:5.1f} %")
    vz = [e.peak_vz for e in eps]
    w = [e.peak_w for e in eps]
    out.append(f"pelvis per episode: peak vertical speed median {pct(vz, 0.5):.2f} m/s "
               f"(p90 {pct(vz, 0.9):.2f}), peak angular speed median {pct(w, 0.5):.1f} rad/s "
               f"(p90 {pct(w, 0.9):.1f})")
    air = [e.air_s for e in eps if e.air_s > 0.0]
    line = (f"nothing on the ground (after the first 0.1 s): "
            f"{100.0 * len(air) / max(len(eps), 1):.1f} % of the episodes")
    if air:
        line += f"; in those, median {pct(air, 0.5):.2f} s, max {max(air):.2f} s"
    out.append(line)
    out.append("time at the torque limit (with the motor curve, the limit falls with speed)")
    for label, share, worst, wshare in sat:
        out.append(f"  {label:22s} {100 * share:5.1f} % of the 2 ms steps (most: {worst} "
                   f"{100 * wshare:.1f} %)")
    out.append("")
    out.extend(self_contact_report(eps, self_flags or {}))
    return out


def self_contact_report(eps: list[Episode], flags: dict, top: int = 8) -> list[str]:
    out = [f"self-contacts while the policy runs (one part of the robot pressing on another, "
           f"> {SELF_MIN_N:.0f} N; a link and its parent never collide)"]
    if not eps:
        return out
    k = len(eps)
    if any(v == "near a joint" or "near a joint" in v for v in flags.values()):
        out.append("  totals without the pairs 'near a joint' (two links one joint apart: in this "
                   "model their shapes overlap at the joint, not a movement of the policy)")
    for label, s5, s50, pk in (
            ("getting up (not standing)", [e.self_s for e in eps], [e.self_hard_s for e in eps],
             [e.self_peak for e in eps]),
            ("standing", [e.stand_self_s for e in eps], [e.stand_self_hard_s for e in eps],
             [e.stand_self_peak for e in eps])):
        out.append(f"  {label:26s} > {SELF_MIN_N:.0f} N median {pct(s5, 0.5):.2f} s (p90 "
                   f"{pct(s5, 0.9):.2f}), > {SELF_HARD_N:.0f} N median {pct(s50, 0.5):.2f} s (p90 "
                   f"{pct(s50, 0.9):.2f}); hardest median {pct(pk, 0.5):.0f} N (p90 {pct(pk, 0.9):.0f})")
    tangled = 100.0 * sum(e.self_at_start > SELF_MIN_N for e in eps) / k
    out.append(f"  already pressing when the policy started: {tangled:.1f} % of the episodes")
    for label, attr in (("getting up", "self_pairs"), ("standing", "stand_pairs")):
        tot: dict = {}
        for e in eps:
            for pair, (s5, s50, f) in getattr(e, attr).items():
                a = tot.setdefault(pair, [0.0, 0.0, 0.0, 0])
                a[0] += s5
                a[1] += s50
                a[2] = max(a[2], f)
                a[3] += 1
        if not tot:
            continue
        out.append(f"  pairs {label}, most time above {SELF_HARD_N:.0f} N first (per episode: s > "
                   f"{SELF_HARD_N:.0f} N, s > {SELF_MIN_N:.0f} N; hardest N; in how many episodes):")
        for pair, (s5, s50, f, n) in sorted(tot.items(), key=lambda kv: (-kv[1][1], -kv[1][0]))[:top]:
            note = f"  [{flags[pair]}]" if pair in flags else ""
            out.append(f"    {pair:52s} {s50 / k:5.2f} s {s5 / k:5.2f} s {f:6.0f} N  {n:3d}/{k}{note}")
    return out


def add_conflicts(e: Episode, depths: dict, baseline: dict, standing: bool | None = None,
                  dt: float = 0.0, limit_m: float | None = None,
                  ignore: frozenset | set = frozenset(), groups: dict | None = None) -> dict:
    """One policy step of link penetration depths (m) per pair of links: keep the deepest,
    beyond the pair's overlap in the walking pose (baseline). With `standing` (True or
    False) also per phase, and dt to the phase's time if a pair is deeper than limit_m
    (pairs in `ignore`, near a joint, left out), also per group of pairs (groups: name ->
    pairs, link_groups). Returns those pairs: {pair: depth}."""
    phase = None if standing is None else (e.conflicts_stand if standing else e.conflicts_up)
    over: dict = {}
    for pair, dep in depths.items():
        extra = dep - baseline.get(pair, 0.0)
        if extra > e.conflicts.get(pair, 0.0):
            e.conflicts[pair] = extra
        if phase is not None and extra > phase.get(pair, 0.0):
            phase[pair] = extra
        if limit_m is not None and extra > limit_m and pair not in ignore:
            over[pair] = extra
    if over and standing is not None:
        if standing:
            e.conflict_stand_s += dt
        else:
            e.conflict_up_s += dt
    if limit_m is not None and standing is not None:
        e.inherited &= set(over)          # a pair of the start that cleared counts again
        new = e.new_stand if standing else e.new_up
        for pair, dep in over.items():
            if pair not in e.inherited and dep > new.get(pair, 0.0):
                new[pair] = dep
    if groups and standing is not None:
        k = 1 if standing else 0
        for name, pairs in groups.items():
            deep = max((d for p, d in over.items() if p in pairs), default=0.0)
            if deep > 0.0:
                t = e.group_s.setdefault(name, [0.0, 0.0])
                t[k] += dt
                g = e.group_deep.setdefault(name, [0.0, 0.0])
                g[k] = max(g[k], deep)
    return over


HAND = "wrist_roll_rubber_hand"
LEG_PARTS = ("pelvis", "hip_", "knee", "ankle")
ARM_PARTS = ("shoulder_yaw", "elbow", HAND)


def link_groups(pairs) -> dict[str, frozenset]:
    """Pairs of links by part of the body: each hand against the hips and legs (pelvis,
    hip, knee, ankle links of either side), the arms against the torso."""
    groups: dict[str, set] = {"right hand - hips and legs": set(),
                              "left hand - hips and legs": set(), "arms - torso": set()}
    for p in pairs:
        a, b = p.split(" - ")
        for side in ("right", "left"):
            hand = f"{side}_{HAND}"
            if hand in (a, b) and any(k in (b if a == hand else a) for k in LEG_PARTS):
                groups[f"{side} hand - hips and legs"].add(p)
        if "torso_link" in (a, b) and any(k in (b if a == "torso_link" else a) for k in ARM_PARTS):
            groups["arms - torso"].add(p)
    return {k: frozenset(v) for k, v in groups.items()}


def group_report(eps: list[Episode], limit_m: float = 0.01) -> list[str]:
    """Per part of the body (link_groups): scenarios with a conflict, time, deepest."""
    names = sorted({g for e in eps for g in e.group_s}, key=lambda g: ("torso" in g, g))
    if not names:
        return []
    k = len(eps)
    out = ["  by part of the body (scenarios with a conflict; time per episode, median; deepest):"]
    for g in names:
        cells = []
        for ph, label in ((0, "getting up"), (1, "standing")):
            t = [e.group_s.get(g, [0.0, 0.0])[ph] for e in eps]
            n = sum(x > 0.0 for x in t)
            deep = max((e.group_deep.get(g, [0.0, 0.0])[ph] for e in eps), default=0.0)
            cells.append(f"{label} {n:2d}/{k} {pct(t, 0.5):5.2f} s {100 * deep:4.1f} cm")
        out.append(f"    {g:28s} " + "   ".join(cells))
    return out


def pose_report(eps: list[Episode], names: list[str], top: int = 6) -> list[str]:
    """The joints furthest from the walking pose at the end of the standing episodes
    (median, deg, signed); a shoulder roll is marked "arm in" when it turns the arm
    towards the body."""
    ok = [e.end_dev for e in eps if e.success and e.end_dev.size == len(names)]
    if not ok:
        return []
    med = np.degrees(np.median(np.stack(ok), axis=0))
    order = np.argsort(-np.abs(med))[:top]
    items = []
    for i in order:
        n = names[int(i)]
        note = ""
        if "shoulder_roll" in n:
            inward = med[i] > 0 if n.startswith("right") else med[i] < 0
            note = " (arm in)" if inward else " (arm out)"
        items.append(f"{n} {med[i]:+.1f}{note}")
    return ["at the end, standing episodes, the joints furthest from the walking pose (median, "
            "deg): " + ", ".join(items)]


def add_proxy(e: Episode, clear: dict, standing: bool, limit_m: float, exact_over: dict) -> None:
    """One policy step of the training's measure (stand-ins, pair -> signed clearance m)
    next to the exact one (exact_over: the pairs deeper than the limit now)."""
    phase = e.proxy_stand if standing else e.proxy_up
    p_over = False
    for name, c in clear.items():
        if -c > phase.get(name, 0.0):
            phase[name] = -c
        p_over |= -c > limit_m
    x_over = any(p in clear for p in exact_over)
    e.agree[0 if (x_over and p_over) else 1 if x_over else 2 if p_over else 3] += 1


def proxy_report(eps: list[Episode], pairs, near, limit_m: float = 0.01) -> list[str]:
    """The training's measure against the exact one, on the same steps and pairs."""
    if not any(sum(e.agree) for e in eps):          # not measured
        return []
    pairs = set(pairs) - set(near)
    k = len(eps)

    def n_over(attr):
        return sum(max((d for p, d in getattr(e, attr).items() if p in pairs), default=0.0)
                   > limit_m for e in eps)
    agree = np.sum([e.agree for e in eps], axis=0)
    either = max(int(agree[:3].sum()), 1)
    return ["  the training's measure (stand-ins of these shapes, getup/clearance.py) on the same "
            "steps, arm-body pairs:",
            f"    scenarios with a conflict: getting up {n_over('proxy_up')}/{k} (exact "
            f"{n_over('conflicts_up')}/{k}), standing {n_over('proxy_stand')}/{k} (exact "
            f"{n_over('conflicts_stand')}/{k})",
            f"    policy steps with a conflict in either: both {100 * agree[0] / either:.0f} %, only "
            f"the exact shapes {100 * agree[1] / either:.0f} %, only the stand-ins "
            f"{100 * agree[2] / either:.0f} %"]


def worst_pair(conflicts: dict, near: set | frozenset = frozenset()) -> tuple[str, float]:
    """The deepest pair (name, m) leaving out the pairs near a joint; ("", 0) if none."""
    items = [(p, d) for p, d in conflicts.items() if p not in near]
    return max(items, key=lambda kv: kv[1]) if items else ("", 0.0)


def conflict_report(eps: list[Episode], near: set, limit_m: float = 0.01, top: int = 10,
                    collide: bool = False) -> list[str]:
    """Scenarios in which a link goes more than limit_m into another one (pairs one joint
    apart, `near`, listed but not counted), as the self-collision benchmark counts them;
    split into getting up and standing when the episodes have the phases."""
    lim = 100.0 * limit_m
    out = [f"link conflicts: one link more than {lim:g} cm into another (links of the robot "
           f"{'collide' if collide else 'do NOT collide'} with each other here; a link and its "
           f"parent are not compared; depth beyond the walking pose)"]
    if not eps:
        return out
    labels = sorted({e.scenario or "all" for e in eps})

    def count(attr: str) -> tuple[int, str]:
        per = []
        for lb in labels:
            sel = [e for e in eps if (e.scenario or "all") == lb]
            per.append(f"{lb} {sum(worst_pair(getattr(e, attr), near)[1] > limit_m for e in sel)}"
                       f"/{len(sel)}")
        n = sum(worst_pair(getattr(e, attr), near)[1] > limit_m for e in eps)
        return n, ", ".join(per)

    n, per = count("conflicts")
    out.append(f"  scenarios with a conflict: {n} of {len(eps)} ({per})")
    phased = any(e.conflicts_up or e.conflicts_stand for e in eps)
    if phased:
        for name, attr, t_attr in (("getting up (not standing)", "conflicts_up", "conflict_up_s"),
                                   ("standing", "conflicts_stand", "conflict_stand_s")):
            n, per = count(attr)
            t = [getattr(e, t_attr) for e in eps]
            out.append(f"    {name:26s} {n:3d} of {len(eps)} ({per}); time with a conflict "
                       f"median {pct(t, 0.5):.2f} s (p90 {pct(t, 0.9):.2f})")
    n0 = sum(worst_pair(e.conflict_at_start, near)[1] > limit_m for e in eps)
    if n0:
        out.append(f"    already in a conflict when the policy started: {n0} of {len(eps)} "
                   "(links that went into each other while limp, before the policy)")
        if phased:
            nu, pu = count("new_up")
            ns, ps_ = count("new_stand")
            out.append(f"    without those (each until it first clears): getting up {nu} of "
                       f"{len(eps)} ({pu}), standing {ns} of {len(eps)} ({ps_})")
    for e_i, e in enumerate(eps):
        p, d = worst_pair(e.conflicts, near)
        if d <= limit_m:
            continue
        line = f"    {(e.scenario or 'episode'):9s} #{e_i + 1:<3d} worst {p} {100 * d:.1f} cm"
        if phased:
            pu, du = worst_pair(e.conflicts_up, near)
            ps, ds = worst_pair(e.conflicts_stand, near)
            up = f"{pu} {100 * du:.1f} cm" if du > limit_m else "none"
            st = f"{ps} {100 * ds:.1f} cm" if ds > limit_m else "none"
            if du >= ds:
                line += f" (getting up); standing: {st}"
            else:
                line += f" (standing); getting up: {up}"
        out.append(line)
    tot: dict = {}
    for e in eps:
        for key, src in (("all", e.conflicts), ("up", e.conflicts_up), ("stand", e.conflicts_stand)):
            for pair, d in src.items():
                if d > limit_m:
                    a = tot.setdefault(pair, {"all": 0, "up": 0, "stand": 0, "deep": 0.0})
                    a[key] += 1
                    a["deep"] = max(a["deep"], d)
    real = [(p, v) for p, v in tot.items() if p not in near]
    if real:
        k = len(eps)
        if phased:
            out.append("  pairs, most scenarios first (in how many: any time, getting up, "
                       "standing; deepest):")
        else:
            out.append("  pairs, most scenarios first (in how many, deepest):")
        for pair, v in sorted(real, key=lambda kv: (-kv[1]["all"], -kv[1]["deep"]))[:top]:
            ph = (f"  getting up {v['up']:3d}/{k}  standing {v['stand']:3d}/{k}" if phased else "")
            out.append(f"    {pair:52s} {v['all']:3d}/{k}{ph}  {100 * v['deep']:5.1f} cm")
    tot = {p: (v["all"], v["deep"]) for p, v in tot.items()}
    nj = [(p, v) for p, v in tot.items() if p in near]
    if nj:
        out.append("  near a joint, not counted (two links one joint apart overlap at that joint "
                   "in this model): " + "; ".join(f"{p} {n}/{len(eps)} {100 * d:.1f} cm"
                                                 for p, (n, d) in sorted(nj, key=lambda kv: -kv[1][0])))
    return out
