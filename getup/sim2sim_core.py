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


def pct(x, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


def report(eps: list[Episode], groups: list[tuple[str, np.ndarray, float]],
           sat: list[tuple[str, float, str, float]], title: str, mode: str) -> list[str]:
    """groups: (label, joint indices, no-load speed); sat: (label, share of the 2 ms steps
    at the torque limit, worst joint, its share)."""
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
    return out
