"""Fallen poses: where the get-up episodes start.

Every get-up episode starts from a pose in which the robot already lies on the ground, at
rest. The poses come from a pool made once, offline, by dropping the robot many times in
plain MuJoCo (CPU) with the same scene model as the training (``make_fallen_poses.py``):

  1. orientation: lying on the back (supine), on the front (prone), on the left or right
     side (+-25 deg of tilt), completely random, STANDING (+-10 deg) on its feet with a
     shove (it tips over and collapses, like a real fall), or COLLAPSE: standing (+-5 deg)
     in about the walking pose, barely moving, motors limp from the start (what the robot
     does when its motors go limp while it stands): knees and hips fold where it stands
     and it ends up lying, mostly on the back or the front, the legs folded under or
     beside it (a limp robot does not stay kneeling: nothing holds the hips); random
     heading;
  2. joints: a random target pose between the walking pose and a random pose within
     ``JOINT_SAMPLE_RANGES``; poses where limbs go through each other are redrawn;
  3. released a few cm above the ground (lowest point 2-25 cm), with a little random
     velocity, and simulated until everything is still (at least 0.8 s, at most 4 s), with
     the motors either limp (damping only, like the robot's damping mode), holding the
     random pose, or holding it first and going limp after 0.2-0.8 s;
  4. the resting pose is kept (root height and orientation, joint angles) together with
     how it ended up lying (``POSTURES``, from the pelvis axes, see ``classify``).

Why not drop the robot at the start of each training episode instead (as HoST does)?
Then the first ~1 s of every episode would be falling, the policy could not act yet, and
the spawn height would need forward kinematics on the GPU at every reset. Starting at rest
from a pool is cheaper and every step of the episode is useful. Training only re-draws the
heading and the position on the ground.

numpy only at import time; ``generate`` needs mujoco (installed with mjlab).
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

POSTURES = ("supine", "prone", "left_side", "right_side", "upright", "other")
DROPS = ("supine", "prone", "left_side", "right_side", "random", "standing", "collapse")
MODES = ("limp", "hold", "hold_then_limp")
DEFAULT_POOL = Path(__file__).resolve().parent.parent / "data" / "fallen_poses.npz"

# Where the joints may be when the robot hits the ground (rad, by joint-name regex). The
# fall itself (limp joints) can move them further. Narrower than the joint limits on
# purpose: a thigh 2.8 rad behind the body is not how a robot falls.
JOINT_SAMPLE_RANGES: tuple[tuple[str, tuple[float, float]], ...] = (
    (r".*_hip_pitch_joint", (-1.8, 0.5)),
    (r"left_hip_roll_joint", (-0.3, 1.0)),
    (r"right_hip_roll_joint", (-1.0, 0.3)),
    (r".*_hip_yaw_joint", (-0.8, 0.8)),
    (r".*_knee_joint", (0.0, 2.2)),
    (r".*_ankle_pitch_joint", (-0.8, 0.45)),
    (r".*_ankle_roll_joint", (-0.25, 0.25)),
    (r"waist_yaw_joint", (-0.8, 0.8)),
    (r".*_shoulder_pitch_joint", (-2.2, 1.6)),
    (r"left_shoulder_roll_joint", (-0.3, 1.8)),
    (r"right_shoulder_roll_joint", (-1.8, 0.3)),
    (r".*_shoulder_yaw_joint", (-1.2, 1.2)),
    (r".*_elbow_joint", (-0.8, 1.9)),
    (r".*_wrist_roll_joint", (-1.2, 1.2)),
)


# -- small helpers (numpy, quaternions w x y z) -------------------------------------------
def resolve_regex_values(table, names, default=None):
    """{regex: value} or ((regex, value), ...) -> one value per name (first full match)."""
    items = list(table.items()) if isinstance(table, dict) else list(table)
    out = []
    for n in names:
        for pat, val in items:
            if re.fullmatch(pat, n):
                out.append(val)
                break
        else:
            if default is None:
                raise KeyError(f"no entry matches joint '{n}'")
            out.append(default)
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Same convention as mjlab's quat_from_euler_xyz: q = qz(yaw) * qy(pitch) * qx(roll)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    ])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def random_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random rotation (Shoemake)."""
    u1, u2, u3 = rng.random(3)
    a, b = math.sqrt(1 - u1), math.sqrt(u1)
    x, y, z, w = (a * math.sin(2 * math.pi * u2), a * math.cos(2 * math.pi * u2),
                  b * math.sin(2 * math.pi * u3), b * math.cos(2 * math.pi * u3))
    return np.array([w, x, y, z])


def drop_orientation(drop: str, rng: np.random.Generator, tilt_deg: float = 25.0) -> np.ndarray:
    """Pelvis orientation at release. Body axes: x forward (chest), y left, z up.
    supine: x points up (lying on the back); prone: x points down; left_side: y down."""
    if drop == "random":
        return random_quat(rng)
    t = math.radians(tilt_deg) * rng.uniform(-1.0, 1.0, size=2)
    if drop == "standing":
        t = 0.4 * t
    elif drop == "collapse":
        t = 0.2 * t
    rpy = {
        "standing": (t[0], t[1], 0.0),
        "collapse": (t[0], t[1], 0.0),
        "supine": (t[0], -math.pi / 2 + t[1], 0.0),
        "prone": (t[0], math.pi / 2 + t[1], 0.0),
        "left_side": (-math.pi / 2 + t[0], t[1], 0.0),
        "right_side": (math.pi / 2 + t[0], t[1], 0.0),
    }[drop]
    yaw = rng.uniform(-math.pi, math.pi)
    return quat_mul(quat_from_rpy(0.0, 0.0, yaw), quat_from_rpy(*rpy))


def classify(quat: np.ndarray, threshold: float = 0.6) -> int:
    """How the robot lies, from the world-z components of the pelvis axes (index into
    POSTURES). fx > 0: the chest faces up (supine); fy > 0: the left side is up, so it
    lies on its right side."""
    r = quat_to_mat(np.asarray(quat, dtype=float))
    f = r[2]                                    # (fx, fy, fz)
    k = int(np.argmax(np.abs(f)))
    if abs(f[k]) < threshold:
        return POSTURES.index("other")
    if k == 0:
        return POSTURES.index("supine" if f[0] > 0 else "prone")
    if k == 1:
        return POSTURES.index("right_side" if f[1] > 0 else "left_side")
    return POSTURES.index("upright" if f[2] > 0 else "other")


# Geom type ids (mujoco.mjtGeom), so this module imports without mujoco.
_PLANE, _SPHERE, _CAPSULE, _ELLIPSOID, _CYLINDER, _BOX = 0, 2, 3, 4, 5, 6


def lowest_point(types, sizes, pos, mats, rbound) -> float:
    """Lowest world z over a set of geoms (exact for spheres, capsules, cylinders, boxes;
    bounding sphere for the rest). mats: (n, 9) row-major rotation matrices."""
    z = math.inf
    for t, s, p, m, rb in zip(types, sizes, pos, mats, rbound):
        r = np.asarray(m).reshape(3, 3)
        if t == _SPHERE:
            lo = p[2] - s[0]
        elif t == _CAPSULE:                       # axis = local z, half-length s[1]
            lo = p[2] - abs(r[2, 2]) * s[1] - s[0]
        elif t == _CYLINDER:
            lo = p[2] - abs(r[2, 2]) * s[1] - s[0] * math.sqrt(max(0.0, 1.0 - r[2, 2] ** 2))
        elif t == _BOX:
            lo = p[2] - (abs(r[2, 0]) * s[0] + abs(r[2, 1]) * s[1] + abs(r[2, 2]) * s[2])
        elif t == _PLANE:
            continue
        else:
            lo = p[2] - rb
        z = min(z, float(lo))
    return z


# -- the pool ------------------------------------------------------------------------------
@dataclass
class FallenPoses:
    joint_names: list[str]
    root_pos: np.ndarray        # (P, 3) at rest (only z is used)
    root_quat: np.ndarray       # (P, 4) w x y z
    joint_pos: np.ndarray       # (P, J)
    posture: np.ndarray         # (P,) index into POSTURES (how it ended up lying)
    drop: np.ndarray            # (P,) index into DROPS (how it was released)
    mode: np.ndarray            # (P,) index into MODES
    settle_time: np.ndarray     # (P,) s
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.root_pos)

    def counts(self) -> dict[str, int]:
        return {p: int((self.posture == i).sum()) for i, p in enumerate(POSTURES)}


def save_pool(path: str | Path, pool: FallenPoses) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        joint_names=np.array(pool.joint_names),
        root_pos=pool.root_pos.astype(np.float32),
        root_quat=pool.root_quat.astype(np.float32),
        joint_pos=pool.joint_pos.astype(np.float32),
        posture=pool.posture.astype(np.int8),
        drop=pool.drop.astype(np.int8),
        mode=pool.mode.astype(np.int8),
        settle_time=pool.settle_time.astype(np.float32),
        postures=np.array(POSTURES),
        meta=np.array(json.dumps(pool.meta)),
    )


def load_pool(path: str | Path) -> FallenPoses:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"no fallen-pose pool at {path}: make it once with make_fallen_poses.py "
            "(see the README)")
    z = np.load(path, allow_pickle=False)
    saved = tuple(str(s) for s in z["postures"])
    if saved != POSTURES:
        raise ValueError(f"{path}: posture list {saved} differs from this code's {POSTURES}; "
                         "make the pool again")
    return FallenPoses(
        joint_names=[str(s) for s in z["joint_names"]],
        root_pos=z["root_pos"], root_quat=z["root_quat"], joint_pos=z["joint_pos"],
        posture=z["posture"].astype(np.int64), drop=z["drop"].astype(np.int64),
        mode=z["mode"].astype(np.int64), settle_time=z["settle_time"],
        meta=json.loads(str(z["meta"])),
    )


def released_mask(pool: FallenPoses, names) -> np.ndarray:
    """Which poses were released in one of these ways (names from DROPS)."""
    names = [n.strip() for n in (names.split(",") if isinstance(names, str) else names)]
    bad = [n for n in names if n not in DROPS]
    if bad:
        raise ValueError(f"unknown release {bad} (names: {', '.join(DROPS)})")
    return np.isin(pool.drop, [DROPS.index(n) for n in names])


def subset(pool: FallenPoses, mask: np.ndarray, note: str = "") -> FallenPoses:
    """The poses where mask is true (e.g. released_mask), as a pool of its own."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError(f"no pose left in the pool ({note or 'empty selection'})")
    meta = dict(pool.meta)
    if note:
        meta["subset"] = note
    return FallenPoses(joint_names=list(pool.joint_names), root_pos=pool.root_pos[mask],
                       root_quat=pool.root_quat[mask], joint_pos=pool.joint_pos[mask],
                       posture=pool.posture[mask], drop=pool.drop[mask], mode=pool.mode[mask],
                       settle_time=pool.settle_time[mask], meta=meta)


def sample_weights(posture: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """Per-pose sampling probabilities so that each posture class gets its share of
    `weights` (classes missing from the pool or from `weights` get nothing)."""
    p = np.zeros(len(posture))
    for i, name in enumerate(POSTURES):
        w = float(weights.get(name, 0.0))
        sel = posture == i
        if w > 0 and sel.any():
            p[sel] = w / sel.sum()
    if p.sum() <= 0:
        raise ValueError(f"no pose of the classes {weights} in the pool")
    return p / p.sum()


# -- generation (CPU MuJoCo) -------------------------------------------------------------
@dataclass
class DropCfg:
    drop_weights: dict[str, float] = field(default_factory=lambda: {
        "supine": 0.25, "prone": 0.25, "left_side": 0.1, "right_side": 0.1, "random": 0.1,
        "standing": 0.2})
    mode_weights: dict[str, float] = field(default_factory=lambda: {
        "limp": 0.4, "hold": 0.3, "hold_then_limp": 0.3})
    tilt_deg: float = 25.0
    height_range: tuple[float, float] = (0.02, 0.25)   # lowest point above ground, m
    lin_vel: float = 0.5                               # +- m/s (z: only downwards)
    ang_vel: float = 1.5                               # +- rad/s per axis
    mix_range: tuple[float, float] = (0.2, 1.0)        # 0 = walking pose, 1 = random pose
    standing_mix: float = 0.3                          # "standing" drops: closer to walking
    standing_height: tuple[float, float] = (0.0, 0.01)  # feet just on the ground
    collapse_mix: float = 0.15                         # "collapse": about the walking pose
    collapse_vel: float = 0.1                          # +- m/s; x3 rad/s; always limp
    limp_after: tuple[float, float] = (0.2, 0.8)       # hold_then_limp: s
    min_time: float = 0.8
    max_time: float = 4.0
    still_vel: float = 0.25                            # max |qvel| (rad/s, m/s) at rest
    still_time: float = 0.3
    max_pelvis_z: float = 0.6                          # higher at rest = not fallen
    penetration: float = 0.01                          # deeper self-contact: redraw
    max_tries: int = 30


@dataclass
class RobotIndex:
    """Where the robot is in a compiled scene model (prefix 'robot/')."""
    joint_names: list[str]
    root_qadr: int
    root_vadr: int
    joint_ids: np.ndarray       # model joint ids of the hinges, in model order
    joint_qadr: np.ndarray
    joint_act: np.ndarray       # actuator id of each joint
    geom_ids: np.ndarray        # robot collision geoms
    pelvis_body: int


def index_robot(m, prefix: str = "robot/") -> RobotIndex:
    import mujoco
    name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""  # noqa: E731
    bodies = [b for b in range(m.nbody) if name(mujoco.mjtObj.mjOBJ_BODY, b).startswith(prefix)]
    if not bodies:
        raise ValueError(f"no body named {prefix}* in the model")
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and m.jnt_bodyid[j] in bodies]
    if len(free) != 1:
        raise ValueError(f"expected one free joint on the robot, found {len(free)}")
    hinges = [j for j in range(m.njnt) if m.jnt_bodyid[j] in bodies
              and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    act_of = {}
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            act_of[int(m.actuator_trnid[a, 0])] = a
        # The drops set ctrl = target angle: only right for <position> actuators (build the
        # scene with realistic=False; the DC motor model uses torque actuators).
        kp = m.actuator_gainprm[a, 0]
        if not (m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and kp > 0
                and abs(m.actuator_biasprm[a, 1] + kp) < 1e-6 * max(1.0, kp)):
            raise ValueError(f"actuator {name(mujoco.mjtObj.mjOBJ_ACTUATOR, a)} is not a "
                             "position actuator: build the scene with realistic=False")
    missing = [name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinges if j not in act_of]
    if missing:
        raise ValueError(f"joints without an actuator: {missing}")
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies
             and (m.geom_contype[g] or m.geom_conaffinity[g])]
    pelvis = [b for b in bodies if name(mujoco.mjtObj.mjOBJ_BODY, b) == prefix + "pelvis"]
    return RobotIndex(
        joint_names=[name(mujoco.mjtObj.mjOBJ_JOINT, j)[len(prefix):] for j in hinges],
        root_qadr=int(m.jnt_qposadr[free[0]]),
        root_vadr=int(m.jnt_dofadr[free[0]]),
        joint_ids=np.array(hinges),
        joint_qadr=np.array([m.jnt_qposadr[j] for j in hinges]),
        joint_act=np.array([act_of[j] for j in hinges]),
        geom_ids=np.array(geoms),
        pelvis_body=int(pelvis[0] if pelvis else bodies[0]),
    )


def _choice(rng, weights: dict[str, float], names: tuple[str, ...]) -> int:
    w = np.array([max(0.0, float(weights.get(n, 0.0))) for n in names])
    if w.sum() <= 0:
        raise ValueError(f"all weights are zero: {weights}")
    return int(rng.choice(len(names), p=w / w.sum()))


def _self_penetration(m, d, geom_set: set, tol: float) -> bool:
    n = d.ncon
    if n == 0:
        return False
    con = d.contact
    geom = np.asarray(con.geom).reshape(-1, 2)[:n] if hasattr(con, "geom") else \
        np.stack((np.asarray(con.geom1)[:n], np.asarray(con.geom2)[:n]), axis=1)
    dist = np.asarray(con.dist)[:n]
    for (g1, g2), dd in zip(geom, dist):
        if dd < -tol and int(g1) in geom_set and int(g2) in geom_set:
            return True
    return False


def _lowest(m, d, ix: RobotIndex) -> float:
    g = ix.geom_ids
    return lowest_point(m.geom_type[g], m.geom_size[g], d.geom_xpos[g], d.geom_xmat[g],
                        m.geom_rbound[g])


def drop_once(m, d, ix: RobotIndex, default_q: np.ndarray, lo: np.ndarray, hi: np.ndarray,
              cfg: DropCfg, rng: np.random.Generator):
    """One drop. Returns (qpos at rest, drop id, mode id, settle time) or None."""
    import mujoco
    drop = _choice(rng, cfg.drop_weights, DROPS)
    mode = _choice(rng, cfg.mode_weights, MODES)
    collapse = DROPS[drop] == "collapse"
    if collapse:
        mode = MODES.index("limp")
    geom_set = set(int(g) for g in ix.geom_ids)
    q0 = ix.root_qadr
    for _ in range(cfg.max_tries):
        mujoco.mj_resetData(m, d)
        mix = rng.uniform(*cfg.mix_range)
        if DROPS[drop] == "standing":
            mix *= cfg.standing_mix
        elif collapse:
            mix *= cfg.collapse_mix
        target = default_q + mix * (rng.uniform(lo, hi) - default_q)
        d.qpos[q0:q0 + 3] = (0.0, 0.0, 2.0)
        d.qpos[q0 + 3:q0 + 7] = drop_orientation(DROPS[drop], rng, cfg.tilt_deg)
        d.qpos[ix.joint_qadr] = target
        mujoco.mj_forward(m, d)
        if not _self_penetration(m, d, geom_set, cfg.penetration):
            break
    else:
        return None
    # Release height: lowest point of the robot at a random height above the ground.
    h = cfg.standing_height if DROPS[drop] in ("standing", "collapse") else cfg.height_range
    d.qpos[q0 + 2] += rng.uniform(*h) - _lowest(m, d, ix)
    v0 = ix.root_vadr
    lin, ang = (cfg.collapse_vel, 3.0 * cfg.collapse_vel) if collapse else (cfg.lin_vel, cfg.ang_vel)
    d.qvel[v0:v0 + 2] = rng.uniform(-lin, lin, size=2)
    d.qvel[v0 + 2] = rng.uniform(-lin, 0.0)
    d.qvel[v0 + 3:v0 + 6] = rng.uniform(-ang, ang, size=3)
    mujoco.mj_forward(m, d)

    dt = m.opt.timestep
    limp_at = 0.0 if MODES[mode] == "limp" else (
        rng.uniform(*cfg.limp_after) if MODES[mode] == "hold_then_limp" else math.inf)
    still = 0.0
    t = 0.0
    while t < cfg.max_time:
        if t >= limp_at:      # limp: target = where the joint is -> only the damping acts
            d.ctrl[ix.joint_act] = d.qpos[ix.joint_qadr]
        else:
            d.ctrl[ix.joint_act] = target
        mujoco.mj_step(m, d)
        t += dt
        if not np.all(np.isfinite(d.qpos)):
            return None
        still = still + dt if np.abs(d.qvel).max() < cfg.still_vel else 0.0
        if t >= cfg.min_time and still >= cfg.still_time:
            break
    else:
        return None
    mujoco.mj_kinematics(m, d)
    if d.xpos[ix.pelvis_body][2] > cfg.max_pelvis_z or _lowest(m, d, ix) < -0.02:
        return None
    return d.qpos.copy(), drop, mode, t


def generate(m, ix: RobotIndex, default_q: np.ndarray, n: int, cfg: DropCfg | None = None,
             seed: int = 0, log_every: float = 10.0, tag: str = "") -> FallenPoses:
    """n fallen poses (single process). default_q: the walking pose, in ix.joint_names order."""
    import mujoco
    cfg = cfg or DropCfg()
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    ranges = resolve_regex_values(JOINT_SAMPLE_RANGES, ix.joint_names)
    # Joint limits (shrunk by 5 %) intersected with the sampling ranges.
    jr = m.jnt_range[ix.joint_ids].astype(float)
    shrink = 0.05 * (jr[:, 1] - jr[:, 0])
    lo = np.maximum(np.array([r[0] for r in ranges]), jr[:, 0] + shrink)
    hi = np.minimum(np.array([r[1] for r in ranges]), jr[:, 1] - shrink)
    if np.any(lo >= hi):
        bad = [ix.joint_names[i] for i in np.nonzero(lo >= hi)[0]]
        raise ValueError(f"empty sampling range for {bad}")
    rows, drops, modes, times = [], [], [], []
    tries, t0, last = 0, time.time(), time.time()
    while len(rows) < n:
        tries += 1
        if tries > 20 * n + 100:
            raise RuntimeError(f"only {len(rows)} usable poses out of {tries} drops: check "
                               "DropCfg (still_vel, max_time, max_pelvis_z)")
        out = drop_once(m, d, ix, default_q, lo, hi, cfg, rng)
        if out is not None:
            rows.append(out[0])
            drops.append(out[1])
            modes.append(out[2])
            times.append(out[3])
        if log_every > 0 and time.time() - last > log_every:
            last = time.time()
            print(f"  {tag}{len(rows)}/{n} poses ({tries} drops, {time.time() - t0:.0f} s)",
                  flush=True)
    q = np.array(rows)
    q0 = ix.root_qadr
    quat = q[:, q0 + 3:q0 + 7]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return FallenPoses(
        joint_names=list(ix.joint_names),
        root_pos=q[:, q0:q0 + 3], root_quat=quat, joint_pos=q[:, ix.joint_qadr],
        posture=np.array([classify(qq) for qq in quat]), drop=np.array(drops),
        mode=np.array(modes), settle_time=np.array(times),
        meta={"seed": seed, "drops_tried": tries, "drop_cfg": asdict(cfg),
              "timestep": float(m.opt.timestep)},
    )


def _worker(job):
    import mujoco
    mjb, ix, default_q, n, cfg, seed, tag = job
    m = mujoco.MjModel.from_binary_path(mjb)
    return generate(m, ix, default_q, n, cfg, seed, log_every=20.0, tag=tag)


def merge(pools: list[FallenPoses]) -> FallenPoses:
    first = pools[0]
    for p in pools[1:]:
        if p.joint_names != first.joint_names:
            raise ValueError("pools with different joints")
    cat = lambda k: np.concatenate([getattr(p, k) for p in pools])  # noqa: E731
    meta = dict(first.meta)
    meta["drops_tried"] = int(sum(p.meta.get("drops_tried", 0) for p in pools))
    meta["seeds"] = [p.meta.get("seed") for p in pools]
    return FallenPoses(first.joint_names, cat("root_pos"), cat("root_quat"), cat("joint_pos"),
                       cat("posture"), cat("drop"), cat("mode"), cat("settle_time"), meta)


def generate_parallel(m, ix: RobotIndex, default_q: np.ndarray, n: int, workers: int,
                      cfg: DropCfg | None = None, seed: int = 0, tmp_dir: str | None = None
                      ) -> FallenPoses:
    """Same as generate, split over `workers` processes (each loads the model from an
    MJB file, so nothing GPU-related is inherited)."""
    if workers <= 1:
        return generate(m, ix, default_q, n, cfg, seed)
    import multiprocessing as mp
    import tempfile

    import mujoco
    tmp = Path(tempfile.mkdtemp(dir=tmp_dir))
    mjb = str(tmp / "getup_scene.mjb")
    mujoco.mj_saveModel(m, mjb, None)
    sizes = [n // workers + (1 if i < n % workers else 0) for i in range(workers)]
    jobs = [(mjb, ix, default_q, k, cfg or DropCfg(), seed * 1000 + i, f"[{i}] ")
            for i, k in enumerate(sizes) if k > 0]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = pool.map(_worker, jobs)
    return merge(parts)
