#!/usr/bin/env python3
"""Checks of the get-up code without mjlab, MuJoCo or torch:  python3 checks/getup_check.py

Runs the REAL getup/*.py code with a numpy stand-in for torch (checks/faketorch.py) and
small stubs for mjlab (same logic as mjlab 1.2.0 where it matters: name matching,
CommandTerm). Unitree's real constants file (gains, action scales) is used if the
unitree_rl_mjlab checkout is found (UNITREE_RL_MJLAB=<path>), else a copy of its numbers.

It checks the plumbing and the arithmetic: pose geometry (which way the robot faces for
each posture), the pool file, the reward values from lying to standing (they must grow),
the pace (standing early earns nothing more), the torque budget and the flight penalty,
the standing detection and the episode bookkeeping, the assist force, the curriculum,
the left-right mirror (with history), the reset event, the options of older runs. It
says nothing about how well a policy will learn.
"""
import dataclasses
import math
import os
import re
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import faketorch  # noqa: E402

sys.modules["torch"] = faketorch
import torch  # noqa: E402

OK = True


def check(cond, msg):
    global OK
    print(("  ok   " if cond else "  FAIL ") + msg)
    OK = OK and bool(cond)


# -- stubs -------------------------------------------------------------------------------
def module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


class _Anything(types.ModuleType):
    """Module whose every attribute is a named dummy function (mdp terms, dr functions)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        f = lambda *a, **k: None  # noqa: E731
        f.__name__ = name
        setattr(self, name, f)
        return f


def anything(name):
    m = _Anything(name)
    sys.modules[name] = m
    return m


class _Kw:
    """Config stand-in: keeps its keyword arguments as attributes."""

    def __init__(self, *args, **kw):
        self.args = args
        self.__dict__.update(kw)


def resolve_matching_names(keys, names, preserve_order=False):
    """Same as mjlab.utils.lab_api.string (IsaacLab): order of `names`, every key must match."""
    if isinstance(keys, str):
        keys = [keys]
    idx, out, used = [], [], [False] * len(keys)
    for i, n in enumerate(names):
        hit = None
        for k, key in enumerate(keys):
            if re.fullmatch(key, n):
                if hit is not None:
                    raise ValueError(f"multiple matches for {n}")
                hit = k
                used[k] = True
        if hit is not None:
            idx.append(i)
            out.append(n)
    if not all(used):
        raise ValueError(f"not all keys matched: {keys}")
    return idx, out


def resolve_matching_names_values(data, names, preserve_order=False):
    idx, out, vals = [], [], []
    for i, n in enumerate(names):
        for key, v in data.items():
            if re.fullmatch(key, n):
                idx.append(i)
                out.append(n)
                vals.append(v)
                break
    return idx, out, vals


def quat_from_euler_xyz(roll, pitch, yaw):
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    return torch.stack([cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp,
                        cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp], dim=-1)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = (q1[:, i] for i in range(4))
    w2, x2, y2, z2 = (q2[:, i] for i in range(4))
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=-1)


def sample_uniform(lo, hi, size, device=None):
    if isinstance(size, int):
        size = (size,)
    return torch.rand(*size) * (hi - lo) + lo


@dataclass
class SceneEntityCfg:
    name: str
    joint_names: object = None
    body_names: object = None
    geom_names: object = None
    joint_ids: object = field(default_factory=lambda: slice(None))
    body_ids: object = field(default_factory=lambda: slice(None))


@dataclass(kw_only=True)
class CommandTermCfg:
    resampling_time_range: tuple
    debug_vis: bool = False


class CommandTerm:
    """Same logic as mjlab 1.2.0's CommandTerm."""

    def __init__(self, cfg, env):
        self.cfg, self._env, self.metrics = cfg, env, {}
        self.time_left = torch.zeros(self.num_envs)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.long)

    num_envs = property(lambda self: self._env.num_envs)
    device = property(lambda self: self._env.device)

    def reset(self, env_ids):
        extras = {}
        for k, v in self.metrics.items():
            extras[k] = torch.mean(v[env_ids]).item()
            v[env_ids] = 0.0
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def compute(self, dt):
        self._update_metrics()
        self.time_left -= dt
        ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(ids) > 0:
            self._resample(ids)
        self._update_command()

    def _resample(self, env_ids):
        if len(env_ids) != 0:
            self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
            self._resample_command(env_ids)
            self.command_counter[env_ids] += 1


@dataclass
class ObservationTermCfg:
    func: object = None
    params: dict = field(default_factory=dict)
    noise: object = None
    clip: object = None
    scale: object = None
    history_length: int = 0


@dataclass(kw_only=True)
class ActuatorCfgStub:
    target_names_expr: tuple
    stiffness: float = 0.0
    damping: float = 0.0
    effort_limit: float | None = None
    armature: float = 0.0
    frictionloss: float = 0.0
    saturation_effort: float = 0.0
    velocity_limit: float = 0.0


@dataclass(frozen=True)
class ElectricActuator:
    reflected_inertia: float
    velocity_limit: float
    effort_limit: float


def reflected_inertia_from_two_stage_planetary(rotor_inertia, gear_ratio):
    r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
    r2 = rotor_inertia[1] * gear_ratio[2] ** 2
    return r1 + r2 + rotor_inertia[2]


class EntityCfg(_Kw):
    class InitialStateCfg(_Kw):
        pass


module("mujoco", MjSpec=SimpleNamespace(from_file=lambda *a: None),
       mjtGeom=SimpleNamespace(mjGEOM_SPHERE=2, mjGEOM_CAPSULE=3))
module("mjlab")
module("mjlab.actuator", BuiltinPositionActuatorCfg=ActuatorCfgStub,
       DcMotorActuatorCfg=ActuatorCfgStub)
module("mjlab.entity", EntityArticulationInfoCfg=_Kw, EntityCfg=EntityCfg)
module("mjlab.utils")
module("mjlab.utils.spec_config", CollisionCfg=_Kw)
module("mjlab.utils.actuator", ElectricActuator=ElectricActuator,
       reflected_inertia_from_two_stage_planetary=reflected_inertia_from_two_stage_planetary)
module("mjlab.utils.os", update_assets=lambda *a, **k: None, dump_yaml=lambda *a, **k: None)
module("mjlab.utils.lab_api")
module("mjlab.utils.lab_api.math", quat_from_euler_xyz=quat_from_euler_xyz, quat_mul=quat_mul,
       sample_uniform=sample_uniform)
module("mjlab.utils.lab_api.string", resolve_matching_names_values=resolve_matching_names_values,
       resolve_matching_names=resolve_matching_names)
module("mjlab.utils.noise", UniformNoiseCfg=_Kw)
module("mjlab.managers")
module("mjlab.managers.scene_entity_config", SceneEntityCfg=SceneEntityCfg)
module("mjlab.managers.command_manager", CommandTerm=CommandTerm, CommandTermCfg=CommandTermCfg)
module("mjlab.managers.event_manager", EventTermCfg=_Kw)
module("mjlab.managers.observation_manager", ObservationGroupCfg=_Kw,
       ObservationTermCfg=ObservationTermCfg)
module("mjlab.managers.reward_manager", RewardTermCfg=_Kw)
module("mjlab.managers.termination_manager", TerminationTermCfg=_Kw)
envs = module("mjlab.envs", ManagerBasedRlEnvCfg=_Kw)
mdp = anything("mjlab.envs.mdp")
mdp.dr = anything("mjlab.envs.mdp.dr")
envs.mdp = mdp
module("mjlab.envs.mdp.actions", JointPositionActionCfg=_Kw)
module("mjlab.scene", SceneCfg=_Kw)
module("mjlab.sensor", ContactMatch=_Kw, ContactSensorCfg=_Kw)
module("mjlab.sim", MujocoCfg=_Kw, SimulationCfg=_Kw)
module("mjlab.terrains", TerrainEntityCfg=_Kw)


class ViewerConfig(_Kw):
    class OriginType:
        ASSET_BODY = "asset_body"


module("mjlab.viewer", ViewerConfig=ViewerConfig)

# Unitree's constants: the real file if the checkout is there.
URL = Path(os.environ.get("UNITREE_RL_MJLAB", ROOT.parent / "ext" / "unitree_rl_mjlab"))
CONST = URL / "src" / "assets" / "robots" / "unitree_g1" / "g1_23dof_constants.py"
XML = URL / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "g1_23dof.xml"
if not CONST.exists():
    print(f"unitree_rl_mjlab not found at {URL} (set UNITREE_RL_MJLAB): cannot run")
    sys.exit(1)
module("src", SRC_PATH=URL / "src")
module("src.assets")
module("src.assets.robots")
unitree_g1 = module("src.assets.robots.unitree_g1")
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("src.assets.robots.unitree_g1.g1_23dof_constants", CONST)
g1c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g1c
spec.loader.exec_module(g1c)
unitree_g1.g1_23dof_constants = g1c

sys.path.insert(0, str(ROOT))
from getup import getup_env_cfg as gec  # noqa: E402
from getup import getup_mdp as gm  # noqa: E402
from getup import poses as ps  # noqa: E402
from getup import robot as rb  # noqa: E402
from getup import run_options as ro  # noqa: E402
from getup import symmetry as sym  # noqa: E402

# -- the robot as in g1_23dof.xml ---------------------------------------------------------
xml = XML.read_text()
BODIES = re.findall(r'<body name="([^"]+)"', xml)
JOINTS = [j for j in re.findall(r'<joint name="([^"]+)"', xml)]
GEOMS = re.findall(r'<geom name="([^"]+)"', xml)
assert len(JOINTS) == 23, JOINTS
DEFAULT = np.array(ps.resolve_regex_values(g1c.HOME_KEYFRAME.joint_pos, JOINTS, default=0.0),
                   dtype=np.float32)


# -- 1. robot config -----------------------------------------------------------------------
print("robot config")
sc = rb.action_scale(0.5)
knee = sc[".*_knee_joint"]
check(abs(knee - 0.5 * 139.0 / g1c.STIFFNESS_7520_22) < 1e-9,
      f"knee action scale 0.5 x 139 Nm / {g1c.STIFFNESS_7520_22:.1f} = {knee:.3f} rad")
walk = g1c.G1_23DOF_ACTION_SCALE
check(all(abs(sc[k] - 2 * walk[k]) < 1e-9 for k in walk), "action scale = 2 x walking's (0.25)")
check(len(sc) == 12 and set(sc) == set(walk), "same joint groups as Unitree's action scale")
check(set(rb.CONTACT_BODIES) <= set(BODIES) | set(rb.CONTACT_BODIES), "contact body patterns")
_, cnames = resolve_matching_names(rb.CONTACT_BODIES, BODIES)
check(len(cnames) == 16 and all(b in cnames for b in rb.FOOT_BODIES),
      f"ground contact sensor watches {len(cnames)} bodies incl. both feet")
coll_bodies = set()
cur = None
for line in xml.splitlines():
    m = re.search(r'<body name="([^"]+)"', line)
    if m:
        cur = m.group(1)
    if "<geom" in line and ('class="collision"' in line or 'class="foot_capsule"' in line):
        coll_bodies.add(cur)
diff = sorted(str(b) for b in coll_bodies ^ set(cnames))
check(not diff, "the contact sensor watches exactly the bodies that have collision geoms"
      + (f" (differ: {diff})" if diff else ""))
check(rb.HEAD_GEOM in GEOMS and all(h in GEOMS for h in rb.HAND_GEOMS), "head/hand geoms exist")

# the head collision geom with the size of the real head (v5)
def _fake_spec():
    geoms = [SimpleNamespace(name=n, type=2, size=[0.06, 0.0, 0.0], pos=[0.0, 0.0, 0.43],
                             quat=[1.0, 0.0, 0.0, 0.0]) for n in ("torso_collision", rb.HEAD_GEOM)]
    return SimpleNamespace(geoms=geoms)
spec_h = rb.real_head_spec_fn(_fake_spec)()
head_g = [g for g in spec_h.geoms if g.name == rb.HEAD_GEOM][0]
other_g = [g for g in spec_h.geoms if g.name == "torso_collision"][0]
check(head_g.type == 3 and list(head_g.size)[:2] == [0.07, 0.025]
      and list(head_g.pos) == [0.01, 0.0, 0.405] and other_g.type == 2 and other_g.pos == [0.0, 0.0, 0.43],
      "head collision: capsule r 7 cm, axis z 38-43 cm, same name; other geoms untouched")
try:
    rb.real_head_spec_fn(lambda: SimpleNamespace(geoms=[]))()
    check(False, "no head geom -> error")
except ValueError:
    check(True, "no head geom -> error")
cfg_h, cfg_s = rb.get_robot_cfg(realistic=False), rb.get_robot_cfg(realistic=False, real_head=False)
check(cfg_h.spec_fn is not cfg_s.spec_fn and cfg_s.spec_fn is g1c.get_spec,
      "get_robot_cfg: real head by default, Unitree's sphere with real_head=False")
# does the capsule cover the real head mesh? (numpy STL reader; skipped without the file)
_stl = XML.parent / "assets" / "head_link.STL"
if _stl.is_file():
    import struct
    _b = _stl.read_bytes()
    _n = struct.unpack("<I", _b[80:84])[0]
    _v = np.frombuffer(_b[84:84 + 50 * _n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                            ("a", "<u2")]))["v"].reshape(-1, 3)
    _v = _v.astype(float) + np.array([0.0039635, 0.0, -0.044])     # geom pos in g1_23dof.xml
    _v = _v[_v[:, 2] >= 0.32]                                       # above the head's base
    _c, _h, _r = np.array(rb.HEAD_CAPSULE["pos"]), rb.HEAD_CAPSULE["half_length"], rb.HEAD_CAPSULE["radius"]
    _a, _e = _c - [0, 0, _h], _c + [0, 0, _h]
    _t = np.clip((_v - _a) @ (_e - _a) / ((_e - _a) @ (_e - _a)), 0, 1)
    _out = np.linalg.norm(_v - (_a + _t[:, None] * (_e - _a)), axis=1) - _r
    _sph = np.linalg.norm(_v - [0, 0, 0.43], axis=1) - 0.06
    check(_out.max() < 0.01 and _sph.max() > 0.04,
          f"real head mesh: outside the capsule by at most {100 * _out.max():.1f} cm "
          f"(Unitree's sphere: {100 * _sph.max():.1f} cm)")
else:
    print(f"  skip head mesh check ({_stl} not found)")

# -- 2. poses ---------------------------------------------------------------------------------
print("fallen poses (geometry)")
rng = np.random.default_rng(1)
axes_ok = {}
for drop in ("supine", "prone", "left_side", "right_side"):
    good = 0
    for _ in range(200):
        r = ps.quat_to_mat(ps.drop_orientation(drop, rng, tilt_deg=0.0))
        f = r[2]          # world z of body x, y, z
        want = {"supine": f[0] > 0.99, "prone": f[0] < -0.99,
                "left_side": f[1] < -0.99, "right_side": f[1] > 0.99}[drop]
        good += bool(want)
    axes_ok[drop] = good == 200
check(all(axes_ok.values()), "release orientations: supine = chest up, prone = chest down, "
      "left_side = left side down, right_side = right down")
cls_ok = all(ps.POSTURES[ps.classify(ps.drop_orientation(d, rng, 25.0))] == d
             for d in ("supine", "prone", "left_side", "right_side") for _ in range(100))
check(cls_ok, "classify() gives back the release posture (+-25 deg tilt)")
check(ps.POSTURES[ps.classify(np.array([1.0, 0, 0, 0]))] == "upright", "identity = upright")
st_ok = all(ps.quat_to_mat(ps.drop_orientation("standing", rng, 25.0))[2, 2] > math.cos(math.radians(14.2))
            for _ in range(200))
check(st_ok, "standing release: upright within 10 deg per axis")
co_ok = all(ps.quat_to_mat(ps.drop_orientation("collapse", rng, 25.0))[2, 2] > math.cos(math.radians(7.1))
            for _ in range(200))
check(co_ok and ps.DROPS[-1] == "collapse" and ps.DropCfg().drop_weights.get("collapse", 0.0) == 0.0
      and ps.DropCfg().collapse_mix < ps.DropCfg().standing_mix,
      "collapse release: upright within 5 deg per axis, closer to the walking pose than "
      "'standing'; last in DROPS (older pools keep their indices), not in the default mix")
q = ps.quat_from_rpy(0.3, -0.4, 1.2)
r = ps.quat_to_mat(q)
check(abs(np.linalg.det(r) - 1) < 1e-9 and np.allclose(r @ r.T, np.eye(3)), "rotation matrix")
# quat_from_rpy matches the torch version used at reset (same convention as mjlab)
qt = quat_from_euler_xyz(torch.tensor([0.3]), torch.tensor([-0.4]), torch.tensor([1.2])).numpy()[0]
check(np.allclose(q, qt, atol=1e-6), "numpy quat_from_rpy == mjlab quat_from_euler_xyz")
# lowest point: vertical capsule (half-length 0.2, r 0.05) at z = 1 -> 0.75; lying -> 0.95
eye = np.eye(3).reshape(9)
ry = ps.quat_to_mat(ps.quat_from_rpy(0, math.pi / 2, 0)).reshape(9)
lp_v = ps.lowest_point([3], [[0.05, 0.2, 0]], [[0, 0, 1.0]], [eye], [0.25])
lp_h = ps.lowest_point([3], [[0.05, 0.2, 0]], [[0, 0, 1.0]], [ry], [0.25])
lp_s = ps.lowest_point([2, 6], [[0.1, 0, 0], [0.1, 0.2, 0.3]], [[0, 0, 1.0], [0, 0, 2.0]],
                       [eye, eye], [0.1, 0.4])
check(abs(lp_v - 0.75) < 1e-9 and abs(lp_h - 0.95) < 1e-9 and abs(lp_s - 0.9) < 1e-9,
      f"lowest point: capsule upright {lp_v:.2f}, lying {lp_h:.2f}; sphere+box {lp_s:.2f}")
post = np.array([0, 0, 1, 1, 1, 2, 5])
w = ps.sample_weights(post, {"supine": 1, "prone": 1, "left_side": 2})
check(abs(w[post == 0].sum() - 0.25) < 1e-9 and abs(w[post == 2].sum() - 0.5) < 1e-9
      and w[post == 5].sum() == 0, "sample weights: share per class, not per pose")
ranges = ps.resolve_regex_values(ps.JOINT_SAMPLE_RANGES, JOINTS)
jr = {n: tuple(map(float, re.search(rf'<joint name="{n}"[^>]*range="([^"]+)"', xml).group(1).split()))
      for n in JOINTS}
inside = all(jr[n][0] < r[0] < r[1] < jr[n][1] or (n.startswith(("left_hip_roll", "right_hip_roll")))
             for n, r in zip(JOINTS, ranges))
check(inside and len(ranges) == 23, "every joint has a sampling range inside its limits")
check(all(lo <= d <= hi for d, (lo, hi) in zip(DEFAULT, ranges)),
      "the walking pose is inside the sampling ranges")
# pool file round trip, joints stored in a shuffled order
n_pool = 40
perm = rng.permutation(23)
quats = np.array([ps.drop_orientation(["supine", "prone", "left_side", "right_side"][i % 4], rng)
                  for i in range(n_pool)])
pool = ps.FallenPoses(
    joint_names=[JOINTS[i] for i in perm],
    root_pos=np.column_stack([rng.normal(size=(n_pool, 2)), rng.uniform(0.1, 0.3, n_pool)]),
    root_quat=quats, joint_pos=rng.normal(size=(n_pool, 23)),
    posture=np.array([ps.classify(q) for q in quats]), drop=np.zeros(n_pool, int),
    mode=np.zeros(n_pool, int), settle_time=np.ones(n_pool), meta={"test": 1, "head": rb.HEAD_SHAPE})
tmp = Path(tempfile.mkdtemp()) / "pool.npz"
ps.save_pool(tmp, pool)
back = ps.load_pool(tmp)
check(back.joint_names == pool.joint_names and np.allclose(back.joint_pos, pool.joint_pos, atol=1e-6)
      and np.array_equal(back.posture, pool.posture) and back.meta == {"test": 1, "head": rb.HEAD_SHAPE},
      f"pool save/load ({back.counts()})")
# only the poses released a given way (eval/sim2sim --released)
pool_r = dataclasses.replace(pool, drop=np.array([ps.DROPS.index("collapse") if i % 5 == 0
                                                  else ps.DROPS.index("supine")
                                                  for i in range(n_pool)]))
m_c = ps.released_mask(pool_r, "collapse")
sub = ps.subset(pool_r, m_c, note="released as collapse")
tmp2 = Path(tempfile.mkdtemp()) / "sub.npz"
ps.save_pool(tmp2, sub)
back2 = ps.load_pool(tmp2)
check(int(m_c.sum()) == 8 and len(back2) == 8 and np.all(back2.drop == ps.DROPS.index("collapse"))
      and np.allclose(back2.joint_pos, pool_r.joint_pos[m_c], atol=1e-6)
      and back2.meta.get("subset") == "released as collapse"
      and int(ps.released_mask(pool_r, "collapse,supine").sum()) == n_pool,
      "pool subset by release (8 of 40 collapse), saved and loaded")
try:
    ps.released_mask(pool_r, "kneel")
    check(False, "unknown release name refused")
except ValueError:
    check(True, "unknown release name refused")
try:
    ps.subset(pool_r, ps.released_mask(pool_r, "random"), note="released as random")
    check(False, "empty subset refused")
except ValueError:
    check(True, "empty subset refused")

# -- 3. env config -----------------------------------------------------------------------------
print("env config")
cfg = gec.make_getup_env_cfg(pool_path=tmp)
rew = cfg.rewards
g = cfg.commands["getup"]
check(set(g.reg_terms) == set(gec.REG_TERMS) and all(g.reg_terms[k] == rew[k].weight for k in g.reg_terms),
      "regularization terms known to the curriculum with their full weights")
check(all(rew[k].weight < 0 for k in gec.REG_TERMS), "regularization weights are penalties")
actor = cfg.observations["actor"]
check(list(actor.terms) == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"]
      and actor.history_length == 5 and actor.enable_corruption,
      "policy: 5 deployable signals x 5 steps of history, with noise")
crit = cfg.observations["critic"]
check(all(crit.terms[k].noise is None for k in crit.terms) and "assist" in crit.terms,
      f"critic: {len(crit.terms)} terms, no noise, sees the assist force")
check(set(crit.terms) <= set(sym._SIGNS) | set(sym._PAIRS)
      | {"joint_pos", "joint_vel", "actions", "body_contact"}, "the mirror knows every term")
check(cfg.events["reset_pose"].mode == "reset" and cfg.events["friction"].mode == "startup",
      "events: reset from the pool, randomization at startup")
check(cfg.sim.njmax >= 400 and cfg.sim.nconmax >= 96, "room for the contacts of a lying robot")
pc = gec.make_getup_env_cfg(play=True, pool_path=tmp)
check(not pc.commands["getup"].curriculum and pc.commands["getup"].assist_force == 0.0
      and not pc.observations["actor"].enable_corruption, "play: no curriculum, no assist, no noise")
paced = ("head_up", "pelvis_height", "stand_still", "stand_pose", "feet_support")
check(all(rew[k].params["pace_s"] == gec.PACE_S == 3.0 for k in paced)
      and crit.terms["pace"].params["pace_s"] == 3.0 and "pace" not in actor.terms,
      "pace 3 s: task and standing rewards paced, the critic sees the phase, the policy not")
check(rew["torque_budget"].params["budget"] == 0.8 and "torque_budget" in g.reg_terms
      and rew["joint_overspeed"].params["ratio"] == 0.6 and rew["airborne"].weight < 0,
      "torque budget 80 % and speed budget 60 % (both in the curriculum), flight penalized")
lc = gec.make_getup_env_cfg(pool_path=tmp, pace_s=0.0, torque_budget=0.0, speed_budget=0.9)
check("torque_budget" not in lc.rewards and "torque_budget" not in lc.commands["getup"].reg_terms
      and lc.rewards["head_up"].params["pace_s"] == 0.0
      and lc.rewards["joint_overspeed"].params["ratio"] == 0.9
      and lc.observations["critic"].terms["pace"].params["pace_s"] == 0.0,
      "pace 0, torque budget 0, speed 0.9: the first version's task (no torque term)")
check(ro.pace_options({"action_scale": 0.5}) == {"pace_s": 0.0, "torque_budget": 0.0, "speed_budget": 0.9}
      and ro.pace_options({"pace_s": 3.0, "torque_budget": 0.8, "speed_budget": 0.6})
      == {"pace_s": 3.0, "torque_budget": 0.8, "speed_budget": 0.6}
      and "before" in ro.describe_pace(ro.pace_options({})),
      "run options: runs from before the pace are rebuilt without it")
check("feet_above_pelvis" in rew and rew["feet_above_pelvis"].weight == -gec.FEET_HIGH == -4.0
      and rew["feet_above_pelvis"].params["margin"] == 0.05
      and rew["head_ground"].weight == -4.0 and rew["head_impact"].weight == -8.0
      and all(cfg.events[k].mode == "startup" for k in ("joint_friction", "joint_damping", "joint_armature"))
      and cfg.events["joint_friction"].params["ranges"] == (0.0, 0.4),
      "v5 style: feet above the pelvis cost 4/m, head costs x2, joint friction 0-0.4 N m, "
      "damping, armature randomized at startup")
oc = gec.make_getup_env_cfg(pool_path=tmp, **ro.style_options({"action_scale": 0.5, "pace_s": 3.0}))
check("feet_above_pelvis" not in oc.rewards and oc.rewards["head_impact"].weight == -4.0
      and oc.rewards["head_ground"].weight == -2.0 and "joint_friction" not in oc.events,
      "runs from before v5 (no feet_high in their options) are rebuilt without the v5 style")
nr = gec.make_getup_env_cfg(pool_path=tmp, randomize=False)
check("joint_friction" not in nr.events and "friction" not in nr.events,
      "randomize=False: no joint randomization either")
new_style = {"feet_high": 4.0, "head_scale": 2.0, "joint_dr": True}
check(ro.style_options(dict(new_style, pace_s=3.0)) == new_style
      and ro.style_options({}) == ro.LEGACY_STYLE and "fixed" in ro.describe_style(ro.LEGACY_STYLE),
      "run options: v5 style saved and read back; older runs get the legacy style")

# -- fake env -------------------------------------------------------------------------------------
N = 8
HEAD_LOCAL = GEOMS.index(rb.HEAD_GEOM)
GLOBAL_GEOM_OFFSET = 3          # pretend the terrain owns the first geoms


class FakeSensor:
    def __init__(self, p, history=0):
        self.data = SimpleNamespace(found=torch.zeros(N, p), force=torch.zeros(N, p, 3),
                                    force_history=torch.zeros(N, p, history, 3) if history else None)


class FakeRobot:
    def __init__(self):
        self.joint_names = tuple(JOINTS)
        self.body_names = tuple(BODIES)
        self.geom_names = tuple(GEOMS)
        self.actuator_names = tuple(JOINTS)
        nb, ng = len(BODIES), len(GEOMS) + GLOBAL_GEOM_OFFSET
        self.data = SimpleNamespace(
            root_link_pos_w=torch.zeros(N, 3), projected_gravity_b=torch.zeros(N, 3),
            root_link_lin_vel_w=torch.zeros(N, 3), root_link_ang_vel_w=torch.zeros(N, 3),
            joint_pos=torch.tensor(np.tile(DEFAULT, (N, 1))),
            default_joint_pos=torch.tensor(np.tile(DEFAULT, (N, 1))),
            joint_pos_target=torch.tensor(np.tile(DEFAULT, (N, 1))),
            joint_vel=torch.zeros(N, 23), body_link_pos_w=torch.zeros(N, nb, 3),
            body_link_lin_vel_w=torch.zeros(N, nb, 3), root_link_vel_w=torch.zeros(N, 6),
            data=SimpleNamespace(geom_xpos=torch.zeros(N, ng, 3)),
            indexing=SimpleNamespace(geom_ids=torch.tensor(
                np.arange(len(GEOMS)) + GLOBAL_GEOM_OFFSET)))
        self.wrench = None
        self.writes = {}

    def find_bodies(self, keys, preserve_order=False):
        return resolve_matching_names(keys, self.body_names, preserve_order)

    def find_geoms(self, keys, preserve_order=False):
        return resolve_matching_names(keys, self.geom_names, preserve_order)

    def find_joints(self, keys, preserve_order=False):
        return resolve_matching_names(keys, self.joint_names, preserve_order)

    def write_external_wrench_to_sim(self, force, torque, env_ids=None, body_ids=None):
        self.wrench = (force.clone(), list(body_ids))

    def write_root_link_pose_to_sim(self, pose, env_ids=None):
        self.writes["pose"] = (pose.clone(), env_ids.clone())

    def write_root_link_velocity_to_sim(self, vel, env_ids=None):
        self.writes["vel"] = vel.clone()

    def write_joint_state_to_sim(self, pos, vel, env_ids=None, joint_ids=None):
        self.writes["joints"] = pos.clone()


class FakeRewardManager:
    def __init__(self, rewards):
        self.cfgs = {k: SimpleNamespace(weight=v.weight) for k, v in rewards.items()}

    def get_term_cfg(self, name):
        return self.cfgs[name]


class FakeScene:
    def __init__(self, items):
        self.items = items
        self.env_origins = torch.zeros(N, 3)

    def __getitem__(self, k):
        return self.items[k]


class FakeEnv:
    def __init__(self):
        self.num_envs, self.device, self.step_dt = N, "cpu", 0.02
        self.max_episode_length_s = 10.0
        self.episode_length_buf = torch.zeros(N, dtype=torch.long)
        robot = FakeRobot()
        self._scene = {"robot": robot, "body_ground": FakeSensor(len(cnames)),
                       "head_ground": FakeSensor(1, history=4)}
        self.scene = FakeScene(self._scene)
        self.reward_manager = FakeRewardManager(rew)


env = FakeEnv()
robot = env._scene["robot"]
contact = env._scene["body_ground"]
head_s = env._scene["head_ground"]
ids = gm.robot_ids(env)
check(ids.head == HEAD_LOCAL + GLOBAL_GEOM_OFFSET, "head geom: global id through the entity indexing")
check([ids.contact_names[i] for i in ids.feet_cols] == list(rb.FOOT_BODIES),
      "foot columns of the contact sensor")


def set_state(i, pelvis, head, tilt_deg, contacts=(), feet=(True, True), head_force=0.0):
    t = math.radians(tilt_deg)
    robot.data.root_link_pos_w[i] = torch.tensor([0.0, 0.0, pelvis])
    robot.data.projected_gravity_b[i] = torch.tensor([math.sin(t), 0.0, -math.cos(t)])
    robot.data.data.geom_xpos[i, ids.head] = torch.tensor([0.0, 0.0, head])
    f = np.zeros(len(cnames), dtype=np.float32)
    for b in contacts:
        f[cnames.index(b)] = 1.0
    for k, on in enumerate(feet):
        f[ids.feet_cols[k]] = 1.0 if on else 0.0
    contact.data.found[i] = torch.tensor(f)
    head_s.data.found[i] = torch.tensor([1.0 if head_force > 0 else 0.0])
    head_s.data.force_history[i] = torch.tensor(np.tile([0.0, 0.0, head_force], (1, 4, 1)))


# -- 4. rewards from lying to standing -------------------------------------------------------------
print("rewards (per second, weights and params of make_rewards, 4 s into the episode)")
env.episode_length_buf = torch.full((N,), 200, dtype=torch.long)     # 4 s: past the 3 s pace
scen = [  # name, pelvis z, head z, tilt deg, other contacts, feet, head force
    ("lying on the back", 0.12, 0.13, 90, ("pelvis", "torso_link"), (True, True), 0.0),
    ("lying, head down", 0.12, 0.06, 90, ("pelvis", "torso_link"), (True, True), 80.0),
    ("sitting up", 0.10, 0.75, 10, ("pelvis",), (True, True), 0.0),
    ("on hands and knees", 0.42, 0.55, 80, ("left_knee_link", "right_knee_link",
                                          "left_wrist_roll_rubber_hand"), (True, True), 0.0),
    ("kneeling upright", 0.45, 0.95, 5, ("left_knee_link", "right_knee_link"), (True, True), 0.0),
    ("squatting", 0.45, 0.93, 15, (), (True, True), 0.0),
    ("standing", 0.76, 1.23, 2, (), (True, True), 0.0),
]
for i, sc_ in enumerate(scen):
    set_state(i, *sc_[1:])
dev_term = gm.joint_deviation_beyond(SimpleNamespace(params=rew["joint_deviation"].params), env)


def reward_values():
    """The reward terms that depend on the state (not on the actions), with their params."""
    p = {k: getattr(rew[k], "params", {}) for k in paced}
    return {
        "head_up": gm.head_up(env, **p["head_up"]),
        "pelvis_height": gm.pelvis_height(env, **p["pelvis_height"]),
        "stand_still": gm.stand_still(env, **p["stand_still"]),
        "stand_pose": gm.stand_pose(env, **p["stand_pose"]),
        "feet_support": gm.feet_support(env, **p["feet_support"]),
        "head_ground": gm.head_ground_contact(env), "head_impact": gm.head_impact(env),
        "joint_deviation": dev_term(env, None), "airborne": gm.airborne(env),
    }


funcs = reward_values()
tot = sum(funcs[k] * rew[k].weight for k in funcs)
st = gm.getup_state(env)
for i, sc_ in enumerate(scen):
    parts = ", ".join(f"{k} {float(funcs[k][i] * rew[k].weight):+.2f}" for k in funcs
                      if abs(float(funcs[k][i])) > 1e-4)
    print(f"    {sc_[0]:20s} total {float(tot[i]):+5.2f}/s  standing={bool(st.standing[i])}  ({parts})")
order = [float(tot[i]) for i in (0, 2, 4, 6)]
check(order == sorted(order), "lying < sitting < kneeling < standing")
check(float(tot[6]) > 1.8 * float(tot[5]), "standing pays > 1.8 x squatting (post-task rewards)")
check(float(tot[1]) < float(tot[0]), "head on the ground with an impact costs")
check([bool(x) for x in st.standing][:7] == [False] * 6 + [True], "only the last state is standing")
rew_tol = gm.tolerance(torch.tensor([1.20, 1.10, 0.95, 0.30]), 1.10, margin=0.8)
check(np.allclose(rew_tol.numpy(), [1.0, 1.0, math.exp(-0.5 * (0.15 / 0.8 * math.sqrt(-2 * math.log(0.1))) ** 2), 0.1],
                  atol=1e-5), f"tolerance: 1 at/above the target, {float(rew_tol[2]):.2f} 15 cm below, "
      "0.1 at the margin")
robot.data.joint_pos[6, JOINTS.index("waist_yaw_joint")] = 1.4
check(abs(float(dev_term(env, None)[6]) - 0.5) < 1e-5, "joint deviation beyond 0.9 rad (waist 1.4 -> 0.5)")
robot.data.joint_pos[6] = torch.tensor(DEFAULT)

# -- 4b. pace: standing early earns nothing more ----------------------------------------------------
print("pace (3 s)")
env.episode_length_buf = torch.tensor([1, 75, 150, 300, 0, 0, 0, 0], dtype=torch.long)
ph = gm.pace_phase(env, 3.0).numpy()
check(np.allclose(ph[:4], [0.02 / 3, 0.5, 1.0, 1.0], atol=1e-6)
      and np.allclose(gm.pace_phase(env, 0.0).numpy(), 1.0),
      "phase: 0 -> 1 over 3 s, then 1; no pace: always 1")


def totals_at(step):
    env.episode_length_buf = torch.full((N,), step, dtype=torch.long)
    f = reward_values()
    return sum(f[k] * rew[k].weight for k in f), f


t_half, f_half = totals_at(75)            # 1.5 s
t_end, _ = totals_at(160)                 # 3.2 s
t_start, _ = totals_at(1)                 # first step
print(f"    at 1.5 s: lying {float(t_half[0]):+.2f}/s, sitting {float(t_half[2]):+.2f}, kneeling "
      f"{float(t_half[4]):+.2f}, squatting {float(t_half[5]):+.2f}, standing {float(t_half[6]):+.2f}"
      f"   (after 3 s: standing {float(t_end[6]):+.2f})")
check(abs(float(t_half[6]) - float(t_half[4])) < 1e-4 and abs(float(t_half[6]) - 4.5) < 1e-4,
      "at 1.5 s standing pays the same as kneeling upright (4.5/s: both ahead of the schedule)")
check(float(t_half[0]) < 0.5 * float(t_half[4]),
      "at 1.5 s still lying pays less than half: falling behind the schedule costs")
check(float(t_start[0]) > 0.95 * 4.5 and float(t_start[6]) - float(t_start[0]) < 0.05,
      "at the start lying is on the schedule (almost the full task reward)")
check(abs(float(t_end[6]) - float(tot[6])) < 1e-4 and float(f_half["stand_pose"][6]) == 0.0,
      "the standing rewards count from 3 s on (before: 0)")
env.episode_length_buf = torch.full((N,), 200, dtype=torch.long)
unpaced = sum(f(env) * rew[k].weight for k, f in (
    ("head_up", gm.head_up), ("pelvis_height", gm.pelvis_height), ("stand_still", gm.stand_still),
    ("stand_pose", gm.stand_pose), ("feet_support", gm.feet_support)))
paced_now = reward_values()
paced_sum = sum(paced_now[k] * rew[k].weight for k in paced)
check(np.allclose(unpaced.numpy(), paced_sum.numpy(), atol=1e-5),
      "after the pace the rewards are exactly the unpaced ones (the table above)")

# -- 4c. torque budget and flight -------------------------------------------------------------------
print("torque budget, flight")
env.episode_length_buf = torch.full((N,), 200, dtype=torch.long)
tb = gm.torque_over_budget(SimpleNamespace(params=rew["torque_budget"].params), env)
robot.data.joint_pos_target = robot.data.joint_pos.clone()
check(np.allclose(tb(env, **rew["torque_budget"].params).numpy(), 0.0),
      "targets at the joints, at rest: no cost")
el, kn = JOINTS.index("left_elbow_joint"), JOINTS.index("right_knee_joint")
robot.data.joint_pos_target[0, el] = robot.data.joint_pos[0, el] + 2.0
ratio = 2.0 * g1c.STIFFNESS_5020 / 25.0
check(abs(float(tb(env, **rew["torque_budget"].params)[0]) - (ratio - 0.8) ** 2) < 1e-5,
      f"elbow asked 2 rad away: {2.0 * g1c.STIFFNESS_5020:.1f} N m = {ratio:.2f} x peak -> "
      f"({ratio:.2f} - 0.8)^2 = {(ratio - 0.8) ** 2:.3f}, x weight {rew['torque_budget'].weight} "
      f"= {(ratio - 0.8) ** 2 * rew['torque_budget'].weight:+.2f}/s")
robot.data.joint_pos_target[0, el] = robot.data.joint_pos[0, el]
robot.data.joint_vel[1, kn] = 16.0
kr = float(tb.ratio(env)[1, kn])
check(abs(kr - 16.0 * g1c.DAMPING_7520_22 / 139.0) < 1e-5 and float(tb(env, **rew["torque_budget"].params)[1]) == 0.0,
      f"knee at 16 rad/s, target = position: damping alone asks {100 * kr:.0f} % of the peak (below 80 %)")
robot.data.joint_vel[1, kn] = 0.0
check(abs(rew["torque_budget"].weight * 0.2 ** 2 + 0.2) < 1e-9
      and abs(rew["torque_budget"].weight * 1.2 ** 2 + 7.2) < 1e-9,
      "a motor at its peak costs 0.2/s, one asked twice its peak 7.2/s")
set_state(7, 0.6, 1.0, 5, contacts=(), feet=(False, False))
air = gm.airborne(env).numpy()
env.episode_length_buf = torch.full((N,), 3, dtype=torch.long)
air_early = gm.airborne(env).numpy()
check(air[7] == 1.0 and np.all(air[:7] == 0.0) and air_early[7] == 0.0,
      "flight: 1 with nothing on the ground (not in the first 0.1 s); the 7 states above: 0")
env.episode_length_buf = torch.zeros(N, dtype=torch.long)

# -- 4d. feet slip and pushes -------------------------------------------------------------------
print("feet slip, pushes")
check("feet_slip" in g.reg_terms and rew["feet_slip"].weight == -1.0,
      "feet slip penalized (-1 per (m/s)^2), in the curriculum")
fb = [BODIES.index(b) for b in rb.FOOT_BODIES]
slip_cfg = SimpleNamespace(name="robot", body_ids=fb)
robot.data.body_link_lin_vel_w[6, fb[0]] = torch.tensor([0.3, 0.4, 0.0])   # standing, left foot slides
robot.data.body_link_lin_vel_w[7, fb[1]] = torch.tensor([1.0, 0.0, 0.0])   # env 7 in the air
sl = gm.feet_slip(env, slip_cfg).numpy()
check(abs(sl[6] - 0.25) < 1e-6 and sl[7] == 0.0 and np.all(sl[:6] == 0.0),
      f"slip: left foot on the ground at 0.5 m/s -> {sl[6]:.2f}; a foot in the air does not count")
robot.data.body_link_lin_vel_w = torch.zeros(N, len(BODIES), 3)
saved_pos = robot.data.body_link_pos_w.clone()
saved_root = robot.data.root_link_pos_w.clone()
robot.data.root_link_pos_w = torch.zeros(N, 3)
robot.data.root_link_pos_w[:, 2] = 0.15                                  # pelvis 0.15 m
robot.data.body_link_pos_w = torch.zeros(N, len(BODIES), 3)
robot.data.body_link_pos_w[:, fb[0], 2] = 0.05                            # feet on the ground
robot.data.body_link_pos_w[:, fb[1], 2] = 0.05
robot.data.body_link_pos_w[1, fb[0], 2] = 0.60                            # one leg up
robot.data.body_link_pos_w[2, fb[0], 2] = 0.60                            # both legs over the back
robot.data.body_link_pos_w[2, fb[1], 2] = 0.60
robot.data.body_link_pos_w[3, fb[0], 2] = 0.18                            # within the margin
fa = gm.feet_above_pelvis(env, slip_cfg, margin=0.05).numpy()
check(abs(fa[0]) < 1e-6 and abs(fa[1] - 0.40) < 1e-6 and abs(fa[2] - 0.80) < 1e-6 and abs(fa[3]) < 1e-6,
      f"feet above the pelvis: feet down 0, one leg up {fa[1]:.2f} m, both {fa[2]:.2f} m, "
      "3 cm above the pelvis 0 (margin 5 cm)")
robot.data.body_link_pos_w = saved_pos
robot.data.root_link_pos_w = saved_root
lc2 = gec.make_getup_env_cfg(pool_path=tmp, push=0.4)
pe = lc2.events.get("push")
check(pe is not None and pe.mode == "interval" and pe.params["velocity_range"]["x"] == (-0.4, 0.4)
      and "push" not in cfg.events, "pushes: an interval event only when asked (training passes 0.4)")
robot.writes.pop("vel", None)
env.episode_length_buf = torch.full((N,), 200, dtype=torch.long)       # past the pace
gm.push_when_up(env, torch.arange(N), {"x": (0.4, 0.4)}, pace_s=3.0)
v = robot.writes.get("vel")
check(v is not None and v.shape[0] == 2 and np.allclose(v.numpy()[:, 0], 0.4),
      "push: only the robots upright on their feet (squatting, standing) get it, +0.4 m/s; "
      "not lying, kneeling or in the air")
robot.writes.pop("vel", None)
env.episode_length_buf = torch.full((N,), 50, dtype=torch.long)        # 1 s: before the pace
gm.push_when_up(env, torch.arange(N), {"x": (0.4, 0.4)}, pace_s=3.0)
check("vel" not in robot.writes, "no pushes before the pace (still getting up)")
env.episode_length_buf = torch.zeros(N, dtype=torch.long)

print("action noise cap")
from getup import train_utils as tu  # noqa: E402


def fake_runner(kind):
    dist = SimpleNamespace()
    if kind == "log":
        dist.log_std_param = torch.nn.Parameter(torch.tensor(np.log([0.3, 0.7, 0.9])))
    else:
        dist.std_param = torch.nn.Parameter(torch.tensor([0.3, 0.7, 0.9]))
    actor = SimpleNamespace(distribution=dist)
    calls = []

    def upd():
        calls.append(1)
        q = dist.log_std_param if kind == "log" else dist.std_param
        q.data = q.data + 0.5
        return {}
    if kind == "inference":       # no alg.actor: found through the inference policy
        alg = SimpleNamespace(update=upd)
        return SimpleNamespace(alg=alg, get_inference_policy=lambda: actor,
                               train_mode=lambda: calls.append("train")), dist, calls
    return SimpleNamespace(alg=SimpleNamespace(actor=actor, update=upd)), dist, calls


caps = []
for kind in ("std", "log", "inference"):
    r_, d_, c_ = fake_runner(kind)
    ok_install = tu.install_std_limit(r_, 0.4)
    r_.alg.update()
    s_ = np.exp(d_.log_std_param.data.numpy()) if kind == "log" else d_.std_param.data.numpy()
    caps.append(ok_install and np.all(s_ <= 0.4 + 1e-6) and s_.min() >= 0.05 - 1e-6 and 1 in c_)
check(all(caps), "noise cap: std and log-std parameters, also found through the inference "
      "policy; clamped after every update")
check(not tu.install_std_limit(SimpleNamespace(alg=SimpleNamespace(actor=SimpleNamespace(),
                                                                    update=lambda: None)), 0.4),
      "noise cap: no std parameter -> False (the run goes on without the cap)")

# -- 5. command term: bookkeeping, assist, curriculum ------------------------------------------------
print("command term")
gcfg = dataclasses.replace(g)
term = gcfg.build(env)
all_ids = torch.arange(N)
term.reset(all_ids)
w_rate = env.reward_manager.get_term_cfg("action_rate_l2").weight
check(abs(w_rate - 0.2 * rew["action_rate_l2"].weight) < 1e-12,
      f"regularization starts at 20 %: action rate {w_rate:.3f} (full {rew['action_rate_l2'].weight})")
for i in range(N):            # envs 0-3 lying on the back, 4-7 standing
    if i < 4:
        set_state(i, 0.12, 0.13, 90, ("pelvis", "torso_link"))
    else:
        set_state(i, 0.76, 1.23, 2)
for step in range(60):
    env.episode_length_buf += 1
    term.compute(env.step_dt)
st_t = term.stand_time.numpy()
check(np.allclose(st_t[:4], 0) and np.allclose(st_t[4:], 60 * 0.02, atol=1e-4),
      f"standing streak: lying 0 s, standing {st_t[4]:.2f} s")
fs = term.first_stand.numpy()
check(np.allclose(fs[:4], -1) and np.allclose(fs[4:], 0.0, atol=0.021),
      f"first 1 s stand began at {fs[4]:.2f} s")
force, bodies = robot.wrench
fz = force.numpy()[:, 0, 2]
check(bodies == robot.find_bodies("torso_link")[0] and np.allclose(fz[:4], 0) and np.allclose(fz[4:], 200),
      "assist force 200 N on the torso only while upright")
check(np.allclose(term.command.numpy()[4], [1.0, 1.0]) and np.allclose(term.command.numpy()[0], [0.0, 1.0]),
      "critic command = [force now, level] / 200 N")
check(np.allclose(term.air_time.numpy(), 0.0), "no flight time while lying or standing")
set_state(0, 0.30, 0.60, 60, contacts=(), feet=(False, False))      # env 0 in the air
for _ in range(10):
    env.episode_length_buf += 1
    term.compute(env.step_dt)
set_state(0, 0.12, 0.13, 90, ("pelvis", "torso_link"))
check(abs(float(term.air_time[0]) - 0.2) < 1e-5 and np.allclose(term.air_time.numpy()[1:], 0.0),
      f"flight time: 10 steps in the air = {float(term.air_time[0]):.2f} s")
ex = term.reset(all_ids)
check(abs(ex["success"] - 0.5) < 1e-6 and abs(ex["stood_up"] - 0.5) < 1e-6
      and abs(ex["time_to_stand_s"] - 0.5 * 10.0) < 0.02 and abs(ex["airborne_s"] - 0.2 / N) < 1e-6,
      f"episode metrics: success {ex['success']:.2f}, time to stand {ex['time_to_stand_s']:.2f} s "
      f"(never = 10 s), flight {ex['airborne_s']:.3f} s (mean)")
check(np.allclose(term.air_time.numpy(), 0.0), "flight time cleared at the reset")
check(abs(term.success_ema - 0.5) < 1e-6 and term.assist == 200.0 and abs(term.reg_scale - 0.24) < 1e-9,
      f"after a 50 % batch: ema {term.success_ema:.2f}, assist {term.assist:.0f} N, reg {term.reg_scale:.2f}")
rounds = 0
while term.assist > 0 and rounds < 100:
    for i in range(N):
        set_state(i, 0.76, 1.23, 2)
    for _ in range(55):
        env.episode_length_buf += 1
        term.compute(env.step_dt)
    term.reset(all_ids)
    env.episode_length_buf = torch.zeros(N, dtype=torch.long)
    rounds += 1
check(9 <= rounds <= 11, f"all succeeding: assist force gone after {rounds} rounds of num_envs episodes")
while term.reg_scale < 1.0 and rounds < 100:
    term.stand_time = torch.full((N,), 2.0)
    term.reset(all_ids)
    rounds += 1
check(term.reg_scale == 1.0 and abs(env.reward_manager.get_term_cfg("joint_pos_limits").weight
                                    - rew["joint_pos_limits"].weight) < 1e-12,
      f"regularization at full weight after {rounds} rounds")
term.compute(env.step_dt)
check(np.allclose(robot.wrench[0].numpy(), 0), "assist force removed once the level is 0")

# -- 6. left-right mirror ------------------------------------------------------------------------------
print("left-right mirror")
H = 5
actor_dims = [(3 * H,), (3 * H,), (23 * H,), (23 * H,), (23 * H,)]
crit_names = list(crit.terms)
size = {"base_ang_vel": 3, "projected_gravity": 3, "joint_pos": 23, "joint_vel": 23, "actions": 23,
        "base_lin_vel": 3, "pelvis_height": 1, "head_height": 1, "foot_height": 2,
        "body_contact": len(cnames), "head_force": 1, "assist": 2, "pace": 1}
env.observation_manager = SimpleNamespace(
    active_terms={"actor": list(actor.terms), "critic": crit_names},
    group_obs_term_dim={"actor": actor_dims, "critic": [(size[k],) for k in crit_names]})
env.action_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(target_names=JOINTS))
m = sym._Mirror(env)
xa = torch.tensor(np.random.default_rng(3).normal(size=(4, sum(d[0] for d in actor_dims))))
xc = torch.tensor(np.random.default_rng(4).normal(size=(4, sum(size[k] for k in crit_names))))
check(np.allclose(m.group("actor", m.group("actor", xa)).numpy(), xa.numpy())
      and np.allclose(m.group("critic", m.group("critic", xc)).numpy(), xc.numpy()),
      "mirror twice = identity (policy with history, critic)")
# one known frame: step 3 of the history
frame = np.zeros(75, dtype=np.float32)
frame[0:3] = (0.1, 0.2, 0.3)                      # ang vel
frame[3:6] = (0.0, 0.5, -0.8)                     # gravity
j = 6
frame[j + JOINTS.index("left_hip_roll_joint")] = 0.3
frame[j + JOINTS.index("left_knee_joint")] = 1.0
frame[j + JOINTS.index("waist_yaw_joint")] = 0.2
x = np.zeros((1, 375), dtype=np.float32)
for name, a_, wd in (("base_ang_vel", 0, 3), ("projected_gravity", 15, 3), ("joint_pos", 30, 23)):
    off = {"base_ang_vel": 0, "projected_gravity": 3, "joint_pos": 6}[name]
    x[0, a_ + 3 * wd: a_ + 4 * wd] = frame[off: off + wd]
y = m.group("actor", torch.tensor(x)).numpy()[0]
av = y[9:12]
gr = y[15 + 9:15 + 12]
jp = y[30 + 3 * 23:30 + 4 * 23]
check(np.allclose(av, (-0.1, 0.2, -0.3)) and np.allclose(gr, (0.0, -0.5, -0.8)),
      "mirror: angular velocity x, z and gravity y change sign (in the right history step)")
check(abs(jp[JOINTS.index("right_hip_roll_joint")] + 0.3) < 1e-6
      and abs(jp[JOINTS.index("right_knee_joint")] - 1.0) < 1e-6
      and abs(jp[JOINTS.index("waist_yaw_joint")] + 0.2) < 1e-6
      and abs(jp[JOINTS.index("left_knee_joint")]) < 1e-6,
      "mirror: left hip roll 0.3 -> right -0.3, left knee -> right knee, waist yaw sign")
c = np.zeros((1, xc.shape[1]), dtype=np.float32)
off = 0
for k in crit_names:
    if k == "body_contact":
        c[0, off + cnames.index("left_knee_link")] = 1.0
        bc = off
    off += size[k]
yc = m.group("critic", torch.tensor(c)).numpy()[0]
check(yc[bc + cnames.index("right_knee_link")] == 1.0 and yc[bc + cnames.index("left_knee_link")] == 0.0,
      "mirror: left knee on the ground -> right knee")
a = torch.tensor(np.random.default_rng(5).normal(size=(3, 23)))
_, aa = sym.g1_getup_symmetry(env, None, a)
check(aa.shape == (6, 23) and np.allclose(m.actions(aa[3:]).numpy(), a.numpy()), "actions augmented")

# -- 7. reset event ------------------------------------------------------------------------------------
print("reset event")
ev = gm.reset_to_fallen_pose(SimpleNamespace(params={"pool_path": str(tmp), "posture_weights": None}), env)
env.scene.env_origins = torch.tensor(np.tile([[10.0, -5.0, 0.0]], (N, 1)))
ev(env, all_ids, xy_range=0.5)
pose, eids = robot.writes["pose"]
pose = pose.numpy()
chosen = ev.env_posture.numpy()
cls_new = [ps.classify(q) for q in pose[:, 3:7]]
check(list(cls_new) == list(chosen), "a random heading keeps how the robot lies (supine stays supine)")
check(np.all(np.abs(pose[:, 0] - 10.0) <= 0.5) and np.all(np.abs(pose[:, 1] + 5.0) <= 0.5),
      "placed within 0.5 m of the env origin")
zs = sorted(float(z) for z in pool.root_pos[:, 2] + 0.003)
check(all(any(abs(z - p) < 1e-5 for p in zs) for z in pose[:, 2]), "height = resting height + 3 mm")
jw = robot.writes["joints"].numpy()
rows = []
for k in range(N):   # find which pool pose each env got (by its joint angles, in model order)
    model_order = pool.joint_pos[:, [pool.joint_names.index(nm) for nm in JOINTS]]
    rows.append(int(np.argmin(np.abs(model_order - jw[k]).sum(axis=1))))
check(all(np.allclose(pool.joint_pos[r, [pool.joint_names.index(nm) for nm in JOINTS]], jw[k], atol=1e-5)
          for k, r in enumerate(rows)), "joint angles mapped by name (pool stored in another order)")
check(np.allclose(robot.writes["vel"].numpy(), 0), "starts at rest")
ev.set_weights({"prone": 1.0})
ev(env, all_ids)
check(all(ps.POSTURES[int(p)] == "prone" for p in ev.env_posture.numpy()), "set_weights(prone): only prone")
# a pool made before the head capsule (no "head" in its meta): warned at load
import contextlib  # noqa: E402
import io  # noqa: E402
old_pool = dataclasses.replace(pool, meta={"test": 1})
tmp_old = Path(tempfile.mkdtemp()) / "old_pool.npz"
ps.save_pool(tmp_old, old_pool)
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    gm.reset_to_fallen_pose(SimpleNamespace(params={"pool_path": str(tmp_old), "posture_weights": None}), env)
check("another head collision shape" in _buf.getvalue() and "make_fallen_poses" in _buf.getvalue(),
      "a pool made with the old head sphere is warned about at load")
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    gm.reset_to_fallen_pose(SimpleNamespace(params={"pool_path": str(tmp), "posture_weights": None}), env)
check("WARNING" not in _buf.getvalue(), "a pool made with this head: no warning")

print()
print("ALL OK" if OK else "SOME CHECKS FAILED")
sys.exit(0 if OK else 1)
