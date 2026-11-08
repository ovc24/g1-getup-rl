"""Rewards, observations, terminations and the reset event of the get-up task.

Reward design (after HoST, "Learning Humanoid Standing-up Control across Diverse
Postures", RSS 2025, which got the G1-23DoF up on the real robot; MIT-licensed code):

  task       head high AND pelvis upright (product), plus pelvis height. Always on.
  standing   only once the robot stands (pelvis > 0.62 m, head > 1.0 m, pelvis tilt
             < 32 deg, nothing but the feet on the ground): keep still, walking pose,
             both feet on the ground. They make standing worth much more than sitting
             or kneeling, which already get most of the task reward.
  style      head (camera) on the ground, head impacts, joints far from the walking pose
             (waist, hip yaw/roll) beyond a threshold, nothing on the ground (a jump).
  regular.   torques, joint speeds and accelerations, action rate, joint limits, motor
             speed and torque budgets. Scaled by a curriculum (getup_term.py): weak at
             first, so that the robot dares to move, full once it gets up often.

Pace (``pace_s``, 0 = off). Without it every second spent standing pays the full standing
reward, so getting up 0.1 s earlier is worth more than all the regularization of the
movement costs, and the policy learns the fastest way up there is (up in 0.5 s, joints
near the motors' no-load speed). With a pace, the task targets rise from lying to
standing over the first ``pace_s`` seconds of the episode (``pace_phase``) and the
standing rewards start at ``pace_s``: being ahead of the schedule pays exactly as much as
being on it, so standing up earlier earns nothing and the regularization decides how
fast to move. The critic sees the phase (observation ``pace``); the policy does not (it
is a pure feedback controller: on the robot it needs no clock).

``tolerance`` is the smooth "reward 1 inside [lower, upper], falling off outside" used by
HoST (from dm_control): Gaussian, value_at_margin at a distance of `margin`.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform
from mjlab.utils.lab_api.string import resolve_matching_names_values

from .poses import DEFAULT_POOL, POSTURES, load_pool, sample_weights
from .robot import CONTACT_BODIES, FOOT_BODIES, HEAD_GEOM, HEAD_SHAPE
from .stand import DEFAULT_POSTURE_WEIGHTS, STAND, StandCfg  # noqa: F401  (re-exported)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def tolerance(x: torch.Tensor, lower: "float | torch.Tensor", upper: float = math.inf,
              margin: float = 0.0, value_at_margin: float = 0.1) -> torch.Tensor:
    """1 inside [lower, upper]; outside exp(-0.5 (d * s)^2) with d = distance / margin and
    s chosen so that the value is value_at_margin at d = 1 (dm_control's gaussian).
    lower may be a tensor (one bound per env: the paced targets)."""
    inside = (x >= lower) & (x <= upper)
    if margin <= 0.0:
        return inside.float()
    d = torch.where(x < lower, lower - x, x - upper) / margin
    s = math.sqrt(-2.0 * math.log(value_at_margin))
    return torch.where(inside, torch.ones_like(x), torch.exp(-0.5 * (d * s) ** 2))


# -- state shared by rewards, command and evaluation --------------------------------------
class _Ids(NamedTuple):
    head: int                   # model (global) geom id of the head sphere
    feet_cols: list[int]        # columns of the ground contact sensor: left, right foot
    other_cols: list[int]       # all other bodies
    contact_names: list[str]


_IDS: dict[int, _Ids] = {}


def robot_ids(env: "ManagerBasedRlEnv", robot_name: str = "robot") -> _Ids:
    key = id(env)
    ids = _IDS.get(key)
    if ids is None:
        robot = env.scene[robot_name]
        head, _ = robot.find_geoms(HEAD_GEOM)
        head_gid = int(robot.data.indexing.geom_ids[head[0]])
        # Same call as the contact sensor makes for its primaries, so same order.
        _, names = robot.find_bodies(CONTACT_BODIES)
        feet = [names.index(b) for b in FOOT_BODIES]
        other = [i for i in range(len(names)) if i not in feet]
        ids = _Ids(head_gid, feet, other, list(names))
        _IDS[key] = ids
    return ids


class GetUpState(NamedTuple):
    pelvis_z: torch.Tensor      # (N,) m
    head_z: torch.Tensor        # (N,) m
    up: torch.Tensor            # (N,) -gravity_z in the pelvis frame (1 = upright)
    feet_contact: torch.Tensor  # (N, 2) bool, left, right
    other_contact: torch.Tensor  # (N,) bool: a body other than the feet on the ground
    standing: torch.Tensor      # (N,) bool


def getup_state(env: "ManagerBasedRlEnv", stand: StandCfg = STAND,
                sensor_name: str = "body_ground") -> GetUpState:
    robot = env.scene["robot"]
    ids = robot_ids(env)
    pelvis_z = robot.data.root_link_pos_w[:, 2]
    head_z = robot.data.data.geom_xpos[:, ids.head, 2]
    up = -robot.data.projected_gravity_b[:, 2]
    found = env.scene[sensor_name].data.found
    assert found is not None
    touch = found > 0
    feet = touch[:, ids.feet_cols]
    other = touch[:, ids.other_cols].any(dim=1)
    standing = ((pelvis_z > stand.pelvis_min) & (head_z > stand.head_min)
                & (up > stand.up_min) & ~other)
    return GetUpState(pelvis_z, head_z, up, feet, other, standing)


def airborne_now(s: GetUpState) -> torch.Tensor:
    """(N,) bool: nothing of the robot touches the ground."""
    return ~s.feet_contact.any(dim=1) & ~s.other_contact


# -- pace -------------------------------------------------------------------------------------
def episode_time(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Seconds since the episode began (mjlab counts the step before computing rewards,
    so the first reward after a reset sees one step, 0.02 s)."""
    return env.episode_length_buf.float() * env.step_dt


def pace_phase(env: "ManagerBasedRlEnv", pace_s: float) -> torch.Tensor:
    """(N,) 0 -> 1 over the first pace_s seconds of the episode, then 1. pace_s <= 0: 1."""
    if pace_s <= 0.0:
        return torch.ones(env.num_envs, device=env.device)
    return (episode_time(env) / pace_s).clamp(min=0.0, max=1.0)


def _paced(start: float, end: float, phase: torch.Tensor) -> torch.Tensor:
    return start + (end - start) * phase


def _after_pace(env: "ManagerBasedRlEnv", pace_s: float) -> torch.Tensor:
    """1 from pace_s on (always with pace_s <= 0): when the standing rewards count."""
    return (pace_phase(env, pace_s) >= 1.0).float()


# -- task rewards --------------------------------------------------------------------------
def head_up(env: "ManagerBasedRlEnv", target: float = 1.10, margin: float = 0.8,
            up_target: float = 0.95, up_margin: float = 1.0, pace_s: float = 0.0,
            head_start: float = 0.15, up_start: float = 0.0) -> torch.Tensor:
    """HoST's task reward: head height x pelvis uprightness, both in [0, 1]. Lying ~0,
    sitting up ~0.5, kneeling up ~0.85, standing 1. With pace_s > 0 both targets rise
    from lying (head_start, up_start) to standing (target, up_target) over pace_s."""
    s = getup_state(env)
    p = pace_phase(env, pace_s)
    return (tolerance(s.head_z, _paced(head_start, target, p), margin=margin,
                      value_at_margin=0.1)
            * tolerance(s.up, _paced(up_start, up_target, p), margin=up_margin,
                        value_at_margin=0.05))


def pelvis_height(env: "ManagerBasedRlEnv", target: float = 0.70, margin: float = 0.55,
                  pace_s: float = 0.0, start: float = 0.10) -> torch.Tensor:
    """Pelvis off the ground: separates sitting (head up, pelvis down) from standing.
    With pace_s > 0 the target rises from `start` (lying) over pace_s."""
    p = pace_phase(env, pace_s)
    return tolerance(env.scene["robot"].data.root_link_pos_w[:, 2], _paced(start, target, p),
                     margin=margin, value_at_margin=0.1)


# -- rewards once standing (and, with a pace, from pace_s on) --------------------------------
def stand_still(env: "ManagerBasedRlEnv", lin_std: float = 0.3, ang_std: float = 0.7,
                pace_s: float = 0.0) -> torch.Tensor:
    """Standing and not moving (pelvis horizontal speed, angular speed)."""
    s = getup_state(env)
    robot = env.scene["robot"]
    v = robot.data.root_link_lin_vel_w[:, :2].square().sum(dim=1)
    w = robot.data.root_link_ang_vel_w.square().sum(dim=1)
    return (s.standing.float() * torch.exp(-v / lin_std ** 2) * torch.exp(-w / ang_std ** 2)
            * _after_pace(env, pace_s))


def stand_pose(env: "ManagerBasedRlEnv", std: float = 0.3, pace_s: float = 0.0) -> torch.Tensor:
    """Standing in the walking policy's default pose (so it can take over): exp(-mean
    squared joint error / std^2), std in rad."""
    s = getup_state(env)
    robot = env.scene["robot"]
    err = (robot.data.joint_pos - robot.data.default_joint_pos).square().mean(dim=1)
    return s.standing.float() * torch.exp(-err / std ** 2) * _after_pace(env, pace_s)


def feet_support(env: "ManagerBasedRlEnv", pace_s: float = 0.0) -> torch.Tensor:
    """Standing on both feet (standing already means no other body on the ground)."""
    s = getup_state(env)
    return (s.standing & s.feet_contact.all(dim=1)).float() * _after_pace(env, pace_s)


# -- style ---------------------------------------------------------------------------------
def head_ground_contact(env: "ManagerBasedRlEnv", sensor_name: str = "head_ground") -> torch.Tensor:
    """1 while the head (the camera) touches the ground."""
    found = env.scene[sensor_name].data.found
    assert found is not None
    return (found.reshape(found.shape[0], -1) > 0).any(dim=1).float()


def head_contact_force(env: "ManagerBasedRlEnv", sensor_name: str = "head_ground") -> torch.Tensor:
    """Largest head-ground contact force (N) over the physics substeps of this step."""
    sensor = env.scene[sensor_name]
    fh = sensor.data.force_history            # (N, 1, H, 3), index 0 = latest substep
    if fh is None:
        f = sensor.data.force
        assert f is not None
        return torch.linalg.norm(f.reshape(f.shape[0], -1, 3), dim=-1).amax(dim=1)
    return torch.linalg.norm(fh, dim=-1).reshape(fh.shape[0], -1).amax(dim=1)


def head_impact(env: "ManagerBasedRlEnv", sensor_name: str = "head_ground",
                threshold: float = 50.0, scale: float = 100.0) -> torch.Tensor:
    """(head contact force - threshold) / scale, if positive: hitting the head is much
    worse than resting it."""
    return ((head_contact_force(env, sensor_name) - threshold) / scale).clamp(min=0.0)


class joint_deviation_beyond:
    """Sum over the listed joints of (|q - q_default| - threshold)+, rad. Keeps the waist
    and hips from twisting far (HoST's waist/hip yaw/hip roll deviation penalties, as a
    smooth version)."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        robot = env.scene["robot"]
        ids, _, values = resolve_matching_names_values(cfg.params["thresholds"],
                                                       robot.joint_names)
        self._ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        self._thr = torch.tensor(values, device=env.device, dtype=torch.float32)

    def __call__(self, env: "ManagerBasedRlEnv", thresholds) -> torch.Tensor:
        del thresholds
        robot = env.scene["robot"]
        dev = (robot.data.joint_pos[:, self._ids]
               - robot.data.default_joint_pos[:, self._ids]).abs()
        return (dev - self._thr).clamp(min=0.0).sum(dim=1)


_SPEED_LIMITS: dict[int, torch.Tensor] = {}


def joint_overspeed(env: "ManagerBasedRlEnv",
                    groups: tuple[tuple[tuple[str, ...], float], ...],
                    ratio: float = 0.9, power: float = 2.0) -> torch.Tensor:
    """Sum over joints of (|joint speed| - ratio x motor no-load speed)^power, if > 0.
    groups: ((joint name patterns, no-load speed rad/s), ...)."""
    robot = env.scene["robot"]
    limits = _SPEED_LIMITS.get(id(env))
    if limits is None:
        limits = torch.full((robot.data.joint_vel.shape[1],), float("inf"), device=env.device)
        for names, vmax in groups:
            ids, _ = robot.find_joints(names)
            limits[ids] = vmax
        _SPEED_LIMITS[id(env)] = limits
    excess = (robot.data.joint_vel.abs() - ratio * limits).clamp(min=0.0)
    return excess.pow(power).sum(dim=1)


def airborne(env: "ManagerBasedRlEnv", grace_s: float = 0.1) -> torch.Tensor:
    """1 while nothing touches the ground (a jump, the flight of a kip-up). Not during the
    first grace_s of an episode: the reset puts the robot 3 mm above its resting pose."""
    s = getup_state(env)
    return (airborne_now(s) & (episode_time(env) > grace_s)).float()


class torque_over_budget:
    """Sum over joints of (|PD torque asked| / peak torque - budget)+^2.

    The PD torque asked is kp (target - q) - kd qd with the nominal gains, before the
    motor limits clip it. Keeps the motors away from their limit: no arm pressed against
    the body at full torque, no bang-bang pushes, and on the real robot (whose torque
    limits are not exactly the simulated ones) the policy works where the motors still
    follow the PD law. params: groups ((joint patterns, kp, kd, peak torque), ...),
    budget (share of the peak torque that costs nothing)."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        robot = env.scene["robot"]
        n = len(robot.joint_names)
        kp, kd, peak = [0.0] * n, [0.0] * n, [0.0] * n
        for names, stiffness, damping, effort in cfg.params["groups"]:
            ids, _ = robot.find_joints(names)
            for i in ids:
                kp[i], kd[i], peak[i] = float(stiffness), float(damping), float(effort)
        missing = [robot.joint_names[i] for i in range(n) if peak[i] <= 0.0]
        if missing:
            raise ValueError(f"torque_over_budget: no motor group for {missing}")
        self._kp = torch.tensor(kp, device=env.device)
        self._kd = torch.tensor(kd, device=env.device)
        self._peak = torch.tensor(peak, device=env.device)

    def ratio(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        """(N, joints) |PD torque asked| / peak torque."""
        d = env.scene["robot"].data
        tau = self._kp * (d.joint_pos_target - d.joint_pos) - self._kd * d.joint_vel
        return tau.abs() / self._peak

    def __call__(self, env: "ManagerBasedRlEnv", groups, budget: float = 0.8) -> torch.Tensor:
        del groups
        return (self.ratio(env) - budget).clamp(min=0.0).square().sum(dim=1)


# -- terminations ---------------------------------------------------------------------------
def blown_up(env: "ManagerBasedRlEnv", max_joint_vel: float = 60.0,
             max_lin_vel: float = 12.0) -> torch.Tensor:
    """The simulation went wrong (huge speeds or NaN): end the episode."""
    robot = env.scene["robot"]
    qd = robot.data.joint_vel
    v = robot.data.root_link_lin_vel_w
    bad = (qd.abs().amax(dim=1) > max_joint_vel) | (torch.linalg.norm(v, dim=1) > max_lin_vel)
    return bad | ~torch.isfinite(qd).all(dim=1) | ~torch.isfinite(v).all(dim=1)


# -- observations (critic) ----------------------------------------------------------------
def pelvis_height_obs(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def head_height_obs(env: "ManagerBasedRlEnv") -> torch.Tensor:
    ids = robot_ids(env)
    return env.scene["robot"].data.data.geom_xpos[:, ids.head, 2:3]


def body_ground_contact(env: "ManagerBasedRlEnv", sensor_name: str = "body_ground") -> torch.Tensor:
    """1/0 per watched body (CONTACT_BODIES order): touching the ground."""
    found = env.scene[sensor_name].data.found
    assert found is not None
    return (found > 0).float()


def head_force_obs(env: "ManagerBasedRlEnv", sensor_name: str = "head_ground") -> torch.Tensor:
    return (head_contact_force(env, sensor_name) / 100.0).clamp(max=5.0).unsqueeze(1)


def pace_obs(env: "ManagerBasedRlEnv", pace_s: float = 0.0) -> torch.Tensor:
    """Critic only: where the pace schedule is (0 at the start, 1 from pace_s on; always 1
    without a pace). The rewards depend on it, so the critic needs it."""
    return pace_phase(env, pace_s).unsqueeze(1)


def feet_slip(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sum over the feet on the ground of their horizontal speed squared ((m/s)^2):
    getting up by sliding the feet works on the simulated floor, not on every real one.
    asset_cfg: the robot with body_names=FOOT_BODIES (left, right, as feet_contact)."""
    s = getup_state(env)
    v = env.scene[asset_cfg.name].data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    return (v.square().sum(dim=-1) * s.feet_contact.float()).sum(dim=1)


def feet_above_pelvis(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg,
                      margin: float = 0.05) -> torch.Tensor:
    """Sum over the feet of how far (m) each is above the pelvis plus a margin: legs swung
    up for momentum (high kicks, both legs over the back while lying face down). Getting
    up on knees and hands needs none of it. asset_cfg: the robot with
    body_names=FOOT_BODIES."""
    robot = env.scene[asset_cfg.name]
    feet_z = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    pelvis_z = robot.data.root_link_pos_w[:, 2:3]
    return (feet_z - pelvis_z - margin).clamp(min=0.0).sum(dim=1)


def push_when_up(env: "ManagerBasedRlEnv", env_ids: torch.Tensor,
                 velocity_range: dict[str, tuple[float, float]], min_up: float = 0.8,
                 pace_s: float = 0.0) -> None:
    """Interval event: a push (pelvis velocity change, as in the walking training) for the
    envs whose robot is upright on its feet (pelvis < 37 deg from vertical, nothing but
    the feet on the ground) and past the pace: it must keep its balance, or fall and get
    up again. Robots still getting up are left alone."""
    if env_ids is None or len(env_ids) == 0:
        return
    s = getup_state(env)
    ok = ((s.up[env_ids] > min_up) & ~s.other_contact[env_ids]
          & s.feet_contact[env_ids].any(dim=1) & (pace_phase(env, pace_s)[env_ids] >= 1.0))
    ids = env_ids[ok]
    if len(ids) == 0:
        return
    robot = env.scene["robot"]
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    lo = torch.tensor([velocity_range.get(k, (0.0, 0.0))[0] for k in keys], device=env.device)
    hi = torch.tensor([velocity_range.get(k, (0.0, 0.0))[1] for k in keys], device=env.device)
    vel = robot.data.root_link_vel_w[ids].clone()
    vel = vel + lo + (hi - lo) * torch.rand(len(ids), 6, device=env.device)
    robot.write_root_link_velocity_to_sim(vel, env_ids=ids)


def foot_height(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height of the feet (asset_cfg: the robot with body_names=FOOT_BODIES; pass a new
    SceneEntityCfg in the term's params, the manager resolves the body ids)."""
    return env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids, 2]


# -- reset event ----------------------------------------------------------------------------
class reset_to_fallen_pose:
    """Reset event: a pose from the fallen-pose pool (poses.py), turned to a random
    heading and placed at a random spot near the env origin, at rest.

    The instance keeps which posture class each env started from (``env_posture``,
    index into poses.POSTURES) for the evaluation, and ``set_weights`` changes how often
    each class is drawn (eval_getup.py)."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv"):
        p = cfg.params
        asset_cfg = p.get("asset_cfg") or SceneEntityCfg("robot")
        self._robot = env.scene[asset_cfg.name]
        self._device = env.device
        path = p.get("pool_path") or DEFAULT_POOL
        pool = load_pool(path)
        names = list(self._robot.joint_names)
        missing = [n for n in names if n not in pool.joint_names]
        if missing:
            raise ValueError(f"{path}: no angles for the joints {missing}")
        cols = [pool.joint_names.index(n) for n in names]
        dev = env.device
        self.joint_pos = torch.tensor(pool.joint_pos[:, cols], dtype=torch.float32, device=dev)
        self.root_z = torch.tensor(pool.root_pos[:, 2], dtype=torch.float32, device=dev)
        self.root_quat = torch.tensor(pool.root_quat, dtype=torch.float32, device=dev)
        self.posture = torch.tensor(pool.posture, dtype=torch.long, device=dev)
        self._posture_np = pool.posture
        self.env_posture = torch.full((env.num_envs,), -1, dtype=torch.long, device=dev)
        self.set_weights(p.get("posture_weights"))
        counts = ", ".join(f"{k} {v}" for k, v in pool.counts().items())
        print(f"[INFO] fallen poses: {len(pool)} from {path} ({counts})")
        if pool.meta.get("head") != HEAD_SHAPE:
            print(f"[WARNING] {path} was made with another head collision shape "
                  f"({pool.meta.get('head', 'the 6 cm sphere')}; now: {HEAD_SHAPE}): some "
                  "poses can start with the head in the ground. Make the pool again "
                  "(make_fallen_poses.py).")

    def set_weights(self, weights: dict[str, float] | None) -> None:
        w = sample_weights(self._posture_np, weights or DEFAULT_POSTURE_WEIGHTS)
        self._probs = torch.tensor(w, dtype=torch.float32, device=self._device)

    def __call__(self, env: "ManagerBasedRlEnv", env_ids, pool_path=None,
                 posture_weights=None, xy_range: float = 0.5, z_offset: float = 0.003,
                 asset_cfg=None) -> None:
        del pool_path, posture_weights, asset_cfg       # used in __init__
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=self._device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=self._device)[env_ids]
        k = len(env_ids)
        if k == 0:
            return
        idx = torch.multinomial(self._probs, k, replacement=True)
        zeros = torch.zeros(k, device=self._device)
        yaw = sample_uniform(-math.pi, math.pi, (k,), self._device)
        quat = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), self.root_quat[idx])
        pos = env.scene.env_origins[env_ids].clone()
        pos[:, :2] += sample_uniform(-xy_range, xy_range, (k, 2), self._device)
        pos[:, 2] += self.root_z[idx] + z_offset
        self._robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        self._robot.write_root_link_velocity_to_sim(torch.zeros(k, 6, device=self._device),
                                                    env_ids=env_ids)
        q = self.joint_pos[idx]
        self._robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.env_posture[env_ids] = self.posture[idx]


def posture_names() -> tuple[str, ...]:
    return POSTURES
