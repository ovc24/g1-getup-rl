"""Left-right mirror of the get-up task, for rsl_rl's symmetry augmentation.

rsl_rl (PPO, ``symmetry_cfg``) calls ``g1_getup_symmetry(env, obs, actions)`` on every
mini-batch and also trains on the mirror image of every sample (the same situation
reflected across the robot's sagittal plane, with the reflected action and the same
advantage). Falling on the left side and on the right side are then the same thing
learned once, and the policy gets up symmetrically.

Mirror conventions (G1-23DoF: left and right joints have the same axes and mirror-image
ranges):
  - joints: swap left_* <-> right_*; roll and yaw joints (also waist_yaw) change sign,
    pitch joints (hip/ankle/shoulder pitch, knee, elbow) keep it;
  - vectors in the robot frame: y changes sign; angular velocities: x and z change sign;
  - left/right pairs (foot heights, ground contact of each body): swapped.

Observation terms with history are (N, H * D), oldest step first (mjlab's term-major
layout): every one of the H steps is mirrored. Everything is built by name from the
running env, so the layout is read, not assumed; unknown terms raise an error.
"""
from __future__ import annotations

import torch

_CACHE: dict[int, "_Mirror"] = {}

# Plain vectors (robot frame) and scalars.
_SIGNS: dict[str, tuple[float, ...]] = {
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0, 1.0),
    "base_lin_vel": (1.0, -1.0, 1.0),
    "pelvis_height": (1.0,),
    "head_height": (1.0,),
    "head_force": (1.0,),
    "assist": (1.0, 1.0),
    "pace": (1.0,),
}
_PAIRS = ("foot_height",)            # (left, right) -> (right, left)


def mirror_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


def joint_mirror_map(names) -> tuple[list[int], list[float]]:
    """(index of the mirror joint, sign) for each joint name, in the given order."""
    index = {n: i for i, n in enumerate(names)}
    perm, sign = [], []
    for n in names:
        m = mirror_name(n)
        if m not in index:
            raise ValueError(f"no mirror joint for '{n}' (expected '{m}')")
        perm.append(index[m])
        sign.append(-1.0 if ("_roll" in n or "_yaw" in n) else 1.0)
    return perm, sign


def name_mirror_perm(names) -> list[int]:
    """Index of the mirror of each name (bodies: left_knee_link <-> right_knee_link)."""
    index = {n: i for i, n in enumerate(names)}
    out = []
    for n in names:
        m = mirror_name(n)
        if m not in index:
            raise ValueError(f"no mirror for '{n}' (expected '{m}')")
        out.append(index[m])
    return out


class _Mirror:
    def __init__(self, env) -> None:
        base = getattr(env, "unwrapped", env)
        dev = base.device
        robot = base.scene["robot"]
        jp, js = joint_mirror_map(list(robot.joint_names))
        ap, as_ = joint_mirror_map(list(base.action_manager.get_term("joint_pos").target_names))
        self.j_perm = torch.tensor(jp, device=dev)
        self.j_sign = torch.tensor(js, device=dev)
        self.a_perm = torch.tensor(ap, device=dev)
        self.a_sign = torch.tensor(as_, device=dev)
        self.n_joints = len(jp)
        self.n_actions = len(ap)
        from .getup_mdp import robot_ids               # needs a running env
        self.contact_perm = torch.tensor(name_mirror_perm(robot_ids(base).contact_names),
                                         device=dev)
        # Column slices of each observation term, per group.
        om = base.observation_manager
        self.layout: dict[str, list[tuple[str, int, int]]] = {}
        for group, names in om.active_terms.items():
            start, items = 0, []
            for name, shape in zip(names, om.group_obs_term_dim[group]):
                size = 1
                for d in shape:
                    size *= int(d)
                items.append((name, start, start + size))
                start += size
            self.layout[group] = items
        self.signs = {k: torch.tensor(v, device=dev) for k, v in _SIGNS.items()}

    def joints(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.j_perm] * self.j_sign

    def actions(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.a_perm] * self.a_sign

    def _frame_size(self, name: str) -> int:
        if name in self.signs:
            return self.signs[name].numel()
        if name in ("joint_pos", "joint_vel"):
            return self.n_joints
        if name == "actions":
            return self.n_actions
        if name in _PAIRS:
            return 2
        if name == "body_contact":
            return int(self.contact_perm.numel())
        raise ValueError(f"don't know how to mirror observation term '{name}'")

    def _frame(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Mirror one step of a term: x (..., D)."""
        if name in self.signs:
            return x * self.signs[name]
        if name in ("joint_pos", "joint_vel"):
            return self.joints(x)
        if name == "actions":
            return self.actions(x)
        if name in _PAIRS:
            return x[..., [1, 0]]
        if name == "body_contact":
            return x[..., self.contact_perm]
        raise ValueError(f"don't know how to mirror observation term '{name}'")

    def term(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Mirror a term of width H * D (H steps of history, oldest first)."""
        d = self._frame_size(name)
        w = x.shape[1]
        if w % d != 0:
            raise ValueError(f"term '{name}' has {w} values, not a multiple of {d}")
        h = w // d
        return self._frame(name, x.reshape(x.shape[0], h, d)).reshape(x.shape[0], w)

    def group(self, group: str, x: torch.Tensor) -> torch.Tensor:
        items = self.layout.get(group)
        if items is None:
            raise ValueError(f"unknown observation group '{group}'")
        if items[-1][2] != x.shape[1]:
            raise ValueError(f"group '{group}': {x.shape[1]} columns, layout says {items[-1][2]}")
        return torch.cat([self.term(n, x[:, a:b]) for n, a, b in items], dim=1)


def g1_getup_symmetry(env, obs=None, actions=None):
    """rsl_rl data augmentation: returns ([obs; mirror(obs)], [actions; mirror(actions)])."""
    key = id(getattr(env, "unwrapped", env))
    m = _CACHE.get(key)
    if m is None:
        m = _Mirror(env)
        _CACHE[key] = m
    obs_aug = None
    if obs is not None:
        mirrored = obs.clone()
        for k in obs.keys():
            mirrored[k] = m.group(k, obs[k])
        obs_aug = torch.cat([obs, mirrored], dim=0)
    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat([actions, m.actions(actions)], dim=0)
    return obs_aug, actions_aug
