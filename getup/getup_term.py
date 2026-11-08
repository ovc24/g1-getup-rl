"""The "getup" command term: episode bookkeeping, the assist force and the curriculum.

It gives the policy no command (getting up needs none); it is a command term because
mjlab calls command terms once per env step (after the physics and the resets) and at
every reset, which is what this needs:

1. Bookkeeping per episode: how long the robot has been standing (getup_mdp.getup_state),
   when it first stood for ``hold_s``, how long the head touched the ground and how hard,
   how long nothing touched the ground (jumps). Logged at every reset as
   Metrics/getup/<name>.

2. Assist force (HoST's "pulling force"): an upward force on the torso while the pelvis
   is fairly upright (-gravity_z > 0.8, < 37 deg tilt), 200 N at the start (~60 % of the
   robot's weight). It helps the first, clumsy rises from sitting or kneeling; the
   curriculum takes it away. The critic sees it (it changes what happens), the policy
   does not (the real robot has none).

3. Curriculum, from the success rate of the episodes that end (success = standing for at
   least ``hold_s`` when the episode ends), as a moving average over ~num_envs episodes:
     - above ``assist_success``: the assist force goes down, to 0 in ``assist_rounds``
       x num_envs episodes; it never goes back up;
     - above ``reg_success``: the regularization rewards (``reg_terms``, getup_env_cfg.py)
       go from ``reg_start`` x their weight to the full weight in ``reg_rounds`` x
       num_envs episodes.
   Neither is saved in the checkpoints: to continue a run, pass the values it had
   (train_getup.py --assist-start / --reg-start, read them in the logs).

command (critic only): [assist force now / initial force, assist level / initial force].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .getup_mdp import STAND, StandCfg, airborne_now, getup_state, head_contact_force
from .robot import TORSO_BODY

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class GetUpCommand(CommandTerm):
    cfg: "GetUpCommandCfg"

    def __init__(self, cfg: "GetUpCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.robot_name]
        ids, _ = self.robot.find_bodies(cfg.torso_body)
        if not ids:
            raise ValueError(f"body '{cfg.torso_body}' not found in '{cfg.robot_name}'")
        self._torso = ids[:1]
        n, dev = self.num_envs, self.device
        self.stand_time = torch.zeros(n, device=dev)       # current standing streak, s
        self.first_stand = torch.full((n,), -1.0, device=dev)   # start of the first hold
        self.head_time = torch.zeros(n, device=dev)        # head on the ground, s
        self.head_peak = torch.zeros(n, device=dev)        # largest head force, N
        self.air_time = torch.zeros(n, device=dev)         # nothing on the ground, s
        self.applied = torch.zeros(n, device=dev)          # assist force this step, N
        self._cmd = torch.zeros(n, 2, device=dev)
        self._zero = torch.zeros(n, 1, 3, device=dev)
        self.assist = float(cfg.assist_force if cfg.assist_start is None else cfg.assist_start)
        self.reg_scale = float(cfg.reg_start if cfg.reg_scale_start is None
                               else cfg.reg_scale_start)
        self.success_ema = 0.0
        self.episodes = 0
        self._weights_dirty = True
        self._force_written = False
        for name in ("success", "stood_up", "time_to_stand_s", "head_ground_s",
                     "head_peak_N", "airborne_s", "assist_N", "reg_scale", "success_ema"):
            self.metrics[name] = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def episode_time(self) -> torch.Tensor:
        return self._env.episode_length_buf.float() * self._env.step_dt

    # -- curriculum -----------------------------------------------------------------------
    def _apply_reg_weights(self) -> None:
        rm = getattr(self._env, "reward_manager", None)
        if rm is None:              # rewards are built after the commands: next step
            return
        for name, base in self.cfg.reg_terms.items():
            rm.get_term_cfg(name).weight = base * self.reg_scale
        self._weights_dirty = False

    def _update_curriculum(self, env_ids: torch.Tensor) -> None:
        c = self.cfg
        k = int(env_ids.numel())
        if not c.curriculum or k == 0:
            return
        success = float((self.stand_time[env_ids] >= c.hold_s).float().mean())
        self.episodes += k
        n = float(self.num_envs)
        a = min(1.0, k / (n * c.success_window))
        self.success_ema += a * (success - self.success_ema)
        if self.success_ema > c.assist_success and self.assist > 0.0:
            self.assist = max(0.0, self.assist - c.assist_force * k / (n * c.assist_rounds))
        if self.success_ema > c.reg_success and self.reg_scale < 1.0:
            step = (1.0 - c.reg_start) * k / (n * c.reg_rounds)
            self.reg_scale = min(1.0, self.reg_scale + step)
            self._weights_dirty = True

    # -- CommandTerm interface ------------------------------------------------------------
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(self.device, dtype=torch.long).flatten()
        # The state still holds the episodes that just ended (the reset events have moved
        # the robots, but these per-env numbers are only cleared here).
        self._update_curriculum(env_ids)
        self.stand_time[env_ids] = 0.0
        self.first_stand[env_ids] = -1.0
        self.head_time[env_ids] = 0.0
        self.head_peak[env_ids] = 0.0
        self.air_time[env_ids] = 0.0
        self.applied[env_ids] = 0.0
        if self._weights_dirty:
            self._apply_reg_weights()

    def _update_metrics(self) -> None:
        c = self.cfg
        ep_len = float(self._env.max_episode_length_s)
        self.metrics["success"][:] = (self.stand_time >= c.hold_s).float()
        self.metrics["stood_up"][:] = (self.first_stand >= 0.0).float()
        self.metrics["time_to_stand_s"][:] = torch.where(
            self.first_stand >= 0.0, self.first_stand, torch.full_like(self.first_stand, ep_len))
        self.metrics["head_ground_s"][:] = self.head_time
        self.metrics["head_peak_N"][:] = self.head_peak
        self.metrics["airborne_s"][:] = self.air_time
        self.metrics["assist_N"][:] = self.assist
        self.metrics["reg_scale"][:] = self.reg_scale
        self.metrics["success_ema"][:] = self.success_ema

    def _update_command(self) -> None:
        c = self.cfg
        if self._weights_dirty:
            self._apply_reg_weights()
        dt = self._env.step_dt
        s = getup_state(self._env, c.stand, c.contact_sensor)
        self.stand_time = torch.where(s.standing, self.stand_time + dt,
                                      torch.zeros_like(self.stand_time))
        new = (self.stand_time >= c.hold_s) & (self.first_stand < 0.0)
        self.first_stand = torch.where(new, self.episode_time() - c.hold_s, self.first_stand)
        head = self._env.scene[c.head_sensor].data.found
        assert head is not None
        on = (head.reshape(head.shape[0], -1) > 0).any(dim=1)
        self.head_time = self.head_time + dt * on.float()
        self.head_peak = torch.maximum(self.head_peak, head_contact_force(self._env, c.head_sensor))
        air = airborne_now(s) & (self.episode_time() > c.airborne_grace_s)
        self.air_time = self.air_time + dt * air.float()

        # Assist force on the torso, world +z, only while the pelvis is upright enough.
        f0 = max(c.assist_force, 1e-6)
        if self.assist > 0.0:
            self.applied = self.assist * (s.up > c.assist_min_up).float()
            force = self._zero.clone()
            force[:, 0, 2] = self.applied
            self.robot.write_external_wrench_to_sim(force, self._zero, body_ids=self._torso)
            self._force_written = True
        elif self._force_written:         # level reached 0: clear the last force once
            self.applied = torch.zeros_like(self.applied)
            self.robot.write_external_wrench_to_sim(self._zero, self._zero, body_ids=self._torso)
            self._force_written = False
        self._cmd[:, 0] = self.applied / f0
        self._cmd[:, 1] = self.assist / f0
        self._update_metrics()          # fresh for the reset logging of the next step


@dataclass(kw_only=True)
class GetUpCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)   # never mid-episode
    robot_name: str = "robot"
    torso_body: str = TORSO_BODY
    contact_sensor: str = "body_ground"
    head_sensor: str = "head_ground"
    stand: StandCfg = STAND
    hold_s: float = STAND.hold_s        # standing this long = got up
    airborne_grace_s: float = 0.1       # the reset puts the robot 3 mm above the ground
    # Assist force (N, upwards on the torso). 0 = none (evaluation, deployment checks).
    assist_force: float = 200.0
    assist_start: float | None = None   # continue a run: its last level; None = assist_force
    assist_min_up: float = 0.8
    # Curriculum (see the module docstring).
    curriculum: bool = True
    success_window: float = 1.0         # moving average over this x num_envs episodes
    assist_success: float = 0.5
    assist_rounds: float = 10.0
    reg_terms: dict[str, float] = field(default_factory=dict)   # name -> full weight
    reg_start: float = 0.2
    reg_scale_start: float | None = None
    reg_success: float = 0.3
    reg_rounds: float = 20.0

    def build(self, env: "ManagerBasedRlEnv") -> GetUpCommand:
        return GetUpCommand(self, env)
