"""Get-up environment for the Unitree G1-23DoF (mjlab 1.2.0 + unitree_rl_mjlab).

Every episode (10 s) starts with the robot lying at rest on flat ground, in a pose from
the fallen-pose pool (poses.py): on the back, on the front, on a side, sitting... The
policy has to stand up and then stand still in the walking pose, so that the walking
policy can take over.

What the policy gets (the same five signals as the walking policy minus the velocity
command and the gait clock, all available on the real robot through g1_ctrl):
  pelvis IMU angular velocity (3), gravity in the pelvis frame (3), joint angles minus the
  walking pose (23), joint speeds (23), last action (23) -> 75 numbers, times the last 5
  steps (``history``) = 375. Getting up is a sequence of contact phases, and the history
  lets the policy tell, e.g., "my arm is pushing but nothing moves" from "it moves".
What it does: joint position targets around the walking pose, like the walking policy, but
with a larger scale (robot.action_scale, 0.5 x peak torque / stiffness instead of 0.25).

The critic also gets what the robot cannot measure: pelvis velocity, pelvis and head
height, foot heights, which bodies touch the ground, the head contact force, the assist
force and where the pace schedule is.

How fast (``pace_s``, getup_mdp.py): the task targets rise from lying to standing over
the first 3 s and the standing rewards count from then on, so standing up earlier earns
nothing; joint speeds above 60 % of the motor's no-load speed, PD torques above 80 % of
the peak torque and moments with nothing on the ground cost. Without this the policy
gets up in half a second, like a kip-up.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import getup_mdp as gm
from .getup_term import GetUpCommandCfg
from .robot import (CONTACT_BODIES, FOOT_BODIES, HEAD_GEOM, action_scale, get_robot_cfg,
                    motor_groups)

EPISODE_S = 10.0
HISTORY = 5
PACE_S = 3.0          # task targets: lying -> standing over the first 3 s (0 = no pace)
TORQUE_BUDGET = 0.8   # PD torque asked above 80 % of the motor's peak torque costs (0 = off)
SPEED_BUDGET = 0.6    # joint speed above 60 % of the motor's no-load speed costs
# Regularization rewards: weighted by the curriculum (getup_term.py), see make_rewards.
PUSH = 0.4            # pushes once up: pelvis velocity change up to +-0.4 m/s (0 = none)
FEET_HIGH = 4.0       # feet above the pelvis: 4 per m per second (0 = off; runs before v5)
HEAD_SCALE = 2.0      # head on the ground and head hits: x2 (runs before v5: x1)
JOINT_DR = True       # joint friction, damping, armature per robot (runs before v5: off)
REG_TERMS = ("action_rate_l2", "joint_acc_l2", "joint_vel_l2", "joint_torques_l2",
             "joint_pos_limits", "joint_overspeed", "torque_budget", "feet_slip")


def make_sensors() -> tuple[ContactSensorCfg, ...]:
    # Which bodies touch the ground (one column per body, CONTACT_BODIES in body order).
    body_ground = ContactSensorCfg(
        name="body_ground",
        primary=ContactMatch(mode="body", pattern=CONTACT_BODIES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    # The head (camera) against the ground, force of every physics substep, so that a
    # short hit between two policy steps is not missed.
    head_ground = ContactSensorCfg(
        name="head_ground",
        primary=ContactMatch(mode="geom", pattern=HEAD_GEOM, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=4,
    )
    return body_ground, head_ground


def make_observations(play: bool, history: int,
                      pace_s: float = PACE_S) -> dict[str, ObservationGroupCfg]:
    actor = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2)),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    critic = {name: dataclasses.replace(t, noise=None) for name, t in actor.items()}
    critic.update({
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "pelvis_height": ObservationTermCfg(func=gm.pelvis_height_obs),
        "head_height": ObservationTermCfg(func=gm.head_height_obs),
        "foot_height": ObservationTermCfg(
            func=gm.foot_height,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES)}),
        "body_contact": ObservationTermCfg(func=gm.body_ground_contact),
        "head_force": ObservationTermCfg(func=gm.head_force_obs),
        "assist": ObservationTermCfg(func=mdp.generated_commands,
                                     params={"command_name": "getup"}),
        "pace": ObservationTermCfg(func=gm.pace_obs, params={"pace_s": pace_s}),
    })
    return {
        "actor": ObservationGroupCfg(terms=actor, concatenate_terms=True,
                                     enable_corruption=not play, history_length=history),
        "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True,
                                      enable_corruption=False),
    }


def make_rewards(pace_s: float = PACE_S, torque_budget: float = TORQUE_BUDGET,
                 speed_budget: float = SPEED_BUDGET, feet_high: float = FEET_HIGH,
                 head_scale: float = HEAD_SCALE) -> dict[str, RewardTermCfg]:
    """Weights are per second (mjlab multiplies every reward by dt = 0.02 s).

    Roughly, per second (once the pace schedule has reached standing): lying ~0.2,
    sitting up ~2, kneeling or squatting ~3.5, standing up to 8 (task 4.5 + standing
    3.5). Before that, being on or ahead of the schedule pays the full task reward and
    nothing more. Regularization at full weight costs ~0.5-1 while getting up calmly (it
    starts at 20 % of that, curriculum); a kip-up costs far more (speed and torque
    budgets, flight)."""
    groups = tuple((a.target_names_expr, vmax) for a, vmax in motor_groups())
    motors = tuple((a.target_names_expr, a.stiffness, a.damping, a.effort_limit)
                   for a, _ in motor_groups())
    rewards = {
        # -- task (always on; targets paced) ----------------------------------------------
        "head_up": RewardTermCfg(func=gm.head_up, weight=3.0, params={"pace_s": pace_s}),
        "pelvis_height": RewardTermCfg(func=gm.pelvis_height, weight=1.5,
                                       params={"pace_s": pace_s}),
        # -- once standing (and from pace_s on) ---------------------------------------------
        "stand_still": RewardTermCfg(func=gm.stand_still, weight=1.0, params={"pace_s": pace_s}),
        "stand_pose": RewardTermCfg(func=gm.stand_pose, weight=1.5, params={"pace_s": pace_s}),
        "feet_support": RewardTermCfg(func=gm.feet_support, weight=1.0,
                                      params={"pace_s": pace_s}),
        # -- style -----------------------------------------------------------------------
        "head_ground": RewardTermCfg(func=gm.head_ground_contact, weight=-2.0 * head_scale),
        "head_impact": RewardTermCfg(func=gm.head_impact, weight=-4.0 * head_scale),
        # Feet above the pelvis (m, summed over the feet): swinging the legs up for
        # momentum. 0.45 m (both legs over the back, lying face down) costs 3.6 per second.
        "feet_above_pelvis": RewardTermCfg(
            func=gm.feet_above_pelvis, weight=-feet_high,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
                    "margin": 0.05}),
        "joint_deviation": RewardTermCfg(
            func=gm.joint_deviation_beyond, weight=-2.0,
            params={"thresholds": {"waist_yaw_joint": 0.9, r".*_hip_yaw_joint": 0.8,
                                   r".*_hip_roll_joint": 0.6}}),
        # Feet sliding on the ground (sum of (m/s)^2 of the feet in contact).
        "feet_slip": RewardTermCfg(
            func=gm.feet_slip, weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES)}),
        # Nothing on the ground: a jump. 0.1 s in the air costs 0.5.
        "airborne": RewardTermCfg(func=gm.airborne, weight=-5.0),
        # -- regularization (REG_TERMS: scaled by the curriculum) --------------------------
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
        "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
        "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1.0e-3),
        "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-5),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
        # Speed budget: above speed_budget x the motor's no-load speed, (excess rad/s)^2.
        # 0.6: knees and hip roll 12 rad/s, hip pitch/yaw and waist 19, arms and ankles 22.
        "joint_overspeed": RewardTermCfg(
            func=gm.joint_overspeed, weight=-2.0,
            params={"groups": groups, "ratio": speed_budget, "power": 2.0}),
        # Torque budget: PD torque asked above torque_budget x the peak torque. A motor at
        # its peak costs 0.2 per second, one asked for twice its peak 7.2 per second.
        "torque_budget": RewardTermCfg(
            func=gm.torque_over_budget, weight=-5.0,
            params={"groups": motors, "budget": torque_budget}),
        # -- the simulation blew up (blown_up / nan terminations) --------------------------
        "terminated": RewardTermCfg(func=mdp.is_terminated, weight=-100.0),
    }
    if torque_budget <= 0.0:
        del rewards["torque_budget"]
    if feet_high <= 0.0:
        del rewards["feet_above_pelvis"]
    return rewards


def make_events(pool_path: str | None, posture_weights: dict[str, float] | None,
                randomize: bool, push: float = 0.0,
                pace_s: float = PACE_S, joint_dr: bool = JOINT_DR) -> dict[str, EventTermCfg]:
    events = {
        "reset_pose": EventTermCfg(
            func=gm.reset_to_fallen_pose, mode="reset",
            params={"pool_path": pool_path, "posture_weights": posture_weights,
                    "xy_range": 0.5}),
    }
    if push > 0.0:
        # Every 2-4 s, robots that are up get a push: keep the balance or get up again.
        events["push"] = EventTermCfg(
            func=gm.push_when_up, mode="interval", interval_range_s=(2.0, 4.0),
            params={"velocity_range": {"x": (-push, push), "y": (-push, push),
                                       "yaw": (-0.5, 0.5)},
                    "pace_s": pace_s})
    if not randomize:
        return events
    events.update({
        # One friction per env for every collision geom (feet, hands, body shells).
        "friction": EventTermCfg(
            mode="startup", func=dr.geom_friction,
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
                    "operation": "abs", "ranges": (0.35, 1.1), "shared_random": True}),
        "torso_com": EventTermCfg(
            mode="startup", func=dr.body_com_offset,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                    "operation": "add",
                    "ranges": {0: (-0.03, 0.03), 1: (-0.03, 0.03), 2: (-0.03, 0.03)}}),
        # Extra mass on the torso (head camera and its mount, battery): -1 .. +2 kg.
        "torso_payload": EventTermCfg(
            mode="startup", func=dr.body_mass,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                    "operation": "add", "ranges": (-1.0, 2.0)}),
        "encoder_bias": EventTermCfg(
            mode="startup", func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)}),
        # HoST: +-15 % stiffness and damping, +-10 % motor strength.
        "pd_gains": EventTermCfg(
            mode="startup", func=dr.pd_gains,
            params={"asset_cfg": SceneEntityCfg("robot"), "kp_range": (0.85, 1.15),
                    "kd_range": (0.85, 1.15)}),
        "motor_strength": EventTermCfg(
            mode="startup", func=dr.effort_limits,
            params={"asset_cfg": SceneEntityCfg("robot"), "effort_limit_range": (0.9, 1.1)}),
    })
    if joint_dr:
        # Real joints are not ideal: dry friction, damping, rotor inertia (as in the
        # walking fine-tune; Unitree's MuJoCo scene has 0.2 N m, 0.05 and 0.01).
        events.update({
            "joint_friction": EventTermCfg(
                mode="startup", func=dr.joint_friction,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                        "operation": "abs", "ranges": (0.0, 0.4)}),
            "joint_damping": EventTermCfg(
                mode="startup", func=dr.joint_damping,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                        "operation": "abs", "ranges": (0.0, 0.1)}),
            "joint_armature": EventTermCfg(
                mode="startup", func=dr.joint_armature,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                        "operation": "scale", "ranges": (0.8, 1.25)}),
        })
    return events


def make_getup_env_cfg(play: bool = False, num_envs: int | None = None,
                       realistic: bool = True, pool_path: str | Path | None = None,
                       posture_weights: dict[str, float] | None = None,
                       action_scale_factor: float = 0.5, history: int = HISTORY,
                       randomize: bool = True, assist_force: float | None = None,
                       pace_s: float = PACE_S, torque_budget: float = TORQUE_BUDGET,
                       speed_budget: float = SPEED_BUDGET,
                       push: float = 0.0, feet_high: float = FEET_HIGH,
                       head_scale: float = HEAD_SCALE,
                       joint_dr: bool = JOINT_DR) -> ManagerBasedRlEnvCfg:
    """play=True: no observation noise, no curriculum, no assist force, 4 envs.
    realistic: DC motor torque-speed curve (robot.py). randomize: friction, torso COM and
    mass, encoder bias, PD gains, motor strength (once per env, at startup).
    pace_s, torque_budget, speed_budget: how fast it may get up (make_rewards).
    push: pushes once up (m/s, 0 = none; training uses PUSH).
    feet_high, head_scale: style costs (make_rewards). joint_dr: joint friction, damping
    and armature randomized (only with randomize)."""
    rewards = make_rewards(pace_s, torque_budget, speed_budget, feet_high, head_scale)
    getup = GetUpCommandCfg(reg_terms={n: rewards[n].weight for n in REG_TERMS if n in rewards})
    if play:
        getup.curriculum = False
        getup.assist_force = 0.0
        getup.reg_start = 1.0
    if assist_force is not None:
        getup.assist_force = float(assist_force)
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_robot_cfg(realistic)},
            sensors=make_sensors(),
            num_envs=num_envs if num_envs is not None else (4 if play else 2048),
            env_spacing=2.5,
            extent=2.0,
        ),
        observations=make_observations(play, history, pace_s),
        actions={"joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",),
            scale=action_scale(action_scale_factor), use_default_offset=True)},
        commands={"getup": getup},
        events=make_events(str(pool_path) if pool_path else None, posture_weights, randomize,
                           push, pace_s, joint_dr),
        rewards=rewards,
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "blown_up": TerminationTermCfg(func=gm.blown_up),
            "nan": TerminationTermCfg(func=mdp.nan_detection),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot",
            body_name="pelvis", distance=3.5, elevation=-15.0, azimuth=135.0),
        # Lying on the ground means many contacts (and with friction, 4 constraint rows
        # each instead of 1): more room than the walking task (nconmax None, njmax 300).
        sim=SimulationCfg(
            nconmax=128, njmax=600, contact_sensor_maxmatch=64,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4,
        episode_length_s=EPISODE_S,
    )
    return cfg
