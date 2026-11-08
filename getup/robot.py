"""The Unitree G1-23DoF for the get-up task.

Same model, gains, armature and peak torques as Unitree's walking task (unitree_rl_mjlab,
``g1_23dof_constants.py``), with three changes:

1. Friction on every collision geom. Unitree's walking config gives the non-foot geoms
   ``condim=1``, i.e. frictionless contacts: right for walking (only the feet touch the
   ground), wrong for getting up, where hands, knees, back and hips push on the ground.
   Here every collision geom has ``condim=3`` (normal force + 2D friction) and
   ``priority=1``, so its own friction is used against the ground plane.
2. Optionally (``realistic=True``) a DC motor torque-speed curve: the torque available
   falls linearly from the peak at standstill to zero at the motor's no-load speed, as on
   the real motors (Unitree's no-load speeds: 20 rad/s knee and hip roll, 32 rad/s hip
   pitch/yaw and waist, 37 rad/s ankles and arms).
3. (from v5, ``real_head=True``) The head collision geom has the size of the real head.
   Unitree's model collides with a sphere of 6 cm radius 43 cm above the torso link; the
   real head (``head_link`` mesh, with the depth camera in the face) is 14 x 11 x 17 cm
   above its base and reaches up to 5.6 cm outside that sphere, the face included, so a
   policy trained with the sphere could hit the ground with the camera without ever
   feeling it (in Unitree's MuJoCo model, which collides with the mesh: 700-900 N).
   ``HEAD_CAPSULE`` covers the mesh within 0.5 cm and reaches at most 1.4 cm beyond it.
"""
from __future__ import annotations

from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from src.assets.robots.unitree_g1 import g1_23dof_constants as g1c

FOOT_GEOMS = r"^(left|right)_foot[1-7]_collision$"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
TORSO_BODY = "torso_link"
HEAD_GEOM = "head_collision"           # Unitree: sphere r = 6 cm, 43 cm above the torso link
# The real head as a capsule (torso_link frame): axis x = 1 cm, z = 38 -> 43 cm, radius 7 cm,
# i.e. z 31-50 cm, x -6 .. +8 cm, y +-7 cm (the head_link mesh above its base: z 32-49,
# x -6.2 .. +7.8, y +-5.6).
HEAD_CAPSULE = {"pos": (0.01, 0.0, 0.405), "radius": 0.07, "half_length": 0.025}
HEAD_SHAPE = "capsule r 0.07 m, axis x 0.01, z 0.38-0.43 (torso_link frame)"
HAND_GEOMS = ("left_hand_collision", "right_hand_collision")

# Bodies that carry collision geoms (pelvis, torso + head, and per side: hip, thigh,
# shin, foot, upper arm, forearm, hand). The ground contact sensor watches these.
CONTACT_BODIES = (
    "pelvis", "torso_link",
    r".*_hip_roll_link", r".*_hip_yaw_link", r".*_knee_link", r".*_ankle_roll_link",
    r".*_shoulder_yaw_link", r".*_elbow_link", r".*_wrist_roll_rubber_hand",
)

GETUP_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim=3,
    priority=1,
    # Feet: Unitree's 0.6. Body shells and rubber hands on grass: a guess, randomized
    # in training anyway (getup_env_cfg.py, event "friction").
    friction={FOOT_GEOMS: (0.6,), ".*_collision": (0.7,)},
)


def motor_groups() -> tuple[tuple[BuiltinPositionActuatorCfg, float], ...]:
    """Unitree's G1-23DoF actuator groups with their joint-side no-load speeds (rad/s)."""
    return (
        (g1c.G1_ACTUATOR_5020, g1c.ACTUATOR_5020.velocity_limit),          # arms, 37
        (g1c.G1_ACTUATOR_7520_14, g1c.ACTUATOR_7520_14.velocity_limit),    # hip p/y, waist, 32
        (g1c.G1_ACTUATOR_7520_22, g1c.ACTUATOR_7520_22.velocity_limit),    # knee, hip roll, 20
        (g1c.G1_ACTUATOR_ANKLE, g1c.ACTUATOR_5020.velocity_limit),         # 2 x 5020, ~1:1, 37
    )


def dc_motor_articulation(saturation_scale: float = 1.0) -> EntityArticulationInfoCfg:
    """Unitree's actuators (same gains, armature, peak torques) with a torque-speed curve.

    With saturation_scale > 1 the peak torque holds up to a "corner" speed and only then
    falls (less conservative)."""
    actuators = []
    for a, vmax in motor_groups():
        assert isinstance(a, BuiltinPositionActuatorCfg) and a.effort_limit is not None
        actuators.append(DcMotorActuatorCfg(
            target_names_expr=a.target_names_expr,
            stiffness=a.stiffness,
            damping=a.damping,
            effort_limit=a.effort_limit,
            saturation_effort=a.effort_limit * saturation_scale,
            velocity_limit=vmax,
            armature=a.armature,
            frictionloss=a.frictionloss,
        ))
    return EntityArticulationInfoCfg(
        actuators=tuple(actuators),
        soft_joint_pos_limit_factor=g1c.G1_23DOF_ARTICULATION.soft_joint_pos_limit_factor,
    )


def real_head_spec_fn(spec_fn):
    """A spec_fn that loads the model with spec_fn and makes its head collision geom
    HEAD_CAPSULE (same name, so the head contact sensor keeps working)."""
    def fn():
        import mujoco
        spec = spec_fn()
        heads = [g for g in spec.geoms if g.name == HEAD_GEOM]
        if len(heads) != 1:
            raise ValueError(f"expected one geom '{HEAD_GEOM}' in the robot model, "
                             f"found {len(heads)}")
        g = heads[0]
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size = [HEAD_CAPSULE["radius"], HEAD_CAPSULE["half_length"], 0.0]
        g.pos = list(HEAD_CAPSULE["pos"])
        g.quat = [1.0, 0.0, 0.0, 0.0]          # capsule axis = the torso's z axis
        return spec
    return fn


def get_robot_cfg(realistic: bool = True, real_head: bool = True) -> EntityCfg:
    """Unitree's G1-23DoF entity with get-up collisions (and the DC motor curve, and the
    real head size)."""
    cfg = g1c.get_g1_23dof_robot_cfg()
    cfg.collisions = (GETUP_COLLISION,)
    if realistic:
        cfg.articulation = dc_motor_articulation()
    if real_head:
        cfg.spec_fn = real_head_spec_fn(cfg.spec_fn)
    return cfg


def action_scale(factor: float = 0.5) -> dict[str, float]:
    """Joint position action scale: factor x peak torque / stiffness (rad per unit action).

    Unitree's walking task uses 0.25. Getting up needs the peak torques often (lifting the
    whole body with the knees or the arms), so the default here is 0.5: |action| = 2
    already asks for the peak torque at the default pose. The same numbers go into the
    deployment config, so any factor is fine for the real robot as long as both match."""
    scale: dict[str, float] = {}
    for a, _ in motor_groups():
        assert a.effort_limit is not None
        for name in a.target_names_expr:
            scale[name] = factor * a.effort_limit / a.stiffness
    return scale
