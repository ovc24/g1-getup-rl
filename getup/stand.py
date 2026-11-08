"""When the robot counts as standing, and the default mix of starting postures (numpy-free
constants shared by the training code and sim2sim_getup.py)."""
from __future__ import annotations

from dataclasses import dataclass

HEAD_OFFSET = (0.0, 0.0, 0.43)      # a point in the upper head, torso_link frame (centre of
                                    # Unitree's head sphere; top of the head capsule's axis)

DEFAULT_POSTURE_WEIGHTS = {"supine": 0.3, "prone": 0.3, "left_side": 0.12,
                           "right_side": 0.12, "upright": 0.08, "other": 0.08}


@dataclass(frozen=True)
class StandCfg:
    """When the robot counts as standing (head = centre of the head sphere; the walking
    pose has the pelvis at ~0.76 m and the head at ~1.23 m)."""
    pelvis_min: float = 0.62
    head_min: float = 1.00
    up_min: float = 0.85          # -gravity_z in the pelvis frame = cos(tilt): 32 deg
    hold_s: float = 1.0           # standing this long = got up


STAND = StandCfg()
