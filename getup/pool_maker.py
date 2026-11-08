"""Make the fallen-pose pool from the training scene (needs mjlab, runs on the CPU).

The scene is built exactly as in training (same robot model, collisions with friction,
ground plane, solver settings), but with MuJoCo's built-in position actuators instead of
the DC motor model, so that plain MuJoCo can run it: the drops only need "hold this angle"
or "go limp", which these do the same way.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np

from .poses import (DEFAULT_POOL, DROPS, MODES, POSTURES, DropCfg, FallenPoses,
                    generate_parallel, index_robot, resolve_regex_values, save_pool)
from .robot import HEAD_SHAPE


def build_cpu_model():
    """(MjModel of the training scene with position actuators, RobotIndex, walking pose)."""
    from mjlab.scene import Scene

    from .getup_env_cfg import make_getup_env_cfg

    cfg = make_getup_env_cfg(num_envs=1, realistic=False, randomize=False)
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    ix = index_robot(model)
    joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
    default_q = np.array(resolve_regex_values(joint_pos, ix.joint_names, default=0.0),
                         dtype=float)
    return model, ix, default_q


def describe(pool: FallenPoses) -> list[str]:
    n = len(pool)
    out = [f"{n} poses (from {pool.meta.get('drops_tried', '?')} drops)"]
    c = Counter(int(p) for p in pool.posture)
    out.append("  lying: " + ", ".join(f"{POSTURES[k]} {100 * v / n:.0f} %"
                                       for k, v in sorted(c.items())))
    c = Counter(int(p) for p in pool.drop)
    out.append("  released as: " + ", ".join(f"{DROPS[k]} {100 * v / n:.0f} %"
                                             for k, v in sorted(c.items())))
    c = Counter(int(p) for p in pool.mode)
    out.append("  motors: " + ", ".join(f"{MODES[k]} {100 * v / n:.0f} %"
                                        for k, v in sorted(c.items())))
    z = pool.root_pos[:, 2]
    out.append(f"  head collision: {pool.meta.get('head', 'the 6 cm sphere (made before v5)')}")
    out.append(f"  pelvis height at rest: median {np.median(z):.2f} m, "
               f"p10 {np.quantile(z, 0.1):.2f}, p90 {np.quantile(z, 0.9):.2f}; settle time "
               f"median {np.median(pool.settle_time):.1f} s")
    return out


def make_pool_file(path: str | Path = DEFAULT_POOL, n: int = 6000, workers: int = 4,
                   seed: int = 0, cfg: DropCfg | None = None) -> FallenPoses:
    t0 = time.time()
    model, ix, default_q = build_cpu_model()
    print(f"[INFO] dropping the robot {n} times ({workers} processes, timestep "
          f"{model.opt.timestep * 1000:.0f} ms) ...", flush=True)
    pool = generate_parallel(model, ix, default_q, n, workers, cfg, seed)
    pool.meta["seconds"] = round(time.time() - t0, 1)
    pool.meta["head"] = HEAD_SHAPE           # the head the poses rested with
    save_pool(path, pool)
    print(f"[INFO] saved {path} in {time.time() - t0:.0f} s")
    for line in describe(pool):
        print("[INFO] " + line)
    return pool
