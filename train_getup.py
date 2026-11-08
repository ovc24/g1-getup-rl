#!/usr/bin/env python3
"""Train the G1-23DoF to get up from the ground (from scratch, PPO).

Run it with the unitree_rl_mjlab environment activated, from the unitree_rl_mjlab folder
(logs in ./logs/rsl_rl/g1_23dof_getup/<date>):

    python <this folder>/train_getup.py                        # 2048 envs, 6000 iterations
    python <this folder>/train_getup.py --num-envs 4096        # if the GPU has the memory

Smoke test first (a few minutes): add --num-envs 64 --iterations 5.

How fast it may get up: --pace 3 (default; the task targets go from lying to standing in
3 s and standing earlier earns nothing more), --torque-budget 0.8 (PD torque asked above
80 % of the peak costs), --speed-budget 0.6 (joint speed above 60 % of the no-load speed
costs). --pace 0 --torque-budget 0 --speed-budget 0.9 is nearly the first version's
setting, which got up in about 0.5 s, joints near their no-load speed.

The first run makes the fallen-pose pool (data/fallen_poses.npz, a few minutes on the
CPU) if it is not there yet. Watch in tensorboard (logs/rsl_rl/g1_23dof_getup):
  Metrics/getup/success      share of episodes that end standing (>= 1 s)
  Metrics/getup/assist_N     assist force (200 N -> 0 as the success rate goes up)
  Metrics/getup/reg_scale    regularization weight (0.2 -> 1)
  Metrics/getup/time_to_stand_s   should settle near the pace (2-3 s), not below 1 s
  Metrics/getup/airborne_s, head_ground_s, head_peak_N   ~0, ~0, well below 100 N

Style (v5 defaults): --feet-high 4 (feet above the pelvis cost: no legs swung up for
momentum), --head-scale 2 (head on the ground and head hits cost twice the v4 amount),
joint friction/damping/armature randomized (--no-joint-dr: off). v4 had 0, 1 and off.
From v5 the head collision geom also has the real head's size (getup/robot.py): make the
fallen-pose pool again with this code (it warns if the pool was made with the old head).

v5, a fine-tune of a v4 run that also starts after limp collapses (a pool with "collapse"
releases, see make_fallen_poses.py):
    python <this folder>/train_getup.py --init-checkpoint <v4 run>/model_5999.pt \
        --assist 0 --assist-start 0 --reg-scale-start 1 --iterations 1500 \
        --pool <this folder>/data/fallen_poses_v5.npz --max-std 0.5

Every run also gets params/run_info.json (date, command, git commit of this folder and of
unitree_rl_mjlab and whether they had uncommitted changes, starting checkpoint, md5 of the
pool, also copied there) and
params/code/ (this script and getup/ as they were), next to params/env.yaml and
agent.yaml (the whole configuration) and getup_options.json.

Continue a stopped run from its model_<N>.pt as a new run: --init-checkpoint
<run>/model_<N>.pt plus the curriculum values it had (--assist-start, --reg-scale-start;
the last Metrics/getup/assist_N and reg_scale in tensorboard). --resume <ckpt> continues
exactly (also optimizer and noise), but the curriculum starts again from the values given.
"""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # for "getup"

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.os import dump_yaml  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

import src.tasks  # noqa: E402,F401  (unitree_rl_mjlab)
from src.tasks.velocity.config.g1_23dof.rl_cfg import unitree_g1_23dof_ppo_runner_cfg  # noqa: E402
from src.tasks.velocity.rl import VelocityOnPolicyRunner  # noqa: E402

from getup.getup_env_cfg import (FEET_HIGH, HEAD_SCALE, PACE_S, PUSH,  # noqa: E402
                                 SPEED_BUDGET, TORQUE_BUDGET, make_getup_env_cfg)
from getup.poses import DEFAULT_POOL, POSTURES  # noqa: E402
from getup.provenance import save_provenance  # noqa: E402
from getup.run_options import describe_pace, describe_style, parse_postures  # noqa: E402
from getup.train_utils import install_std_limit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--iterations", type=int, default=6000)
    ap.add_argument("--pool", default=str(DEFAULT_POOL), help="fallen-pose pool (.npz)")
    ap.add_argument("--postures", default=None,
                    help="which starting postures, e.g. 'supine,prone' or 'supine=2,prone=1' "
                         f"(names: {', '.join(POSTURES)}; default: supine 0.3, prone 0.3, "
                         "sides 0.12 each, upright 0.08, other 0.08)")
    ap.add_argument("--no-realistic", action="store_true",
                    help="no DC motor torque-speed curve (full torque at any speed)")
    ap.add_argument("--no-symmetry", action="store_true",
                    help="no left-right mirror augmentation")
    ap.add_argument("--mirror-loss", type=float, default=0.0,
                    help="with symmetry: extra loss pulling towards mirror-symmetric actions")
    ap.add_argument("--action-scale", type=float, default=0.5,
                    help="action scale factor: rad per unit action = factor x peak torque / "
                         "stiffness (walking: 0.25)")
    ap.add_argument("--pace", type=float, default=PACE_S,
                    help="seconds over which the task targets go from lying to standing; "
                         "standing earlier earns nothing more (0 = no pace)")
    ap.add_argument("--torque-budget", type=float, default=TORQUE_BUDGET,
                    help="PD torque asked above this share of the peak torque costs (0 = off)")
    ap.add_argument("--speed-budget", type=float, default=SPEED_BUDGET,
                    help="joint speed above this share of the motor's no-load speed costs")
    ap.add_argument("--feet-high", type=float, default=FEET_HIGH,
                    help="cost of the feet above the pelvis, per m per second (0 = off)")
    ap.add_argument("--head-scale", type=float, default=HEAD_SCALE,
                    help="factor on the head-on-the-ground and head-hit costs")
    ap.add_argument("--no-joint-dr", action="store_true",
                    help="no joint friction/damping/armature randomization")
    ap.add_argument("--push", type=float, default=PUSH,
                    help="pushes once up: pelvis velocity change up to +-this m/s every 2-4 s "
                         "(0 = none)")
    ap.add_argument("--assist", type=float, default=200.0,
                    help="initial assist force on the torso (N); 0 = none")
    ap.add_argument("--assist-start", type=float, default=None,
                    help="start the assist force at this level (continuing a run)")
    ap.add_argument("--reg-scale-start", type=float, default=None,
                    help="start the regularization scale at this value (continuing a run)")
    ap.add_argument("--init-checkpoint", default=None,
                    help="start from this get-up checkpoint (new run, same observations)")
    ap.add_argument("--resume", default=None,
                    help="get-up checkpoint to continue exactly (also optimizer and noise)")
    ap.add_argument("--init-std", type=float, default=0.8, help="initial action noise")
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--max-std", type=float, default=0.0,
                    help="cap on the action noise of every joint (<= 0: none)")
    ap.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    pool = Path(args.pool).expanduser()
    if not pool.exists():
        print(f"[INFO] No fallen-pose pool at {pool}: making it (a few minutes, CPU) ...")
        subprocess.run([sys.executable, str(HERE / "make_fallen_poses.py"), "--out", str(pool)],
                       check=True)

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("MUJOCO_GL", "egl")

    pace = {"pace_s": args.pace, "torque_budget": args.torque_budget,
            "speed_budget": args.speed_budget}
    style = {"feet_high": args.feet_high, "head_scale": args.head_scale,
             "joint_dr": not args.no_joint_dr}
    env_cfg = make_getup_env_cfg(num_envs=args.num_envs, realistic=not args.no_realistic,
                                 pool_path=pool, posture_weights=parse_postures(args.postures, ap),
                                 action_scale_factor=args.action_scale,
                                 assist_force=args.assist, push=args.push, **pace, **style)
    g = env_cfg.commands["getup"]
    g.assist_start = args.assist_start
    g.reg_scale_start = args.reg_scale_start
    env_cfg.seed = args.seed

    agent_cfg = unitree_g1_23dof_ppo_runner_cfg()
    agent_cfg.experiment_name = "g1_23dof_getup"
    agent_cfg.max_iterations = args.iterations
    agent_cfg.logger = args.logger
    agent_cfg.seed = args.seed
    agent_cfg.upload_model = args.logger == "wandb"
    agent_cfg.algorithm.entropy_coef = args.entropy_coef
    if agent_cfg.actor.distribution_cfg is not None:
        agent_cfg.actor.distribution_cfg["init_std"] = args.init_std

    log_dir = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name
               / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] device={device}  envs={args.num_envs}  realistic={not args.no_realistic}  "
          f"symmetry={not args.no_symmetry}  action_scale={args.action_scale}  "
          f"assist={args.assist} N  pool={pool}  log dir: {log_dir}")
    print(f"[INFO] {describe_pace(pace)}, pushes once up +-{args.push} m/s")
    print(f"[INFO] {describe_style(style)}; postures {args.postures or 'default'}")

    agent_dict = asdict(agent_cfg)
    if not args.no_symmetry:
        agent_dict["algorithm"]["symmetry_cfg"] = {
            "use_data_augmentation": True,
            "use_mirror_loss": args.mirror_loss > 0.0,
            "data_augmentation_func": "getup.symmetry:g1_getup_symmetry",
            "mirror_loss_coeff": args.mirror_loss,
        }
    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", agent_dict)
    # What eval_getup.py / play_getup.py need to rebuild the same env for this run.
    (log_dir / "params" / "getup_options.json").write_text(json.dumps({
        "version": 4, "action_scale": args.action_scale, "realistic": not args.no_realistic,
        "postures": args.postures, "pool": str(pool), "history": env_cfg.observations[
            "actor"].history_length, "push": args.push, **pace, **style}, indent=1))
    # How to reproduce it: the command, the git state, a copy of the code and of the pool.
    save_provenance(log_dir, HERE, [Path(__file__), HERE / "getup"],
                    extra={"init_checkpoint": args.init_checkpoint, "resume": args.resume},
                    data=[pool])

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = VelocityOnPolicyRunner(env, agent_dict, str(log_dir), device)
    if args.resume:
        print(f"[INFO] Resuming from {args.resume}")
        runner.load(str(Path(args.resume).expanduser()))
    elif args.init_checkpoint:
        print(f"[INFO] Starting from {args.init_checkpoint}")
        ckpt = str(Path(args.init_checkpoint).expanduser())
        try:
            runner.load(ckpt, load_cfg={"actor": True, "critic": True}, strict=True,
                        map_location=device)
        except RuntimeError as e:          # e.g. a run from before the critic saw the pace
            print(f"[WARNING] the critic of that checkpoint does not fit ({str(e)[:120]}...): "
                  "loading only the policy, the critic starts new")
            runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    if args.max_std > 0:
        before = install_std_limit(runner, args.max_std)
        if before is None:
            print("[WARNING] no std parameter found: --max-std ignored")
        else:
            print(f"[INFO] Action noise capped at std {args.max_std} (up to {before:.2f} "
                  "before the cap)")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
