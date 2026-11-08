#!/usr/bin/env python3
"""Watch a get-up policy in the viewer (no assist force, no sensor noise).

    cd <unitree_rl_mjlab> && source .venv/bin/activate
    python <this folder>/play_getup.py \
        --checkpoint logs/rsl_rl/g1_23dof_getup/<run>/model_<N>.pt

Every 10 s each robot starts again from a new fallen pose. --postures prone shows only
falls on the front, etc. The env options of the run are read from
<run>/params/getup_options.json.

Before training, look at the starting poses with --zero (no policy: all actions 0, so the
motors pull the joints towards the walking pose while the robot lies there):

    python <this folder>/play_getup.py --zero
"""
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer  # noqa: E402

import src.tasks  # noqa: E402,F401
from src.tasks.velocity.config.g1_23dof.rl_cfg import unitree_g1_23dof_ppo_runner_cfg  # noqa: E402
from src.tasks.velocity.rl import VelocityOnPolicyRunner  # noqa: E402

from getup.getup_env_cfg import make_getup_env_cfg  # noqa: E402
from getup.poses import DEFAULT_POOL, POSTURES  # noqa: E402
from getup.run_options import (load_run_options, pace_options, parse_postures,  # noqa: E402
                               resolve_pool, style_options)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", "--checkpoint-file", dest="checkpoint", default=None)
    ap.add_argument("--zero", action="store_true",
                    help="no policy: zero actions (to look at the starting poses)")
    ap.add_argument("--postures", default=None,
                    help=f"starting postures, e.g. 'prone' (names: {', '.join(POSTURES)})")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--time", type=float, default=10.0, help="seconds per episode")
    ap.add_argument("--assist", type=float, default=0.0, help="assist force (N)")
    ap.add_argument("--viewer", choices=("viser", "native"), default="viser",
                    help="viser: in the browser (default); native: MuJoCo window")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if not args.checkpoint and not args.zero:
        ap.error("give --checkpoint (or --zero to watch the starting poses without a policy)")

    opts = load_run_options(args.checkpoint) if args.checkpoint else {}
    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = make_getup_env_cfg(play=True, num_envs=args.num_envs,
                                 realistic=opts.get("realistic", True),
                                 pool_path=resolve_pool(opts.get("pool") or DEFAULT_POOL),
                                 posture_weights=parse_postures(args.postures, ap),
                                 action_scale_factor=opts.get("action_scale", 0.5),
                                 history=opts.get("history", 5), assist_force=args.assist,
                                 **pace_options(opts), **style_options(opts))
    env_cfg.episode_length_s = args.time
    agent_cfg = unitree_g1_23dof_ppo_runner_cfg()

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args.checkpoint and not args.zero:
        runner = VelocityOnPolicyRunner(env, asdict(agent_cfg), device=device)
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                    map_location=device)
        policy = runner.get_inference_policy(device=device)
    else:
        zeros = torch.zeros(env.num_envs, env.num_actions, device=device)

        def policy(obs):
            return zeros
    if args.viewer == "viser":
        print("[INFO] Open the viser link printed below in the browser.")
    (NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer)(env, policy).run()
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
