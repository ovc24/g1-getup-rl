#!/usr/bin/env python3
"""Make the pool of fallen poses the get-up episodes start from (once, ~1-5 min, CPU).

Run it with the unitree_rl_mjlab environment activated, from the unitree_rl_mjlab folder:

    python <this folder>/make_fallen_poses.py                  # -> data/fallen_poses.npz
    python <this folder>/make_fallen_poses.py --n 10000 --workers 8

The robot is dropped many times in plain MuJoCo with the training scene (getup/poses.py
explains how) and every resting pose is kept. train_getup.py makes the pool by itself if
it is missing; run this to make a bigger one or to change the mix (--drops).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # for "getup"

from getup.pool_maker import make_pool_file  # noqa: E402
from getup.poses import DEFAULT_POOL, DROPS, DropCfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_POOL))
    ap.add_argument("--n", type=int, default=6000, help="number of poses")
    ap.add_argument("--workers", type=int, default=4, help="CPU processes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drops", default=None,
                    help=f"release mix, e.g. 'supine=1,prone=1' (names: {', '.join(DROPS)}; "
                         "default: supine 0.25, prone 0.25, sides 0.1 each, random 0.1, "
                         "standing 0.2)")
    args = ap.parse_args()

    cfg = DropCfg()
    if args.drops:
        w = {}
        for item in args.drops.split(","):
            k, v = item.split("=")
            if k.strip() not in DROPS:
                ap.error(f"unknown release '{k}' (names: {', '.join(DROPS)})")
            w[k.strip()] = float(v)
        cfg.drop_weights = w
    make_pool_file(Path(args.out).expanduser(), args.n, args.workers, args.seed, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
