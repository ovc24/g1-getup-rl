#!/usr/bin/env python3
"""Measure a get-up policy: one 10 s episode per env, headless, no assist force.

    cd <unitree_rl_mjlab> && source .venv/bin/activate
    python <this folder>/eval_getup.py \
        --checkpoint logs/rsl_rl/g1_23dof_getup/<run>/model_<N>.pt

By default: mean action (no exploration noise, as on the real robot), sensor noise and the
training's randomization on, 600 envs (= episodes), every starting posture of the pool
(as in training), the FIRST episode of each env. --clean: no sensor noise and no
randomization (the nominal robot). The env options of the run (action scale, motors,
pace, style) are read from <run>/params/getup_options.json.

To compare two runs under the same conditions, give both the same --pool and --joint-dr
(e.g. a v4 run, made without joint friction randomization, and a v5 run, made with it):
    eval_getup.py --checkpoint <v4>/model_5999.pt --pool data/fallen_poses_v5.npz --joint-dr on

Prints, per starting posture: success (standing >= 1 s when the episode ends), stood up at
some point, time to stand, fell again after standing, head (camera) on the ground and the
hardest head hit. Then how violent the movement is: peak joint speeds against the
motors' no-load speeds, peak pelvis speeds, time with nothing on the ground (jumps), and
how often the motors are at their peak torque, while getting up and while standing,
with the joints that are there most.
"""
import argparse
import math
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

import src.tasks  # noqa: E402,F401
from src.tasks.velocity.config.g1_23dof.rl_cfg import unitree_g1_23dof_ppo_runner_cfg  # noqa: E402
from src.tasks.velocity.rl import VelocityOnPolicyRunner  # noqa: E402

from getup.getup_env_cfg import make_getup_env_cfg  # noqa: E402
from getup.getup_mdp import getup_state  # noqa: E402
from getup.poses import (DEFAULT_POOL, DROPS, POSTURES, load_pool, released_mask,  # noqa: E402
                         save_pool, subset)
from getup.robot import FOOT_BODIES, motor_groups  # noqa: E402
from getup.run_options import (describe_pace, describe_style, load_run_options,  # noqa: E402
                               pace_options, parse_postures, resolve_pool, style_options)

GROUP_LABELS = ("arms", "hip pitch/yaw + waist", "knee + hip roll", "ankles")
PHASES = ("getting up", "standing")


def pct(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float(), q)) if x.numel() else float("nan")


def top_joints(share: torch.Tensor, names: list[str], k: int = 3, floor: float = 0.01) -> str:
    order = torch.argsort(share, descending=True)[:k]
    items = [f"{names[int(i)]} {100 * float(share[i]):.0f} %" for i in order
             if float(share[i]) >= floor]
    return ", ".join(items) if items else "none above 1 %"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", "--checkpoint-file", dest="checkpoint", required=True)
    ap.add_argument("--num-envs", type=int, default=600, help="= number of episodes")
    ap.add_argument("--postures", default=None,
                    help="starting postures, e.g. 'prone' or 'supine=1,prone=1' "
                         f"(names: {', '.join(POSTURES)}; default: as in training)")
    ap.add_argument("--pool", default=None, help="fallen-pose pool (default: the run's)")
    ap.add_argument("--released", default=None,
                    help="only poses released this way, e.g. 'collapse' (names: "
                         + ", ".join(DROPS) + ")")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions with the policy's noise, as in training")
    ap.add_argument("--clean", action="store_true",
                    help="no sensor noise and no randomization (nominal robot)")
    ap.add_argument("--assist", type=float, default=0.0,
                    help="assist force (N); the real robot has none")
    ap.add_argument("--push", type=float, default=0.0,
                    help="pushes once up, +-m/s every 2-4 s (training: see the run's options); "
                         "0 = none")
    ap.add_argument("--joint-dr", choices=("run", "on", "off"), default="run",
                    help="joint friction/damping/armature randomization: as in the run's "
                         "training (default), or on/off")
    ap.add_argument("--time", type=float, default=10.0, help="episode length (s)")
    ap.add_argument("--action-scale", type=float, default=None)
    ap.add_argument("--no-realistic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    opts = load_run_options(args.checkpoint)
    scale = args.action_scale if args.action_scale is not None else opts.get("action_scale", 0.5)
    realistic = (not args.no_realistic) and opts.get("realistic", True)
    pool = resolve_pool(args.pool or opts.get("pool") or DEFAULT_POOL)
    if args.released:                       # a pool of only those poses, in a temp folder
        full = load_pool(pool)
        try:
            sub = subset(full, released_mask(full, args.released),
                         note=f"released as {args.released}")
        except ValueError as e:
            ap.error(f"{pool.name}: {e}")
        tag = args.released.replace(",", "_").replace(" ", "")
        pool = Path(tempfile.mkdtemp(prefix="getup_eval_")) / f"{pool.stem}_{tag}.npz"
        save_pool(pool, sub)
        print(f"[INFO] {len(sub)} of {len(full)} poses released as {args.released}")
    pace = pace_options(opts)
    style = style_options(opts)
    if args.joint_dr != "run":
        style["joint_dr"] = args.joint_dr == "on"

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = make_getup_env_cfg(num_envs=args.num_envs, realistic=realistic, pool_path=pool,
                                 posture_weights=parse_postures(args.postures, ap),
                                 action_scale_factor=scale,
                                 history=opts.get("history", 5),
                                 randomize=not args.clean, assist_force=args.assist,
                                 push=args.push, **pace, **style)
    g = env_cfg.commands["getup"]
    g.curriculum = False
    g.reg_start = 1.0
    if args.clean:
        env_cfg.observations["actor"].enable_corruption = False
    env_cfg.episode_length_s = args.time
    env_cfg.seed = args.seed
    agent_cfg = unitree_g1_23dof_ppo_runner_cfg()

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = VelocityOnPolicyRunner(venv, asdict(agent_cfg), device=device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    std = None
    dist_mod = getattr(policy, "distribution", None)
    for attr in ("std_param", "log_std_param"):
        p = getattr(dist_mod, attr, None) if dist_mod is not None else None
        if p is not None:
            p = p.detach()
            std = float(p.mean()) if attr == "std_param" else float(torch.exp(p).mean())

    term = env.command_manager.get_term("getup")
    reset_ev = env.event_manager.get_term_cfg("reset_pose").func
    robot = term.robot
    n = args.num_envs
    hold = g.hold_s

    venv.reset()
    obs = venv.get_observations()
    posture0 = reset_ev.env_posture.clone()

    done = torch.zeros(n, dtype=torch.bool, device=device)
    last_stand = torch.zeros(n, device=device)
    last_first = torch.full((n,), -1.0, device=device)
    last_head_t = torch.zeros(n, device=device)
    last_head_pk = torch.zeros(n, device=device)
    last_air = torch.zeros(n, device=device)
    last_pelvis = torch.zeros(n, device=device)
    last_pose_err = torch.zeros(n, device=device)
    peak_vz = torch.zeros(n, device=device)
    peak_w = torch.zeros(n, device=device)
    peak_feet = torch.full((n,), -10.0, device=device)
    feet_ids, _ = robot.find_bodies(FOOT_BODIES)
    blown = torch.zeros(n, dtype=torch.bool, device=device)
    nj = robot.data.joint_vel.shape[1]
    peak_qd = torch.zeros(n, nj, device=device)
    peak_tau = torch.zeros(nj, device=device)
    vmax = torch.full((nj,), float("inf"), device=device)
    group_of = torch.full((nj,), -1, dtype=torch.long, device=device)
    for gi, (a, v) in enumerate(motor_groups()):
        ids, _ = robot.find_joints(a.target_names_expr)
        peak_tau[ids] = float(a.effort_limit)
        vmax[ids] = v
        group_of[ids] = gi
    # Actuators are ordered by motor group, joints by the model: map one to the other.
    jnames = list(robot.joint_names)
    act_joint = torch.tensor([jnames.index(a) for a in robot.actuator_names], device=device)
    # Steps at the peak torque, [getting up, standing]: per motor group (any of its joints)
    # and per joint.
    sat_group = torch.zeros(2, len(GROUP_LABELS), device=device)
    sat_joint = torch.zeros(2, nj, device=device)
    phase_steps = torch.zeros(2, device=device)
    tm = env.termination_manager

    steps = int(math.ceil(args.time / env.step_dt)) + 5
    for _ in range(steps):
        with torch.no_grad():
            actions = policy(obs, stochastic_output=True) if args.stochastic else policy(obs)
        obs, _, dones, _ = venv.step(actions)
        ended = dones.bool() & ~done
        blown |= ended & (tm.get_term("blown_up") | tm.get_term("nan"))
        done |= ended
        live = ~done
        if not bool(live.any()):
            break
        s = getup_state(env)
        last_stand = torch.where(live, term.stand_time, last_stand)
        last_first = torch.where(live, term.first_stand, last_first)
        last_head_t = torch.where(live, term.head_time, last_head_t)
        last_head_pk = torch.where(live, term.head_peak, last_head_pk)
        last_air = torch.where(live, term.air_time, last_air)
        last_pelvis = torch.where(live, s.pelvis_z, last_pelvis)
        err = (robot.data.joint_pos - robot.data.default_joint_pos).square().mean(dim=1).sqrt()
        last_pose_err = torch.where(live, err, last_pose_err)
        qd = robot.data.joint_vel.abs()
        peak_qd = torch.where(live.unsqueeze(1), torch.maximum(peak_qd, qd), peak_qd)
        vz = robot.data.root_link_lin_vel_w[:, 2].abs()
        w = torch.linalg.norm(robot.data.root_link_ang_vel_w, dim=1)
        peak_vz = torch.where(live, torch.maximum(peak_vz, vz), peak_vz)
        peak_w = torch.where(live, torch.maximum(peak_w, w), peak_w)
        above = (robot.data.body_link_pos_w[:, feet_ids, 2].amax(dim=1)
                 - robot.data.root_link_pos_w[:, 2])
        peak_feet = torch.where(live, torch.maximum(peak_feet, above), peak_feet)
        tau = torch.zeros_like(qd)
        tau[:, act_joint] = robot.data.actuator_force.abs()
        at_peak = tau >= 0.95 * peak_tau
        for ph, sel in enumerate((live & ~s.standing, live & s.standing)):
            sat_joint[ph] += (at_peak & sel.unsqueeze(1)).float().sum(dim=0)
            for gi in range(len(GROUP_LABELS)):
                sat_group[ph, gi] += (at_peak[:, group_of == gi].any(dim=1) & sel).float().sum()
            phase_steps[ph] += sel.float().sum()

    success = last_stand >= hold
    stood = last_first >= 0.0
    fell_again = stood & ~success

    print()
    print(f"=== {Path(args.checkpoint).name}  |  {n} episodes of {args.time:.0f} s  |  "
          f"{'realistic motors' if realistic else 'ideal motors'}, action scale {scale} ===")
    print("    " + ", ".join((
        "WITH policy noise" if args.stochastic else "mean action",
        "clean (no sensor noise, no randomization)" if args.clean
        else "sensor noise + randomization",
        f"assist {args.assist:.0f} N", f"pushes {args.push} m/s", f"seed {args.seed}",
        f"pool {pool.name}")))
    print(f"    trained with {describe_pace(pace)}")
    print(f"    {describe_style(style)}")
    if std is not None:
        print(f"policy action std (training noise): {std:.2f}")
    if not bool(done.all()):
        print(f"WARNING: {int((~done).sum())} envs did not finish the episode")
    if bool(blown.any()):
        print(f"WARNING: {int(blown.sum())} episodes ended because the simulation blew up")
    print()
    hdr = (f"{'start':11s} {'n':>4s} {'success':>8s} {'stood up':>9s} {'t stand':>8s} "
           f"{'p90':>5s} {'fell again':>10s} {'head s':>7s} {'head N p90':>10s}")
    print(hdr)
    rows = [(name, posture0 == i) for i, name in enumerate(POSTURES)]
    rows.append(("all", torch.ones_like(success)))
    for name, sel in rows:
        k = int(sel.sum())
        if k == 0:
            continue
        ts = last_first[sel & stood]
        print(f"{name:11s} {k:4d} {100 * float(success[sel].float().mean()):7.1f}% "
              f"{100 * float(stood[sel].float().mean()):8.1f}% {pct(ts, 0.5):7.1f}s "
              f"{pct(ts, 0.9):4.1f}s {100 * float(fell_again[sel].float().mean()):9.1f}% "
              f"{float(last_head_t[sel].mean()):6.2f}s {pct(last_head_pk[sel], 0.9):9.0f}N")
    print("  success = standing (pelvis > 0.62 m, head > 1.0 m, < 32 deg tilt, only the feet "
          "on the ground) for >= 1 s at the end; t stand = when the first 1 s stand began")
    if bool(success.any()):
        print(f"at the end, standing episodes: pelvis {pct(last_pelvis[success], 0.5):.2f} m "
              f"(median), joints {math.degrees(pct(last_pose_err[success], 0.5)):.1f} deg RMS from "
              f"the walking pose (median; p90 {math.degrees(pct(last_pose_err[success], 0.9)):.1f})")
    print()
    print("how violent (a calm get-up: joints well below their no-load speed, no flight, "
          "motors rarely at their peak)")
    print("peak joint speed per episode vs motor no-load speed (real motor: 0 torque there)")
    for gi, label in enumerate(GROUP_LABELS):
        ids = (group_of == gi).nonzero().flatten()
        v = float(vmax[ids[0]])
        peak = peak_qd[:, ids].max(dim=1).values
        print(f"  {label:22s} no-load {v:4.0f} rad/s   peak median {pct(peak, 0.5):5.1f}   "
              f"p90 {pct(peak, 0.9):5.1f}   episodes above 90 %: "
              f"{100 * float((peak > 0.9 * v).float().mean()):5.1f} %")
    print(f"pelvis per episode: peak vertical speed median {pct(peak_vz, 0.5):.2f} m/s "
          f"(p90 {pct(peak_vz, 0.9):.2f}), peak angular speed median {pct(peak_w, 0.5):.1f} "
          f"rad/s (p90 {pct(peak_w, 0.9):.1f})")
    high = peak_feet > 0.2
    print(f"feet above the pelvis (legs swung up), highest per episode: median "
          f"{pct(peak_feet, 0.5):+.2f} m (p90 {pct(peak_feet, 0.9):+.2f}); a foot more than "
          f"0.2 m above the pelvis in {100 * float(high.float().mean()):.1f} % of the episodes")
    air_any = last_air > 0.0
    line = (f"nothing on the ground (after the first 0.1 s): "
            f"{100 * float(air_any.float().mean()):.1f} % of the episodes")
    if bool(air_any.any()):
        line += (f"; in those, median {pct(last_air[air_any], 0.5):.2f} s, "
                 f"max {float(last_air.max()):.2f} s")
    print(line)
    print("motors at >= 95 % of their peak torque, share of the steps "
          f"(getting up: {int(phase_steps[0])} steps, standing: {int(phase_steps[1])})")
    print(f"  {'':22s} {PHASES[0]:>11s} {PHASES[1]:>10s}")
    for gi, label in enumerate(GROUP_LABELS):
        a, b = (100 * float(sat_group[ph, gi] / phase_steps[ph].clamp(min=1)) for ph in (0, 1))
        print(f"  {label:22s} {a:10.1f} % {b:8.1f} %")
    for ph in (0, 1):
        share = sat_joint[ph] / phase_steps[ph].clamp(min=1)
        print(f"  most at the peak while {PHASES[ph]}: {top_joints(share, jnames)}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
