#!/usr/bin/env python3
"""A scripted scenario in Unitree's MuJoCo model of the G1-23DoF, run with the same files
and the same state machine as g1_ctrl, and recorded as a video without a window: no lag,
every frame drawn, the state and the events written on the video.

    cd <unitree_rl_mjlab> && source .venv/bin/activate
    python <this folder>/sim2sim_scenario.py                                # default scenario
    python <this folder>/sim2sim_scenario.py --walk logs/rsl_rl/g1_23dof_velocity_robust/<run>
    python <this folder>/sim2sim_scenario.py --print-script > my_scenario.txt
    python <this folder>/sim2sim_scenario.py --script my_scenario.txt --out videos/

It loads what g1_ctrl loads (--deploy, default ./deploy, robot g1_23dof):
config/config.yaml (states and their settings), the walking policy
config/policy/velocity/<last>/ and the get-up policy config/policy/getup/<last>/
(exported/policy.onnx + params/deploy.yaml each; <last>: as g1_ctrl picks it). --walk and
--getup take another policy.onnx (a run folder or a file) with the same deploy.yaml: to
compare two walking policies without copying files. A policy folder (policy/<name> of this
repo, or any folder with exported/policy.onnx and params/deploy.yaml) brings its own
deploy.yaml.

The scenario: one event per line, `time event arguments` (# comments); --print-script
shows the default one, which goes through pushes from the four sides while standing and
walking, a push too strong (fall, then the automatic get-up), motors off (collapse, then
the get-up buttons) and a drop from the elastic band.

    walk               the walking policy (on the robot: FixStand, then R2+A)
    cmd VX VY WZ       joystick: m/s forward, m/s to the left, rad/s to the left
    push DIR N [S]     horizontal force on the torso for S s (default 0.1); DIR: forward,
                       backward, left, right (where the robot faces) or degrees (90 = left)
    limp               Passive: motors limp (L2+B)
    getup              the get-up buttons (R2+X), from Passive or Fallen
    band H             hang it from an elastic band, pelvis at about H m, motors limp
    release            band off (key 9): it drops
    end                stop (default: 10 s after the last event)

What happens by itself, as in g1_ctrl with deploy/install_getup.py: Velocity -> Fallen
when tilted more than 1 rad; Fallen -> GetUp once still (if auto_getup in config.yaml);
GetUp limp until still, then the get-up policy; standing still in the walking pose ->
Velocity; not up in time -> on_timeout.

Physics as sim2sim_getup.py: Unitree's scene (meshes), 2 ms steps, PD every 2 ms, the
motor torque-speed curve, no sensor noise (as g1_ctrl with Unitree's simulator). The robot
starts standing in the walking pose. Output: <out>/scenario_<date>.mp4 (960x540, 25 fps)
and .txt (every event and transition, the get-up attempts).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")           # off-screen, as in training

import sim2sim_getup as s2s  # noqa: E402
from getup.poses import quat_from_rpy, quat_to_mat  # noqa: E402
from getup.scenario_core import (DEFAULT_SCRIPT, DT, LABELS, Controller, DeployPolicy,  # noqa: E402
                                 Forces, fsm_cfg_from_config, parse_script)

COLORS = {"passive": (150, 150, 150), "velocity": (60, 200, 90), "fallen": (230, 70, 60),
          "getup_wait": (240, 170, 40), "getup": (250, 140, 30)}


def latest_policy_dir(folder: Path) -> Path:
    """As g1_ctrl (param::parser_policy_dir): the folder itself if it has exported/, else
    the last of its subfolders (sorted by name) that has one."""
    if (folder / "exported").is_dir():
        return folder
    subs = sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
    for p in reversed(subs):
        if (p / "exported").is_dir():
            return p
    raise SystemExit(f"no policy in {folder} (expected <version>/exported/policy.onnx)")


def policy_files(arg: str | None, default_dir: Path) -> tuple[Path, Path]:
    """-> (policy.onnx, the folder whose params/deploy.yaml goes with it)."""
    if arg is None:
        return default_dir / "exported" / "policy.onnx", default_dir
    p = Path(arg).expanduser()
    if not p.is_absolute() and not p.exists() and (HERE / p).exists():
        p = HERE / p                                       # policy/<name> of this repo
    if p.is_dir():
        if not (p / "policy.onnx").is_file() and (p / "exported" / "policy.onnx").is_file():
            # a policy folder (g1_ctrl layout): its own deploy.yaml, if it has one
            own = (p / "params" / "deploy.yaml").is_file()
            return p / "exported" / "policy.onnx", (p if own else default_dir)
        return p / "policy.onnx", default_dir
    return p, default_dir


def short_name(onnx: Path) -> str:
    """A label for the video: the run folder (or g1_ctrl's version folder)."""
    parent = onnx.parent.parent if onnx.parent.name == "exported" else onnx.parent
    return parent.name


class Recorder:
    """Off-screen frames (EGL) streamed to an mp4, with the state and the events on it."""

    def __init__(self, sim, path: Path, title: str, width: int = 960, height: int = 540,
                 fps: int = 25, azimuth: float = 135.0):
        import mujoco
        self.sim, self.path = sim, path
        m = sim.m
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), width)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), height)
        self.r = mujoco.Renderer(m, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.4, azimuth, -12.0
        self.look = None
        self.title, self.n = title, 0
        try:
            import imageio_ffmpeg
            import mediapy
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
            self.w = mediapy.VideoWriter(str(path), shape=(height, width), fps=fps)
            self.w.__enter__()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"cannot write {path}: {e} (needs mediapy and imageio-ffmpeg, as "
                             "sim2sim_getup.py --video)") from None
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.Image, self.ImageDraw = Image, ImageDraw
            self.font, self.big = self._font(ImageFont, 20), self._font(ImageFont, 30)
        except ImportError:
            self.Image = None
            print("[WARNING] no PIL: the video has no text on it")

    @staticmethod
    def _font(ImageFont, size):
        for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    def frame(self, t: float, state: str, lines: list[str], banner: str | None) -> None:
        p = self.sim.d.qpos[self.sim.r:self.sim.r + 3]
        target = np.array([p[0], p[1], 0.55])
        self.look = target if self.look is None else self.look + 0.12 * (target - self.look)
        self.cam.lookat[:] = self.look
        self.r.update_scene(self.sim.d, camera=self.cam)
        img = self.r.render()
        if self.Image is not None:
            im = self.Image.fromarray(img)
            dr = self.ImageDraw.Draw(im, "RGBA")
            col = COLORS.get(state, (200, 200, 200))
            head = f"t {t:5.1f} s   {LABELS[state]}"
            rows = [head] + lines
            h = 14 + 26 * len(rows)
            dr.rectangle([10, 10, 520, 10 + h], fill=(0, 0, 0, 150))
            dr.rectangle([10, 10, 18, 10 + h], fill=col + (255,))
            for i, txt in enumerate(rows):
                dr.text((28, 18 + 26 * i), txt, font=self.font, fill=(255, 255, 255, 255))
            tw = dr.textlength(self.title, font=self.font)
            dr.text((im.width - 12 - tw, im.height - 34), self.title, font=self.font,
                    fill=(255, 255, 255, 200))
            if banner:
                w = dr.textlength(banner, font=self.big)
                x0 = (im.width - w) / 2 - 16
                dr.rectangle([x0, im.height - 90, x0 + w + 32, im.height - 44], fill=(0, 0, 0, 170))
                dr.text((x0 + 16, im.height - 84), banner, font=self.big, fill=(255, 220, 80, 255))
            img = np.asarray(im)
        self.w.add_image(img)
        self.n += 1

    def close(self) -> None:
        self.w.__exit__(None, None, None)


def robot_state(sim):
    """What g1_ctrl's policies read (policy joint order), and what the states check."""
    from types import SimpleNamespace
    d = sim.d
    rot = quat_to_mat(d.qpos[sim.r + 3:sim.r + 7])
    g = rot.T @ np.array([0.0, 0.0, -1.0])
    gyro = d.qvel[sim.rd + 3:sim.rd + 6].copy()            # free joint: body frame (IMU)
    qd = d.qvel[sim.dadr].copy()
    return SimpleNamespace(
        q=d.qpos[sim.qadr].copy(), qd=qd, gyro=gyro, gravity=g,
        tilt=math.acos(float(np.clip(-g[2], -1.0, 1.0))), gyro_norm=float(np.linalg.norm(gyro)),
        qd_max=float(np.abs(qd).max()), pos=d.qpos[sim.r:sim.r + 3].copy(),
        vel=d.qvel[sim.rd:sim.rd + 3].copy(), heading=math.atan2(rot[1, 0], rot[0, 0]))


def describe(e) -> str:
    if e.kind == "cmd":
        return f"joystick: vx {e.args[0]:+.2f} m/s, vy {e.args[1]:+.2f} m/s, wz {e.args[2]:+.2f} rad/s"
    if e.kind == "push":
        names = {0.0: "forward", 90.0: "left", 180.0: "backward", -90.0: "right"}
        d = names.get(e.args[0], f"{e.args[0]:g} deg")
        return f"push {d}: {e.args[1]:.0f} N for {e.args[2]:g} s"
    return {"walk": "walking policy on", "limp": "motors off (L2+B)", "getup": "get-up buttons (R2+X)",
            "band": f"hung from the band (pelvis ~{e.args[0]:.2f} m), motors off" if e.args else "band",
            "release": "band released: it drops", "end": "end"}[e.kind]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", default="deploy", help="g1_ctrl's deploy folder (default ./deploy)")
    ap.add_argument("--robot", default="g1_23dof")
    ap.add_argument("--walk", default=None, help="another walking policy: a run folder, a file or a policy folder")
    ap.add_argument("--getup", default=None, help="another get-up policy: a run folder, a file or a policy folder")
    ap.add_argument("--script", default=None, help="scenario file (default: the built-in one)")
    ap.add_argument("--print-script", action="store_true", help="print the default scenario and exit")
    ap.add_argument("--out", default="scenario_videos", help="folder for the mp4 and the log")
    ap.add_argument("--scene", default=None, help="MuJoCo scene (default: Unitree's scene_g1_23dof.xml)")
    ap.add_argument("--azimuth", type=float, default=135.0, help="camera direction (deg)")
    ap.add_argument("--no-motor-curve", action="store_true", help="no torque-speed curve")
    ap.add_argument("--keep-scene-limits", action="store_true",
                    help="keep the scene's torque limits (right ankle 25 N m)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.print_script:
        print(DEFAULT_SCRIPT, end="")
        return 0
    try:
        events = parse_script(Path(args.script).expanduser().read_text() if args.script else DEFAULT_SCRIPT)
    except (OSError, ValueError) as e:
        raise SystemExit(f"scenario: {e}") from None
    import yaml

    robot_dir = Path(args.deploy).expanduser() / "robots" / args.robot
    cfg_file = robot_dir / "config" / "config.yaml"
    if not cfg_file.is_file():
        raise SystemExit(f"no {cfg_file}: run from unitree_rl_mjlab or pass --deploy")
    config = yaml.safe_load(cfg_file.read_text())
    walk_dir = latest_policy_dir(robot_dir / "config" / "policy" / "velocity")
    getup_dir = latest_policy_dir(robot_dir / "config" / "policy" / "getup")
    walk_onnx, walk_dir = policy_files(args.walk, walk_dir)
    getup_onnx, getup_dir = policy_files(args.getup, getup_dir)
    for f in (walk_onnx, getup_onnx):
        if not f.is_file():
            raise SystemExit(f"no policy file {f}")
    walk_net, getup_net = s2s.OnnxPolicy(walk_onnx), s2s.OnnxPolicy(getup_onnx)
    walk = DeployPolicy(yaml.safe_load((walk_dir / "params" / "deploy.yaml").read_text()), walk_net, "walking")
    getup = DeployPolicy(yaml.safe_load((getup_dir / "params" / "deploy.yaml").read_text()), getup_net, "get-up")
    for p, net in ((walk, walk_net), (getup, getup_net)):
        if net.n_in and net.n_in != p.size:
            raise SystemExit(f"{p.name} policy: the network takes {net.n_in} inputs, its deploy.yaml "
                             f"gives {p.size}")
    if walk.ids_map != getup.ids_map:
        raise SystemExit("the walking and get-up deploy.yaml have different joint_ids_map")
    names_w, names_g = s2s._csv(walk_net.meta, "joint_names"), s2s._csv(getup_net.meta, "joint_names")
    if names_w and names_g and names_w != names_g:
        raise SystemExit("the walking and get-up policies have different joint orders")
    names = names_w or names_g
    if not names or len(names) != walk.n:
        raise SystemExit("no joint_names metadata in the ONNX files (needed to find the joints)")

    import mujoco  # noqa: F401  (fail early with a clear error)
    from src.assets.robots.unitree_g1 import g1_23dof_constants as g1c
    rows = s2s.joint_rows(names, 0.5)
    scene = Path(args.scene) if args.scene else g1c.G1_23DOF_XML.parent / "scene_g1_23dof.xml"
    model, notes = s2s.build_scene(scene, rows, args.keep_scene_limits)
    sim = s2s.Sim(model, rows, curve=not args.no_motor_curve)
    mass = float(sum(model.body_mass[b] for b in range(model.nbody)
                     if model.body_rootid[b] == model.body("pelvis").id))
    fsm = fsm_cfg_from_config(config, walk.ids_map)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"scenario_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    log_lines = []

    def log(t, text):
        line = f"{t:6.2f} s  {text}"
        log_lines.append(line)
        print("  " + line)

    head = [f"scene: {scene}", *(f"  {n}" for n in notes),
            f"walking policy: {walk_onnx}  ({walk_dir / 'params' / 'deploy.yaml'})",
            f"get-up policy:  {getup_onnx}  ({getup_dir / 'params' / 'deploy.yaml'})",
            f"states: fall at tilt > {math.degrees(fsm.fall_tilt):.0f} deg; auto get-up "
            f"{'on' if fsm.auto_getup else 'off'}; GetUp waits until still {fsm.settle_s:g} s "
            f"(max {fsm.max_settle_s:g} s), standing {fsm.hold_s:g} s after {fsm.min_time_s:g} s, "
            f"gives up after {fsm.max_time_s:g} s -> {fsm.on_timeout}; limp damping "
            f"{fsm.passive_kd.min():g}..{fsm.passive_kd.max():g}"]
    for h in head:
        print(h)
    log_lines.extend(head + [""])

    ctrl = Controller(fsm, walk, getup, log)
    forces = Forces(mass)
    sim.place(quat_from_rpy(0.0, 0.0, 0.0), walk.default, clearance=0.002)
    title = f"walk {short_name(walk_onnx)} | get-up {short_name(getup_onnx)}"
    rec = Recorder(sim, stem.with_suffix(".mp4"), title, azimuth=args.azimuth)
    end_t = next((e.t for e in events if e.kind == "end"), (events[-1].t if events else 0.0) + 10.0)
    banner, banner_until = None, -1.0
    ev_i, step, t = 0, 0, 0.0
    falls = 0
    print(f"running {end_t:.0f} s of scenario ...")
    while t < end_t - 1e-9:
        s = robot_state(sim)
        while ev_i < len(events) and events[ev_i].t <= t + 1e-9:
            e = events[ev_i]
            ev_i += 1
            text = describe(e)
            log(t, f"[event] {text}")
            banner, banner_until = text, t + 2.0
            if e.kind == "walk":
                ctrl.request("walk", t)
            elif e.kind == "cmd":
                ctrl.cmd = np.asarray(e.args, float)
            elif e.kind == "push":
                forces.push_at(t, s.heading, *e.args)
            elif e.kind == "limp":
                ctrl.request("limp", t)
            elif e.kind == "getup":
                ctrl.request("getup", t)
            elif e.kind == "band":
                forces.band_on(t, s.pos, e.args[0])
                if ctrl.state != "passive":
                    ctrl.request("limp", t)
            elif e.kind == "release":
                forces.band_off()
        before = ctrl.state
        out = ctrl.tick(t, s)
        if ctrl.state != before:
            banner, banner_until = LABELS[ctrl.state], t + 2.0
            if ctrl.state == "fallen" and before == "velocity":
                falls += 1
        sim.d.xfrc_applied[sim.torso, :3] = forces.force(t, s.pos, s.vel)
        if out.mode == "pd":
            sim.kp[:], sim.kd[:] = out.kp, out.kd
            _, head_f = sim.step(out.q)
        else:
            sim.limp_kd[:] = out.kd
            sim.limp()
            head_f = sim.floor_contacts()[2]
        if ctrl.state in ("getup_wait", "getup") and ctrl.attempts:
            ctrl.attempts[-1]["head_N"] = max(ctrl.attempts[-1]["head_N"], head_f)
        if not sim.finite():
            log(t, "the simulation blew up (NaN): stopped")
            break
        t += DT
        step += 1
        if step % 2 == 0:
            lines = []
            if ctrl.state == "velocity":
                c = walk.command(ctrl.cmd)
                lines.append(f"joystick  vx {c[0]:+.2f}  vy {c[1]:+.2f}  wz {c[2]:+.2f}")
            elif ctrl.state in ("getup", "getup_wait") and ctrl.policy_t0 is not None:
                lines.append(f"get-up policy for {t - ctrl.policy_t0:.1f} s, "
                             f"tilt {math.degrees(s.tilt):.0f} deg")
            if forces.band is not None:
                lines.append("elastic band on")
            rec.frame(t, ctrl.state, lines, banner if t <= banner_until else None)
        if step % 500 == 0:
            print(f"  ... {t:.0f} s")
    rec.close()

    summary = ["", f"falls detected while walking: {falls}", "get-up attempts:"]
    for i, a in enumerate(ctrl.attempts, 1):
        pol = f"policy at {a['policy']:.1f} s" if a["policy"] is not None else "policy never started"
        took = (f", standing {a['end'] - a['policy']:.1f} s after the policy started"
                if a["result"] == "standing" and a["policy"] is not None else "")
        summary.append(f"  {i}. {a['start']:.1f} s: {pol}; {a['result'] or 'still running at the end'}"
                       f"{took}; hardest head contact {a['head_N']:.0f} N")
    if not ctrl.attempts:
        summary.append("  none")
    for line in summary:
        print(line)
    log_lines.extend(summary)
    stem.with_suffix(".txt").write_text("\n".join(log_lines) + "\n")
    print(f"saved {stem.with_suffix('.mp4')} ({rec.n} frames) and {stem.with_suffix('.txt').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
