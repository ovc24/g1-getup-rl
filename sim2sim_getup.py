#!/usr/bin/env python3
"""Sim2sim of the get-up policy: Unitree's MuJoCo model of the G1-23DoF (the one of
unitree_mujoco and g1_ctrl), on the CPU, one robot, one episode after the other.

    cd <unitree_rl_mjlab> && source .venv/bin/activate
    RUN=logs/rsl_rl/g1_23dof_getup/<run>                  # the run FOLDER (its policy.onnx)
    python <this folder>/sim2sim_getup.py --policy $RUN                    # 60 episodes
    python <this folder>/sim2sim_getup.py --policy $RUN --view --episodes 5
    python <this folder>/sim2sim_getup.py --policy $RUN --postures prone
    python <this folder>/sim2sim_getup.py --policy $RUN --start collapse --limp-kd 3
    python <this folder>/sim2sim_getup.py --policy $RUN --released collapse --pool <pool>

--policy takes the run folder, a .onnx file, or a model_<N>.pt (then the policy.onnx that
the same save wrote: every save writes model_<N>.pt and then policy.onnx, so policy.onnx
is always the LAST checkpoint). Needs onnxruntime (uv pip install onnxruntime).

Each episode: a pose of the fallen-pose pool, turned to a random heading, is put just
above the ground of Unitree's scene and left to settle with limp motors (damping only,
like the robot's damping mode) until it is still; then the policy runs for 10 s. The
report has the same table as eval_getup.py.

--start collapse: instead of a pool pose, the robot stands in the walking pose (+-3 deg,
barely moving) and the motors go limp, as when Passive (L2+B) is pressed while it stands
in g1_ctrl; once it lies still, the policy runs. --limp-kd 3: the damping of g1_ctrl's
Passive and Fallen states (all motors; default: the policy's kd per joint).
--released collapse,standing: only pool poses released that way (make_fallen_poses.py).

SAME as in training: the policy (policy.onnx: normalizer + actor, mean action), its 375
inputs (5 signals x 5 steps, with the training noise, in mjlab's order), the PD gains,
torque limits and torque-speed curve, the action scale, the standing test.
DIFFERENT (that is the test):
  - collision shapes: Unitree's scene collides with the MESHES of every link (convex
    hulls) and 4 small spheres per foot; training: capsules and spheres;
  - armature 0.01, damping 0.05 and dry friction 0.2 N m in every joint (training: armature
    per motor, no damping, no dry friction); contact friction 1.0 (training 0.35-1.1);
  - MuJoCo on the CPU, 2 ms steps, PD every 2 ms (training: MuJoCo Warp, 5 ms steps);
  - no randomization; the start poses settle again in this scene first.
Unitree's scene limits the right ankle motors to +-25 N m (left +-50, training 50): raised
to 50 unless --keep-scene-limits.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from getup.poses import (DEFAULT_POOL, DROPS, POSTURES, classify, load_pool,  # noqa: E402
                         lowest_point, quat_from_rpy, quat_mul, quat_to_mat,
                         released_mask, resolve_regex_values, sample_weights)
from getup.run_options import parse_postures, resolve_pool  # noqa: E402
from getup.sim2sim_core import (Episode, ObsHistory, ObsTerm, is_standing,  # noqa: E402
                                mesh_lowest_z, motor_torque, report)
from getup.stand import DEFAULT_POSTURE_WEIGHTS, HEAD_OFFSET, STAND  # noqa: E402

GROUP_LABELS = ("arms", "hip pitch/yaw + waist", "knee + hip roll", "ankles")  # robot.motor_groups()
DT = 0.02


def resolve_policy_file(arg: str) -> Path:
    """Run folder -> its policy.onnx; a policy folder of this repo (policy/<name>, g1_ctrl's
    layout) -> its exported/policy.onnx; .onnx -> itself; model_<N>.pt -> the policy.onnx of
    the same save (the sim2sim runs the ONNX file, as the robot does, not the PyTorch file)."""
    p = Path(arg).expanduser()
    if not p.is_absolute() and not p.exists() and (HERE / p).exists():
        p = HERE / p                                       # policy/<name> of this repo
    if p.is_dir():
        if not (p / "policy.onnx").is_file() and (p / "exported" / "policy.onnx").is_file():
            return p / "exported" / "policy.onnx"
        return p / "policy.onnx"
    if p.suffix != ".pt":
        return p
    onnx = p.parent / "policy.onnx"
    if not onnx.is_file():
        raise SystemExit(f"{p.name} is a PyTorch checkpoint and there is no {onnx} next to it")
    gap = onnx.stat().st_mtime - p.stat().st_mtime      # same save: a few seconds after
    if not 0.0 <= gap < 120.0:
        def it(f: Path) -> int:
            m = re.search(r"(\d+)$", f.stem)
            return int(m.group(1)) if m else -1
        last = max(p.parent.glob("model_*.pt"), key=it)
        raise SystemExit(
            f"{p.name} is a PyTorch checkpoint; the sim2sim runs policy.onnx (what the robot "
            f"runs), and {onnx} was written by another save: it is the policy of {last.name}. "
            f"Test that one with --policy {p.parent}")
    print(f"[INFO] {p.name} is a PyTorch checkpoint: using {onnx} (written by the same save)")
    return onnx


def _csv(meta: dict, key: str):
    v = meta.get(key)
    return None if v is None else [s.strip() for s in v.split(",")]


class OnnxPolicy:
    def __init__(self, path: Path):
        try:
            import onnxruntime as ort
        except ImportError:
            raise SystemExit("onnxruntime is missing: uv pip install onnxruntime") from None
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.out = self.sess.get_outputs()[0].name
        self.meta = dict(self.sess.get_modelmeta().custom_metadata_map)
        shape = self.inp.shape
        self.n_in = shape[-1] if isinstance(shape[-1], int) else None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.sess.run([self.out], {self.inp.name: obs.astype(np.float32)})[0]


def joint_rows(names, factor: float):
    """One row per policy joint: kp, kd, torque limit, no-load speed, default, scale, group."""
    from getup.robot import action_scale, motor_groups
    from src.assets.robots.unitree_g1 import g1_23dof_constants as g1c
    groups = motor_groups()
    scale = action_scale(factor)
    home = g1c.HOME_KEYFRAME.joint_pos
    rows = []
    for n in names:
        hits = [(gi, a, v) for gi, (a, v) in enumerate(groups)
                if any(re.fullmatch(p, n) for p in a.target_names_expr)]
        if len(hits) != 1:
            raise SystemExit(f"joint {n}: {len(hits)} motor groups match (expected 1)")
        gi, a, vmax = hits[0]
        sc = [v for p, v in scale.items() if re.fullmatch(p, n)]
        rows.append(SimpleNamespace(
            name=n, kp=float(a.stiffness), kd=float(a.damping), tau=float(a.effort_limit),
            vmax=float(vmax), default=float(resolve_regex_values(home, [n], default=0.0)[0]),
            scale=float(sc[0]), group=gi))
    return rows


def build_scene(scene: Path, rows, keep_limits: bool):
    """Unitree's scene, the torque limits raised to the training ones where lower."""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(scene))
    want = {r.name: r.tau for r in rows}
    notes = []
    for act in spec.actuators:
        j = act.target
        if j not in want:
            continue
        lim = float(np.max(np.abs(np.asarray(act.ctrlrange))))
        if lim + 1e-6 < want[j]:
            if keep_limits:
                notes.append(f"{act.name}: torque limit {lim:g} N m (training {want[j]:g}), kept")
            else:
                act.ctrlrange = np.array([-want[j], want[j]])
                notes.append(f"{act.name}: torque limit {lim:g} -> {want[j]:g} N m (as in training)")
    return spec.compile(), notes


class Sim:
    """One robot in Unitree's scene: PD + motor model every physics step (2 ms), contacts."""

    def __init__(self, model, rows, curve: bool):
        import mujoco
        self.mj, self.m = mujoco, model
        self.d = mujoco.MjData(model)
        self.nsub = int(round(DT / model.opt.timestep))
        if abs(self.nsub * model.opt.timestep - DT) > 1e-9:
            raise SystemExit(f"physics step {model.opt.timestep} s does not divide 20 ms")
        pelvis = model.body("pelvis").id
        rj = int(model.body_jntadr[pelvis])
        if rj < 0 or model.jnt_type[rj] != mujoco.mjtJoint.mjJNT_FREE:
            raise SystemExit("the pelvis has no free joint in this scene")
        self.r, self.rd = int(model.jnt_qposadr[rj]), int(model.jnt_dofadr[rj])
        jids = [model.joint(r.name).id for r in rows]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.robot_dofs = np.concatenate((np.arange(self.rd, self.rd + 6), self.dadr))
        act = []
        for r, j in zip(rows, jids):
            acts = [a for a in range(model.nu) if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                    and model.actuator_trnid[a, 0] == j]
            if len(acts) != 1:
                raise SystemExit(f"joint {r.name}: {len(acts)} actuators in the scene (expected 1)")
            if abs(model.actuator_gear[acts[0], 0] - 1.0) > 1e-9 or model.actuator_biastype[acts[0]] != 0:
                raise SystemExit(f"joint {r.name}: the scene's actuator is not a plain motor")
            act.append(acts[0])
        self.act = np.array(act)
        lo = model.actuator_ctrlrange[self.act, 0].copy()
        hi = model.actuator_ctrlrange[self.act, 1].copy()
        limited = model.actuator_ctrllimited[self.act].astype(bool)
        lo[~limited], hi[~limited] = -np.inf, np.inf
        self.ctrl_lo, self.ctrl_hi = lo, hi
        self.kp = np.array([r.kp for r in rows])
        self.kd = np.array([r.kd for r in rows])
        self.limp_kd = self.kd.copy()           # damping while limp (g1_ctrl Passive: 3)
        self.tau = np.array([r.tau for r in rows])
        self.vmax = np.array([r.vmax for r in rows])
        self.curve = curve
        bodies = set(np.flatnonzero(model.body_rootid == pelvis).tolist())
        collide = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        self.robot_geoms = np.array([g for g in range(model.ngeom)
                                     if model.geom_bodyid[g] in bodies and collide[g]])
        feet = {model.body(b).id for b in ("left_ankle_roll_link", "right_ankle_roll_link")}
        self.foot_geoms = {int(g) for g in self.robot_geoms if model.geom_bodyid[g] in feet}
        self.robot_set = {int(g) for g in self.robot_geoms}
        self.torso = model.body("torso_link").id
        head_mesh = model.mesh("head_link").id
        heads = [int(g) for g in self.robot_geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                 and model.geom_dataid[g] == head_mesh]
        if len(heads) != 1:
            raise SystemExit(f"expected one colliding head mesh geom, found {len(heads)}")
        self.head = heads[0]
        try:
            self.floor = model.geom("floor").id
        except KeyError:
            planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
            if len(planes) != 1:
                raise SystemExit("no floor geom in the scene") from None
            self.floor = int(planes[0])
        self.verts = {}
        for g in self.robot_geoms:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                self.verts[int(g)] = np.asarray(model.mesh_vert[a:a + n], dtype=float)
        self.sat = np.zeros(len(rows))
        self.substeps = 0
        self._f6 = np.zeros(6)

    # -- state ------------------------------------------------------------------------
    def lowest(self) -> float:
        z = math.inf
        d, m = self.d, self.m
        for g in self.robot_geoms:
            g = int(g)
            if g in self.verts:
                z = min(z, mesh_lowest_z(d.geom_xpos[g], d.geom_xmat[g], self.verts[g]))
            else:
                z = min(z, lowest_point([m.geom_type[g]], [m.geom_size[g]], [d.geom_xpos[g]],
                                        [d.geom_xmat[g]], [m.geom_rbound[g]]))
        return z

    def _pairs(self):
        n = self.d.ncon
        if n == 0:
            return np.zeros((0, 2), dtype=int)
        c = self.d.contact
        try:
            return np.asarray(c.geom).reshape(-1, 2)[:n]
        except AttributeError:
            return np.stack((np.asarray(c.geom1)[:n], np.asarray(c.geom2)[:n]), axis=1)

    def self_penetration(self) -> float:
        """Deepest overlap between two robot geoms (m, >= 0)."""
        pairs = self._pairs()
        if not len(pairs):
            return 0.0
        dist = np.asarray(self.d.contact.dist)[:len(pairs)]
        worst = 0.0
        for (a, b), dd in zip(pairs.tolist(), dist):
            if a in self.robot_set and b in self.robot_set:
                worst = max(worst, -float(dd))
        return worst

    def floor_contacts(self):
        """(other body on the floor, head on the floor, head-floor force N, a foot on the
        floor)."""
        pairs = self._pairs()
        other, head, force, foot = False, False, 0.0, False
        for i, (a, b) in enumerate(pairs.tolist()):
            if a == self.floor:
                g = b
            elif b == self.floor:
                g = a
            else:
                continue
            if g not in self.robot_set:
                continue
            if g in self.foot_geoms:
                foot = True
            else:
                other = True
            if g == self.head:
                head = True
                self.mj.mj_contactForce(self.m, self.d, i, self._f6)
                force += float(np.linalg.norm(self._f6[:3]))
        return other, head, force, foot

    def pelvis_speeds(self) -> tuple[float, float]:
        """(|vertical speed| m/s, |angular speed| rad/s) of the pelvis (free joint: linear
        velocity in the world frame, angular in the body frame; the norm is the same)."""
        v = self.d.qvel
        return abs(float(v[self.rd + 2])), float(np.linalg.norm(v[self.rd + 3:self.rd + 6]))

    def pelvis_quat(self) -> np.ndarray:
        return self.d.qpos[self.r + 3:self.r + 7].copy()

    def observation_values(self, last_action, default_q) -> dict[str, np.ndarray]:
        q, v = self.d.qpos, self.d.qvel
        rot = quat_to_mat(q[self.r + 3:self.r + 7])
        return {
            "base_ang_vel": v[self.rd + 3:self.rd + 6].copy(),     # free joint: body frame
            "projected_gravity": rot.T @ np.array([0.0, 0.0, -1.0]),
            "joint_pos": q[self.qadr] - default_q,
            "joint_vel": v[self.dadr].copy(),
            "actions": np.asarray(last_action, dtype=float),
        }

    def stand_values(self):
        q = self.d.qpos
        rot = quat_to_mat(q[self.r + 3:self.r + 7])
        head = self.d.xpos[self.torso] + self.d.xmat[self.torso].reshape(3, 3) @ np.asarray(HEAD_OFFSET)
        return float(q[self.r + 2]), float(head[2]), float(rot[2, 2])

    # -- moving ----------------------------------------------------------------------------
    def place(self, quat, joint_q, clearance: float = 0.01) -> float:
        """Put the robot with this orientation and these joint angles, its lowest point
        `clearance` above the floor, at rest. Returns the deepest self-overlap (m)."""
        mj, m, d = self.mj, self.m, self.d
        mj.mj_resetData(m, d)
        d.qpos[self.r:self.r + 3] = (0.0, 0.0, 2.0)
        d.qpos[self.r + 3:self.r + 7] = quat
        d.qpos[self.qadr] = joint_q
        mj.mj_kinematics(m, d)
        d.qpos[self.r + 2] += clearance - self.lowest()
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)
        return self.self_penetration()

    def _substeps(self, tau_fn, track: bool):
        d = self.d
        head_on, head_f = False, 0.0
        for _ in range(self.nsub):
            tau_pd = tau_fn(d.qpos[self.qadr], d.qvel[self.dadr])
            tau = motor_torque(tau_pd, d.qvel[self.dadr], self.tau, self.vmax, self.curve)
            tau = np.clip(tau, self.ctrl_lo, self.ctrl_hi)       # what MuJoCo applies
            if track:
                self.sat += np.abs(tau - tau_pd) > 1e-6
                self.substeps += 1
            d.ctrl[self.act] = tau
            self.mj.mj_step(self.m, d)
            if track:
                _, on, f, _ = self.floor_contacts()
                head_on |= on
                head_f = max(head_f, f)
        return head_on, head_f

    def step(self, q_des):
        """One policy step (20 ms) of PD towards q_des. Returns (head touched, max force)."""
        return self._substeps(lambda q, qd: self.kp * (q_des - q) - self.kd * qd, track=True)

    def limp(self):
        """20 ms with the motors limp: damping only (kp = 0)."""
        self._substeps(lambda q, qd: -self.limp_kd * qd, track=False)

    def place_standing(self, joint_q, rng, tilt_deg: float = 3.0, vel: float = 0.1):
        """Standing on its feet in this pose, tilted +-tilt_deg, random heading, a little
        random velocity (+-vel m/s, +-3 vel rad/s)."""
        r, p = np.radians(tilt_deg) * rng.uniform(-1.0, 1.0, size=2)
        yaw = float(rng.uniform(-math.pi, math.pi))
        quat = quat_mul(quat_from_rpy(0.0, 0.0, yaw), quat_from_rpy(float(r), float(p), 0.0))
        self.place(quat, joint_q, clearance=0.002)
        self.d.qvel[self.rd:self.rd + 2] = rng.uniform(-vel, vel, size=2)
        self.d.qvel[self.rd + 3:self.rd + 6] = rng.uniform(-3.0 * vel, 3.0 * vel, size=3)
        self.mj.mj_forward(self.m, self.d)

    def still(self, vel: float) -> bool:
        return float(np.abs(self.d.qvel[self.robot_dofs]).max()) < vel

    def finite(self) -> bool:
        return bool(np.all(np.isfinite(self.d.qpos)) and np.all(np.isfinite(self.d.qvel)))


class Viewer:
    def __init__(self, sim: Sim, speed: float):
        import mujoco.viewer
        self.v = mujoco.viewer.launch_passive(sim.m, sim.d)
        self.speed = max(speed, 1e-3)
        self.t0, self.n = time.time(), 0

    def tick(self) -> bool:
        if not self.v.is_running():
            return False
        self.v.sync()
        self.n += 1
        lag = self.t0 + self.n * DT / self.speed - time.time()
        if lag > 0:
            time.sleep(lag)
        return True


class VideoRecorder:
    """Draws the robot without a window (EGL, as in training) and saves, per episode, an
    mp4 and a sheet of 12 frames from the policy's first 3 s (left to right, top to
    bottom, 0.25 s apart)."""

    FPS = 25

    def __init__(self, sim: Sim, out_dir: str, width: int = 480, height: int = 360):
        import mujoco
        self.mj, self.sim = mujoco, sim
        self.dir = Path(out_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.r = mujoco.Renderer(sim.m, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.8, 135.0, -15.0
        self.frames, self.policy_start, self.n = [], 0, 0

    def tick(self, every: int = 2):
        self.n += 1
        if self.n % every:
            return
        p = self.sim.d.qpos[self.sim.r:self.sim.r + 3]
        self.cam.lookat[:] = [p[0], p[1], max(float(p[2]), 0.4)]
        self.r.update_scene(self.sim.d, camera=self.cam)
        self.frames.append(self.r.render().copy())

    def mark_policy_start(self):
        self.policy_start = len(self.frames)

    def save(self, name: str) -> str:
        frames, self.frames, self.n = self.frames, [], 0
        if not frames:
            return ""
        out = []
        try:
            import imageio_ffmpeg
            import mediapy
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
            mediapy.write_video(str(self.dir / f"{name}.mp4"), frames, fps=self.FPS)
            out.append(f"{name}.mp4")
        except Exception as e:  # noqa: BLE001
            print(f"  (no mp4: {e})")
        k = [self.policy_start + int(i * 0.25 * self.FPS) for i in range(12)]
        k = [min(i, len(frames) - 1) for i in k]
        rows = [np.concatenate([frames[i] for i in k[j:j + 4]], axis=1) for j in (0, 4, 8)]
        sheet = np.concatenate(rows, axis=0)
        try:
            from PIL import Image
            Image.fromarray(sheet).save(self.dir / f"{name}_sheet.png")
            out.append(f"{name}_sheet.png")
        except Exception:  # noqa: BLE001
            np.save(self.dir / f"{name}_sheet.npy", sheet)
            out.append(f"{name}_sheet.npy")
        return ", ".join(out)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True,
                    help="run folder (with policy.onnx), a .onnx file, or a model_<N>.pt")
    ap.add_argument("--scene", default=None,
                    help="MuJoCo scene (default: Unitree's scene_g1_23dof.xml in unitree_rl_mjlab)")
    ap.add_argument("--pool", default=None, help="fallen-pose pool (default: the run's)")
    ap.add_argument("--postures", default=None,
                    help="starting postures, e.g. 'prone' (names: " + ", ".join(POSTURES) + ")")
    ap.add_argument("--released", default=None,
                    help="only pool poses released this way, e.g. 'collapse' (names: "
                         + ", ".join(DROPS) + ")")
    ap.add_argument("--start", choices=("pool", "collapse"), default="pool",
                    help="pool: a pose of the pool; collapse: standing in the walking pose, "
                         "then limp until it lies still (as Passive in g1_ctrl)")
    ap.add_argument("--limp-kd", type=float, default=None,
                    help="damping of every motor while limp (g1_ctrl Passive/Fallen: 3); "
                         "default: the policy's kd per joint")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--time", type=float, default=10.0, help="policy time per episode (s)")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="at most this long with limp motors before the policy starts (s)")
    ap.add_argument("--action-scale", type=float, default=None,
                    help="action scale factor (default: the run's, else 0.5)")
    ap.add_argument("--no-motor-curve", action="store_true",
                    help="full torque at any speed (training with realistic motors: the "
                         "torque falls with speed)")
    ap.add_argument("--keep-scene-limits", action="store_true",
                    help="keep the scene's torque limits where they are lower than in training")
    ap.add_argument("--no-obs-noise", action="store_true", help="no sensor noise on the inputs")
    ap.add_argument("--view", action="store_true", help="watch in the MuJoCo viewer, real time")
    ap.add_argument("--speed", type=float, default=1.0, help="with --view: playback speed")
    ap.add_argument("--video", default=None, metavar="DIR",
                    help="save each episode as DIR/epNN_<posture>.mp4 + a sheet of frames, "
                         "drawn without a window (use it when the viewer window does not work)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.video and not args.view:
        import os
        os.environ.setdefault("MUJOCO_GL", "egl")      # off-screen, as in training
    rng = np.random.default_rng(args.seed)
    pol_path = resolve_policy_file(args.policy)
    if not pol_path.is_file():
        raise SystemExit(f"no policy file: {pol_path} (each checkpoint save writes policy.onnx "
                         "into the run folder)")
    opts_path = pol_path.parent / "params" / "getup_options.json"
    if not opts_path.exists() and pol_path.parent.name == "exported":     # policy/<name>/
        opts_path = pol_path.parent.parent / "params" / "getup_options.json"
    opts = json.loads(opts_path.read_text()) if opts_path.exists() else {}
    factor = args.action_scale if args.action_scale is not None else opts.get("action_scale", 0.5)
    history = int(opts.get("history", 5))
    curve = (not args.no_motor_curve) and bool(opts.get("realistic", True))
    policy = OnnxPolicy(pol_path)

    import mujoco  # noqa: F401  (fail early with a clear error)
    from getup.getup_env_cfg import make_observations
    from src.assets.robots.unitree_g1 import g1_23dof_constants as g1c

    names = _csv(policy.meta, "joint_names")
    if not names:
        raise SystemExit("the ONNX file has no joint_names metadata")
    rows = joint_rows(names, factor)
    meta_scale = _csv(policy.meta, "action_scale")
    if meta_scale and len(meta_scale) == len(rows):
        diff = max(abs(float(a) - r.scale) for a, r in zip(meta_scale, rows))
        if diff > 2e-3:
            raise SystemExit(f"action scale of the policy differs from factor {factor} "
                             f"(max difference {diff:.3f}): pass the run's --action-scale")
    default_q = np.array([r.default for r in rows])
    meta_def = _csv(policy.meta, "default_joint_pos")
    if meta_def and np.abs(np.array([float(x) for x in meta_def]) - default_q).max() > 2e-3:
        raise SystemExit("default joint angles of the policy differ from the walking pose")

    # Inputs, as the training config builds them (terms, order, noise, history).
    nj = len(rows)
    dims = {"base_ang_vel": 3, "projected_gravity": 3, "joint_pos": nj, "joint_vel": nj,
            "actions": nj}
    grp = make_observations(play=False, history=history)["actor"]
    terms = []
    for name, t in grp.terms.items():
        nz = None if t.noise is None else (float(t.noise.n_min), float(t.noise.n_max))
        terms.append(ObsTerm(name, dims[name], nz))
    meta_obs = _csv(policy.meta, "observation_names")
    if meta_obs and meta_obs != [t.name for t in terms]:
        raise SystemExit(f"observation terms differ:\n  policy: {meta_obs}\n  config: "
                         f"{[t.name for t in terms]}")
    obs = ObsHistory(terms, history, noise=not args.no_obs_noise, rng=rng)
    if policy.n_in and policy.n_in != obs.size:
        raise SystemExit(f"the policy expects {policy.n_in} inputs, the config gives {obs.size}")

    scene = Path(args.scene) if args.scene else g1c.G1_23DOF_XML.parent / "scene_g1_23dof.xml"
    model, notes = build_scene(scene, rows, args.keep_scene_limits)
    sim = Sim(model, rows, curve)

    if args.limp_kd is not None:
        sim.limp_kd[:] = args.limp_kd
    collapse = args.start == "collapse"
    pool = probs = cols = None
    if not collapse:
        pool_path = resolve_pool(args.pool or opts.get("pool") or DEFAULT_POOL)
        pool = load_pool(pool_path)
        cols = [pool.joint_names.index(r.name) for r in rows]
        probs = sample_weights(pool.posture,
                               parse_postures(args.postures) or DEFAULT_POSTURE_WEIGHTS)
        if args.released:
            try:
                probs = probs * released_mask(pool, args.released)
            except ValueError as e:
                raise SystemExit(str(e)) from None
            if probs.sum() <= 0:
                raise SystemExit(f"no pose released as '{args.released}' (with those postures) "
                                 f"in {pool_path}")
            probs = probs / probs.sum()

    print(f"scene: {scene}")
    print(f"physics: {model.opt.timestep * 1000:.1f} ms steps, PD every step, policy every "
          f"{sim.nsub} steps; motor curve {'ON' if curve else 'OFF'}; sensor noise "
          f"{'OFF' if args.no_obs_noise else 'ON'}; action scale {factor}; history {history}")
    for n in notes:
        print("scene change: " + n)
    print(f"inputs: {obs.size} = " + " + ".join(f"{t.name} {t.dim}x{history}" for t in terms)
          + (f"  (the policy expects {policy.n_in})" if policy.n_in else ""))
    limp_txt = (f"{args.limp_kd:g} on every motor" if args.limp_kd is not None
                else "the policy's kd per joint")
    if collapse:
        print(f"start: standing in the walking pose, then limp (damping {limp_txt}) until "
              "it lies still")
    else:
        sel = f", released as {args.released}" if args.released else ""
        print(f"start poses: {len(pool)} from {pool_path}{sel}; settled limp (damping "
              f"{limp_txt})")

    viewer = Viewer(sim, args.speed) if args.view else None
    video = VideoRecorder(sim, args.video) if args.video else None
    scale = np.array([r.scale for r in rows])
    steps = int(round(args.time / DT))
    hold = STAND.hold_s
    eps: list[Episode] = []
    t_wall = time.time()
    for ep in range(args.episodes):
        # 1. a start pose, settled in THIS scene with limp motors
        if collapse:
            sim.place_standing(default_q, rng)
        else:
            for _ in range(20):
                i = int(rng.choice(len(pool), p=probs))
                yaw = float(rng.uniform(-math.pi, math.pi))
                quat = quat_mul(quat_from_rpy(0.0, 0.0, yaw), pool.root_quat[i].astype(float))
                if sim.place(quat, pool.joint_pos[i, cols].astype(float)) < 0.02:
                    break
        settle, still = 0.0, 0.0
        max_settle = max(args.settle, 6.0) if collapse else args.settle
        min_settle = 1.0 if collapse else 0.4          # a collapse needs time to start
        while settle < max_settle:
            sim.limp()
            settle += DT
            still = still + DT if sim.still(0.25) else 0.0
            if viewer and not viewer.tick():
                viewer = None
            if video:
                video.tick()
            if settle >= min_settle and still >= 0.3:
                break
        e = Episode(start=classify(sim.pelvis_quat()), settle_s=settle, peak_qd=np.zeros(nj))
        start_z = float(sim.d.qpos[sim.r + 2])
        # 2. the policy
        if video:
            video.mark_policy_start()
        obs.reset()
        last = np.zeros(nj)
        stand_t = 0.0
        for k in range(steps):
            x = obs(sim.observation_values(last, default_q))
            a = np.asarray(policy(x), dtype=float).reshape(-1)
            if a.size != nj or not np.all(np.isfinite(a)):
                raise SystemExit(f"the policy gives {a.size} actions (expected {nj}) or NaN")
            head_on, head_f = sim.step(default_q + scale * a)
            last = a
            if not sim.finite() or np.abs(sim.d.qvel[sim.dadr]).max() > 60.0:
                e.blown = True
                break
            pz, hz, up = sim.stand_values()
            other, _, _, foot = sim.floor_contacts()
            stand_t = stand_t + DT if is_standing(pz, hz, up, other) else 0.0
            if stand_t >= hold and not e.stood:
                e.stood, e.t_stand = True, (k + 1) * DT - hold
            e.head_s += DT if head_on else 0.0
            e.head_peak = max(e.head_peak, head_f)
            e.peak_qd = np.maximum(e.peak_qd, np.abs(sim.d.qvel[sim.dadr]))
            if (k + 1) * DT > 0.1 and not other and not foot:     # as in training
                e.air_s += DT
            vz, w = sim.pelvis_speeds()
            e.peak_vz, e.peak_w = max(e.peak_vz, vz), max(e.peak_w, w)
            if viewer and not viewer.tick():
                viewer = None
            if video:
                video.tick()
        e.success = stand_t >= hold and not e.blown
        e.end_pelvis = float(sim.d.qpos[sim.r + 2])
        e.end_pose_err = float(np.sqrt(np.mean((sim.d.qpos[sim.qadr] - default_q) ** 2)))
        eps.append(e)
        if video:
            saved = video.save(f"ep{ep + 1:02d}_{'collapse_' if collapse else ''}"
                               f"{POSTURES[e.start]}")
            if saved:
                print(f"  saved {saved} in {video.dir}")
        state = ("stood up" if e.success else "fell again" if e.stood else "did not stand")
        print(f"  episode {ep + 1:3d}/{args.episodes}: {POSTURES[e.start]:10s} "
              f"(pelvis {start_z:4.2f} m) {state:13s}"
              + (f" at {e.t_stand:4.1f} s" if e.stood else "        ")
              + f"  head {e.head_s:4.2f} s, {e.head_peak:4.0f} N"
              + (f", in the air {e.air_s:.2f} s" if e.air_s > 0.0 else "")
              + f"   ({time.time() - t_wall:.0f} s so far)",
              flush=True)

    groups = []
    for gi, label in enumerate(GROUP_LABELS):
        ids = np.array([j for j, r in enumerate(rows) if r.group == gi])
        groups.append((label, ids, float(rows[ids[0]].vmax)))
    share = sim.sat / max(sim.substeps, 1)
    sat = []
    for label, ids, _ in groups:                 # mean over the group, and its worst joint
        worst = ids[int(np.argmax(share[ids]))]
        sat.append((label, float(share[ids].mean()), rows[worst].name, float(share[worst])))
    mode = (f"Unitree scene, {model.opt.timestep * 1000:.0f} ms steps, motor curve "
            f"{'on' if curve else 'off'}, sensor noise {'off' if args.no_obs_noise else 'on'}, "
            f"{args.time:.0f} s per episode after settling, seed {args.seed}"
            + (", starts: limp collapse from standing" if collapse else "")
            + (f", released as {args.released}" if args.released and not collapse else ""))
    print()
    for line in report(eps, groups, sat, f"sim2sim {pol_path.parent.name}/{pol_path.name}", mode):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
