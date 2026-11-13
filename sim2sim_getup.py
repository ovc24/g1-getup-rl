#!/usr/bin/env python3
"""Sim2sim of the get-up policy: Unitree's MuJoCo model of the G1-23DoF (the one of
unitree_mujoco and g1_ctrl), on the CPU, one robot, one episode after the other.

    cd <unitree_rl_mjlab> && source .venv/bin/activate
    RUN=logs/rsl_rl/g1_23dof_getup/<run>                  # the run FOLDER (its policy.onnx)
    python <this folder>/sim2sim_getup.py --policy $RUN                    # 60 episodes
    python <this folder>/sim2sim_getup.py --policy $RUN --view --episodes 5
    python <this folder>/sim2sim_getup.py --policy $RUN --postures prone
    python <this folder>/sim2sim_getup.py --policy $RUN --start collapse --limp-kd 3
    python <this folder>/sim2sim_getup.py --policy $RUN --start collapse --limp-kd 0   # torque off
    python <this folder>/sim2sim_getup.py --policy $RUN --released collapse --pool <pool>
    python <this folder>/sim2sim_getup.py --policy $RUN --benchmark 5      # self-collision benchmark

--policy takes the run folder, a .onnx file, or a model_<N>.pt (then the policy.onnx that
the same save wrote: every save writes model_<N>.pt and then policy.onnx, so policy.onnx
is always the LAST checkpoint). Needs onnxruntime (uv pip install onnxruntime).

Each episode: a pose of the fallen-pose pool, turned to a random heading, is put just
above the ground of Unitree's scene and left to settle with limp motors (damping only,
like the robot's damping mode) until it is still; then the policy runs for 10 s. The
report has the same table as eval_getup.py, and the self-contacts: the time one part of
the robot presses on another (> 5 N, and > 50 N, about what an arm motor can push back
with) while getting up and while standing, the hardest one, which pairs of links, and how
often the start pose was already pressing (a collapse with the torque off, --limp-kd 0,
folds the robot with the arms between the legs). Pairs one joint apart (pelvis - hip roll,
torso - shoulder roll) overlap around their joint in this model: listed, marked, and left
out of the totals; pairs that already touch in the walking pose are marked too.

--start collapse: instead of a pool pose, the robot stands in the walking pose (+-3 deg,
barely moving) and the motors go limp, as when Passive (L2+B) is pressed while it stands
in g1_ctrl; once it lies still, the policy runs. --limp-kd 3: the damping of g1_ctrl's
Passive and Fallen states (all motors; default: the policy's kd per joint).
--released collapse,standing: only pool poses released that way (make_fallen_poses.py).
--start facedown: tipping over forward from the walking pose (15 +-5 deg, 0.6 m/s), limp
until it lies still.

--benchmark N: the self-collision benchmark of a team simulator (what the policy would
do to the robot's body): N episodes each from supine and prone (pool poses) and from a
face-down fall, with the robot's links NOT colliding with each other (only with the
floor); a conflict is one link more than 1 cm (--conflict-cm) into another, measured
every policy step from the collision shapes (MuJoCo's mj_geomDistance), deepest per pair
of links, beyond what the pair already overlaps in the walking pose. A link and its
parent are not compared; pairs one joint apart (they overlap at that joint in this
model) are listed but not counted. Fails (exit 1) if any episode has a conflict or does
not end standing. --no-self-collision and --conflicts do each part alone. Unitree's
shapes are convex hulls, fuller than the real parts: stricter than the real robot.
The conflicts are split into getting up and standing (is_standing at that step), with
the time with a conflict in each, and the starts already in a conflict (links that went
into each other during the limp fall, before the policy).

Watching it (as the team's benchmark video):
    python <this folder>/sim2sim_getup.py --policy $RUN --label getup_v6 --benchmark 5 --video videos/bench_v6
    python <this folder>/scripts/side_by_side.py videos/bench_v5 videos/bench_v6 --out videos/v5_vs_v6
Each video has a header: the policy and the start, "getting up" / "STOOD UP at X s", and
the time with a conflict so far (getting up, standing); the links in a conflict at that
moment are drawn red and named at the bottom with their depth. The same --seed gives two
policies the same starts and camera, so side_by_side.py shows them next to each other
(all.mp4: every start one after the other). --view shows the red links live (and the text,
if this MuJoCo's viewer can).

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
from getup.sim2sim_core import (Episode, ObsHistory, ObsTerm,  # noqa: E402
                                add_conflicts, add_proxy, add_self_contacts, conflict_report,
                                group_report, is_standing, link_groups, mesh_lowest_z,
                                motor_torque, pose_report, proxy_report, report, worst_pair)
from getup.sim2sim_video import (SCENARIO_TEXT, Highlighter, Mp4Writer,  # noqa: E402
                                 Overlay, draw_frame, pair_label, policy_label, tmp_name)
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
        self.body_of = {int(g): model.body(int(model.geom_bodyid[g])).name for g in self.robot_geoms}
        # Pairs of links one joint apart (a link and its grandparent, e.g. pelvis and hip
        # roll): their shapes overlap around the joint between them in this model.
        # Unitree's training model excludes pelvis - hip roll for that reason.
        par = model.body_parentid
        cb = sorted({int(model.geom_bodyid[g]) for g in self.robot_geoms})
        self.near_joint = {" - ".join(sorted((model.body(a).name, model.body(b).name)))
                           for a in cb for b in cb
                           if a < b and (par[par[a]] == b or par[par[b]] == a)}
        # Pairs of robot shapes on links that are not parent and child: what can go into
        # each other (link_depths), with their bounding spheres to skip far pairs.
        cand = []
        for i in range(len(self.robot_geoms)):
            for k in range(i + 1, len(self.robot_geoms)):
                a, b = int(self.robot_geoms[i]), int(self.robot_geoms[k])
                ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
                if ba != bb and par[ba] != bb and par[bb] != ba:
                    cand.append((a, b))
        self.cand = np.array(cand, dtype=int).reshape(-1, 2)
        self.cand_name = [" - ".join(sorted((self.body_of[a], self.body_of[b]))) for a, b in cand]
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

    def disable_self_collision(self) -> None:
        """The robot's links no longer collide with each other, only with the floor (as in
        the deploy simulator of the self-collision benchmark): contype 2, conaffinity 1 on
        the robot's shapes; the floor keeps 1 and 1."""
        self.m.geom_contype[self.robot_geoms] = 2
        self.m.geom_conaffinity[self.robot_geoms] = 1

    def link_depths(self, min_depth: float = 0.002) -> dict[str, float]:
        """How deep (m) one link of the robot is into another, per pair of links, where
        deeper than min_depth: geometry (mj_geomDistance), so it works with self-collisions
        off. A link and its parent are not compared."""
        m, d = self.m, self.d
        if self.cand.size == 0:
            return {}
        a, b = self.cand[:, 0], self.cand[:, 1]
        gap = (np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b], axis=1)
               - m.geom_rbound[a] - m.geom_rbound[b])
        out: dict[str, float] = {}
        for k in np.flatnonzero(gap < 0.0):
            dep = -float(self.mj.mj_geomDistance(m, d, int(a[k]), int(b[k]), 0.0, None))
            if dep > min_depth:
                name = self.cand_name[k]
                out[name] = max(out.get(name, 0.0), dep)
        return out

    def clearances(self, names: set[str], distmax: float = 0.2) -> dict[str, float]:
        """Signed distance (m) per pair of links in `names`, the smallest over their shapes
        (negative: that deep), up to distmax."""
        out: dict[str, float] = {}
        for k, name in enumerate(self.cand_name):
            if name in names:
                dist = float(self.mj.mj_geomDistance(self.m, self.d, int(self.cand[k, 0]),
                                                     int(self.cand[k, 1]), distmax, None))
                out[name] = min(out.get(name, distmax), dist)
        return out

    def self_contact_pairs(self) -> set[str]:
        """Pairs of robot links touching now (geometry only, after mj_forward)."""
        out = set()
        for a, b in self._pairs().tolist():
            if a in self.robot_set and b in self.robot_set and self.body_of[a] != self.body_of[b]:
                out.add(" - ".join(sorted((self.body_of[a], self.body_of[b]))))
        return out

    def self_contact_forces(self) -> dict[str, float]:
        """Force (N) per pair of robot links pressing on each other in the last physics
        step. MuJoCo leaves out a link and its parent and the scene's excluded pairs."""
        out: dict[str, float] = {}
        for i, (a, b) in enumerate(self._pairs().tolist()):
            if a not in self.robot_set or b not in self.robot_set:
                continue
            ba, bb = self.body_of[a], self.body_of[b]
            if ba == bb:
                continue
            self.mj.mj_contactForce(self.m, self.d, i, self._f6)
            key = " - ".join(sorted((ba, bb)))
            out[key] = out.get(key, 0.0) + float(np.linalg.norm(self._f6[:3]))
        return out

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

    def _substeps(self, tau_fn, track: bool, episode: Episode | None = None,
                  standing: bool = False):
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
                if episode is not None:
                    add_self_contacts(episode, self.self_contact_forces(), self.m.opt.timestep,
                                      standing, self.near_joint)
        return head_on, head_f

    def step(self, q_des, episode: Episode | None = None, standing: bool = False):
        """One policy step (20 ms) of PD towards q_des. Returns (head touched, max force);
        the self-contacts of every physics step go into `episode` (getting up or standing)."""
        return self._substeps(lambda q, qd: self.kp * (q_des - q) - self.kd * qd, track=True,
                              episode=episode, standing=standing)

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

    def place_falling_forward(self, joint_q, rng, tilt_deg: float = 15.0, vel: float = 0.6):
        """Standing on its feet in this pose, tipping over forward: pitched tilt_deg +-5
        forward, moving forward at vel m/s and pitching at 2 vel rad/s; random heading
        (the benchmark's face-down fall)."""
        pitch = math.radians(tilt_deg + float(rng.uniform(-5.0, 5.0)))
        yaw = float(rng.uniform(-math.pi, math.pi))
        quat = quat_mul(quat_from_rpy(0.0, 0.0, yaw), quat_from_rpy(0.0, pitch, 0.0))
        self.place(quat, joint_q, clearance=0.002)
        self.d.qvel[self.rd:self.rd + 2] = (vel * math.cos(yaw), vel * math.sin(yaw))
        self.d.qvel[self.rd + 4] = 2.0 * vel             # free joint: body-frame angular velocity
        self.mj.mj_forward(self.m, self.d)

    def still(self, vel: float) -> bool:
        return float(np.abs(self.d.qvel[self.robot_dofs]).max()) < vel

    def finite(self) -> bool:
        return bool(np.all(np.isfinite(self.d.qpos)) and np.all(np.isfinite(self.d.qvel)))


def camera_azimuth(sim: Sim) -> float:
    """A three-quarter view across the body (deg): from the side of the line pelvis -> head
    when lying; when upright, of the facing direction (a fall forward lies along it)."""
    p = sim.d.qpos[sim.r:sim.r + 3]
    head = sim.d.xpos[sim.torso] + sim.d.xmat[sim.torso].reshape(3, 3) @ np.asarray(HEAD_OFFSET)
    v = np.asarray(head[:2] - p[:2], dtype=float)
    if float(np.linalg.norm(v)) < 0.15:
        v = quat_to_mat(sim.pelvis_quat())[:2, 0]
    return math.degrees(math.atan2(float(v[1]), float(v[0]))) - 60.0


class ProxyMeter:
    """The training's link-clearance measure (getup/clearance.py: stand-ins of this
    scene's shapes) on this robot now, to compare it with the exact one: pair -> m."""

    def __init__(self, sim: Sim):
        import torch
        from getup.clearance import LinkClearance
        self.torch, self.sim = torch, sim
        self.lc = LinkClearance()
        self.ids = [sim.m.body(b).id for b in self.lc.bodies]

    def __call__(self) -> dict[str, float]:
        t, d = self.torch, self.sim.d
        pos = t.tensor(d.xpos[self.ids][None], dtype=t.float32)
        rot = t.tensor(d.xmat[self.ids].reshape(1, -1, 3, 3), dtype=t.float32)
        return dict(zip(self.lc.names, self.lc.measure(pos, rot)[0].tolist()))


class Viewer:
    """The MuJoCo viewer, in real time: links in a conflict drawn red (Highlighter) and,
    if this MuJoCo has it, the header's text in the corners (else a label above the robot)."""

    def __init__(self, sim: Sim, speed: float, overlay: Overlay | None = None):
        import mujoco
        import mujoco.viewer
        self.mj, self.sim, self.ov = mujoco, sim, overlay
        self.v = mujoco.viewer.launch_passive(sim.m, sim.d)
        self.speed = max(speed, 1e-3)
        self.t0, self.n = time.time(), 0
        self.texts = overlay is not None and hasattr(self.v, "set_texts")

    def _show(self) -> None:
        ov, mj = self.ov, self.mj
        badge, _ = ov.badge()
        top1 = f"{ov.title}   {badge}" + (f"   t {ov.t:.1f} s" if ov.t is not None else "")
        top2 = ov.counter()
        now = "\n".join(ov.now_lines(" - "))
        if self.texts:
            try:
                big = int(mj.mjtFontScale.mjFONTSCALE_150)
                grid = mj.mjtGridPos
                self.v.set_texts([(big, int(grid.mjGRID_TOPLEFT), top1, top2 or " "),
                                  (big, int(grid.mjGRID_BOTTOMLEFT), now or " ", " ")])
                return
            except Exception:  # noqa: BLE001
                self.texts = False
        try:                     # older MuJoCo: one text label above the robot
            text = " | ".join(s for s in (top1, top2, now.replace("\n", "; ")) if s)[:99]
            p = self.sim.d.qpos[self.sim.r:self.sim.r + 3]
            with self.v.lock():
                scn = self.v.user_scn
                scn.ngeom = 0
                g = scn.geoms[0]
                mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                                np.array([p[0], p[1], max(float(p[2]), 0.3) + 0.9]),
                                np.eye(3).ravel(), np.array([1.0, 1.0, 1.0, 1.0], np.float32))
                g.label = text
                scn.ngeom = 1
        except Exception:  # noqa: BLE001
            self.ov = None

    def tick(self) -> bool:
        if not self.v.is_running():
            return False
        if self.ov is not None:
            self._show()
        self.v.sync()
        self.n += 1
        lag = self.t0 + self.n * DT / self.speed - time.time()
        if lag > 0:
            time.sleep(lag)
        return True


class VideoRecorder:
    """Draws the robot without a window (EGL, as in training) and saves, per episode, an
    mp4 (streamed to disk) and a sheet of 12 frames from the policy's first 3 s (left to
    right, top to bottom, 0.25 s apart). With an Overlay, each frame gets the header of
    the team's benchmark video (getup/sim2sim_video.py). The camera keeps one three-quarter
    view across the body per episode (aim()), so two policies from the same start are
    seen from the same place."""

    FPS = 25

    def __init__(self, sim: Sim, out_dir: str, width: int = 800, height: int = 600,
                 overlay: Overlay | None = None):
        import mujoco
        self.mj, self.sim, self.ov = mujoco, sim, overlay
        self.dir = Path(out_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        vis = sim.m.vis.global_
        vis.offwidth, vis.offheight = max(int(vis.offwidth), width), max(int(vis.offheight), height)
        self.r = mujoco.Renderer(sim.m, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.6, 135.0, -18.0
        self.sheet_at = {int(i * 0.25 * self.FPS): i for i in range(12)}
        self.writer: Mp4Writer | None = None
        self.mp4_ok = True
        self._reset()

    def _reset(self):
        self.n, self.count, self.policy_start = 0, 0, None
        self.sheet: dict[int, np.ndarray] = {}
        self.last = None

    def aim(self):
        try:
            self.cam.azimuth = camera_azimuth(self.sim)
        except Exception:  # noqa: BLE001
            pass

    def tick(self, every: int = 2):
        self.n += 1
        if self.n % every:
            return
        p = self.sim.d.qpos[self.sim.r:self.sim.r + 3]
        self.cam.lookat[:] = [p[0], p[1], max(float(p[2]), 0.4)]
        self.r.update_scene(self.sim.d, camera=self.cam)
        frame = self.r.render().copy()
        if self.ov is not None:
            try:
                frame = draw_frame(frame, self.ov)
            except Exception as e:  # noqa: BLE001
                print(f"  (no header on the video: {e})")
                self.ov = None
        if self.mp4_ok:
            try:
                if self.writer is None:
                    self.writer = Mp4Writer(tmp_name(self.dir), self.FPS)
                self.writer.add(frame)
            except Exception as e:  # noqa: BLE001
                print(f"  (no mp4: {e})")
                self.mp4_ok, self.writer = False, None
        if self.policy_start is not None:
            i = self.sheet_at.get(self.count - self.policy_start)
            if i is not None:
                self.sheet[i] = frame
        self.last = frame
        self.count += 1

    def mark_policy_start(self):
        self.policy_start = self.count

    def save(self, name: str) -> str:
        out = []
        if self.writer is not None:
            try:
                if self.writer.close():
                    self.writer.path.replace(self.dir / f"{name}.mp4")
                    out.append(f"{name}.mp4")
            except Exception as e:  # noqa: BLE001
                print(f"  (no mp4: {e})")
            self.writer = None
        if self.last is not None:
            frames = [self.sheet.get(i) for i in range(12)]
            for i in range(12):             # an episode shorter than 3 s: its last frame
                if frames[i] is None:
                    frames[i] = frames[i - 1] if i and frames[i - 1] is not None else self.last
            try:
                from PIL import Image
                if frames[0].shape[1] > 480:        # tiles 480 px wide: a sheet of 1920
                    k = 480 / frames[0].shape[1]
                    size = (480, int(round(frames[0].shape[0] * k)))
                    frames = [np.asarray(Image.fromarray(f).resize(size, Image.BILINEAR))
                              for f in frames]
            except Exception:  # noqa: BLE001
                pass
            rows = [np.concatenate(frames[j:j + 4], axis=1) for j in (0, 4, 8)]
            sheet = np.concatenate(rows, axis=0)
            try:
                from PIL import Image
                Image.fromarray(sheet).save(self.dir / f"{name}_sheet.png")
                out.append(f"{name}_sheet.png")
            except Exception:  # noqa: BLE001
                np.save(self.dir / f"{name}_sheet.npy", sheet)
                out.append(f"{name}_sheet.npy")
        self._reset()
        return ", ".join(out)


def video_size(text: str) -> tuple[int, int]:
    """'640x480' -> (640, 480): even numbers (the mp4 needs them)."""
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text or "")
    if not m:
        raise SystemExit(f"--video-size: expected WxH, e.g. 640x480 (got {text!r})")
    w, h = int(m.group(1)), int(m.group(2))
    if w < 160 or h < 120 or w % 2 or h % 2:
        raise SystemExit(f"--video-size: even numbers, at least 160x120 (got {text!r})")
    return w, h


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
    ap.add_argument("--start", choices=("pool", "collapse", "facedown"), default="pool",
                    help="pool: a pose of the pool; collapse: standing in the walking pose, "
                         "then limp until it lies still (as Passive in g1_ctrl); facedown: "
                         "tipping over forward from the walking pose, limp until still")
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
                         "drawn without a window (use it when the viewer window does not work); "
                         "a header shows the policy, the start, getting up / STOOD UP and, with "
                         "--conflicts or --benchmark, the link conflicts (those links red)")
    ap.add_argument("--video-size", default="800x600", metavar="WxH",
                    help="size of the video picture (the header band is added above it)")
    ap.add_argument("--label", default=None,
                    help="name of the policy in the videos and the viewer (default: the policy "
                         "folder's name, e.g. getup_v6, or the run folder's)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-self-collision", action="store_true",
                    help="the robot's links do not collide with each other (only with the floor)")
    ap.add_argument("--conflicts", action="store_true",
                    help="measure link conflicts: one link into another, deepest per pair")
    ap.add_argument("--conflict-cm", type=float, default=1.0,
                    help="a conflict is a link deeper than this into another (cm)")
    ap.add_argument("--benchmark", type=int, default=0, metavar="N",
                    help="the self-collision benchmark: N episodes each from supine, prone and "
                         "a face-down fall, links not colliding with each other, conflicts "
                         "measured; fails (exit 1) if any has a conflict or does not stand up")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.video:
        video_size(args.video_size)                    # a wrong size stops here, not later
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
    bench = args.benchmark > 0
    if bench:      # as test_getup_self_collision_benchmark: supine, prone, face-down fall
        specs = ([("pool", "supine")] * args.benchmark + [("pool", "prone")] * args.benchmark
                 + [("facedown", None)] * args.benchmark)
    else:
        specs = [(args.start, None)] * args.episodes
    nocoll = args.no_self_collision or bench
    measure = args.conflicts or bench
    if nocoll:
        sim.disable_self_collision()
    collapse = args.start == "collapse" and not bench
    pool = cols = None
    probs: dict = {}
    if any(kind == "pool" for kind, _ in specs):
        pool_path = resolve_pool(args.pool or opts.get("pool") or DEFAULT_POOL)
        pool = load_pool(pool_path)
        cols = [pool.joint_names.index(r.name) for r in rows]
        for post in sorted({post or "" for kind, post in specs if kind == "pool"}):
            pr = sample_weights(pool.posture, parse_postures(post or args.postures)
                                or DEFAULT_POSTURE_WEIGHTS)
            if args.released:
                try:
                    pr = pr * released_mask(pool, args.released)
                except ValueError as e:
                    raise SystemExit(str(e)) from None
            if pr.sum() <= 0:
                raise SystemExit(f"no pose of '{post or args.postures}'"
                                 + (f" released as '{args.released}'" if args.released else "")
                                 + f" in {pool_path}")
            probs[post or None] = pr / pr.sum()

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
    if bench:
        print(f"benchmark: {args.benchmark} episodes each from supine and prone (poses of "
              f"{pool_path}) and a face-down fall (limp, damping {limp_txt}); a conflict: one "
              f"link more than {args.conflict_cm:g} cm into another")
    elif args.start == "collapse":
        print(f"start: standing in the walking pose, then limp (damping {limp_txt}) until "
              "it lies still")
    elif args.start == "facedown":
        print(f"start: tipping over forward from the walking pose, limp (damping {limp_txt}) "
              "until it lies still")
    else:
        sel = f", released as {args.released}" if args.released else ""
        print(f"start poses: {len(pool)} from {pool_path}{sel}; settled limp (damping "
              f"{limp_txt})")
    if nocoll:
        print("links of the robot do NOT collide with each other (only with the floor), as in "
              "the deploy simulator of the self-collision benchmark")

    # Self-contact notes: links one joint apart, and pairs already touching in the walking
    # pose (standing, at rest): the shapes of this model, not something the policy does.
    sim.place(np.array([1.0, 0.0, 0.0, 0.0]), default_q, clearance=0.002)
    walk_touch = sim.self_contact_pairs()
    self_flags = {p: "near a joint" for p in sim.near_joint}
    for p in walk_touch:
        self_flags[p] = (self_flags[p] + ", " if p in self_flags else "") + "touches in the walking pose"
    if walk_touch:
        print("touching in the walking pose (this model's shapes): " + "; ".join(sorted(walk_touch)))
    baseline = sim.link_depths(0.0) if measure else {}
    if baseline:
        print("already overlapping in the walking pose (subtracted from the conflicts): "
              + "; ".join(f"{p} {100 * d:.1f} cm" for p, d in sorted(baseline.items())))
    lim = args.conflict_cm / 100.0
    parts = link_groups(set(sim.cand_name)) if measure else {}
    meter = None
    if measure:
        arm = {p for g in parts.values() for p in g}
        close = sorted(sim.clearances(arm).items(), key=lambda kv: kv[1])[:4]
        print("closest arm-body pairs in the walking pose: "
              + "; ".join(f"{pair_label(p)} {100 * d:.1f} cm" for p, d in close))
        try:
            meter = ProxyMeter(sim)
            from getup.clearance import check_scene
            note = check_scene(scene)
            if note:
                print(f"WARNING: {note}")
        except Exception as ex:  # noqa: BLE001
            print(f"(the training's link-clearance measure is not compared here: {ex})")

    # Pictures: the viewer and/or videos, with the header (policy, start, getting up /
    # STOOD UP, link conflicts) and the links of a conflict drawn red.
    label = args.label or policy_label(pol_path)
    show = bool(args.view or args.video)
    ov = Overlay(label=label, measure=measure, limit_cm=args.conflict_cm) if show else None
    hl = Highlighter(sim.m) if show and measure else None
    viewer = Viewer(sim, args.speed, ov) if args.view else None
    video = VideoRecorder(sim, args.video, *video_size(args.video_size), overlay=ov) if args.video else None
    scale = np.array([r.scale for r in rows])
    steps = int(round(args.time / DT))
    hold = STAND.hold_s
    eps: list[Episode] = []
    n_scen: dict = {}
    t_wall = time.time()
    for ep, (kind, post) in enumerate(specs):
        # 1. a start pose, settled in THIS scene with limp motors
        fall = kind in ("collapse", "facedown")
        if kind == "collapse":
            sim.place_standing(default_q, rng)
        elif kind == "facedown":
            sim.place_falling_forward(default_q, rng)
        else:
            for _ in range(20):
                i = int(rng.choice(len(pool), p=probs[post]))
                yaw = float(rng.uniform(-math.pi, math.pi))
                quat = quat_mul(quat_from_rpy(0.0, 0.0, yaw), pool.root_quat[i].astype(float))
                if sim.place(quat, pool.joint_pos[i, cols].astype(float)) < 0.02:
                    break
        scen = (post or kind) if bench else None
        title = ""
        if ov is not None:
            if bench:
                n_scen[scen] = n_scen.get(scen, 0) + 1
                title = (f"{label} · {SCENARIO_TEXT.get(scen, scen)} "
                         f"{n_scen[scen]}/{args.benchmark}")
            else:
                title = f"{label} · episode {ep + 1}/{len(specs)}"
            ov.new_episode(title)
        if video:
            video.aim()
        settle, still = 0.0, 0.0
        max_settle = max(args.settle, 6.0) if fall else args.settle
        min_settle = 1.0 if fall else 0.4              # a collapse needs time to start
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
        e = Episode(start=classify(sim.pelvis_quat()), settle_s=settle, peak_qd=np.zeros(nj),
                    scenario=(post or kind) if bench else "")
        e.self_at_start = max((f for p, f in sim.self_contact_forces().items()
                               if p not in sim.near_joint), default=0.0)
        start_z = float(sim.d.qpos[sim.r + 2])
        if measure:      # links already in each other (gone in while limp, before the policy)
            e.conflict_at_start = {p: dd - baseline.get(p, 0.0) for p, dd in sim.link_depths().items()
                                   if dd - baseline.get(p, 0.0) > 0.0}
            e.inherited = {p for p, dd in e.conflict_at_start.items()
                           if dd > lim and p not in sim.near_joint}
        if ov is not None:
            lies = POSTURES[e.start]
            if lies != scen:
                ov.title = f"{title} (lies {lies.replace('_', ' ')})"
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
            head_on, head_f = sim.step(default_q + scale * a, e, standing=stand_t > 0.0)
            last = a
            if not sim.finite() or np.abs(sim.d.qvel[sim.dadr]).max() > 60.0:
                e.blown = True
                break
            pz, hz, up = sim.stand_values()
            other, _, _, foot = sim.floor_contacts()
            standing = is_standing(pz, hz, up, other)
            stand_t = stand_t + DT if standing else 0.0
            if stand_t >= hold and not e.stood:
                e.stood, e.t_stand = True, (k + 1) * DT - hold
            over = {}
            if measure:    # every policy step, split into getting up and standing
                over = add_conflicts(e, sim.link_depths(), baseline, standing=standing, dt=DT,
                                     limit_m=lim, ignore=sim.near_joint, groups=parts)
                if meter is not None:
                    add_proxy(e, meter(), standing, lim, over)
            e.head_s += DT if head_on else 0.0
            e.head_peak = max(e.head_peak, head_f)
            e.peak_qd = np.maximum(e.peak_qd, np.abs(sim.d.qvel[sim.dadr]))
            if (k + 1) * DT > 0.1 and not other and not foot:     # as in training
                e.air_s += DT
            vz, w = sim.pelvis_speeds()
            e.peak_vz, e.peak_w = max(e.peak_vz, vz), max(e.peak_w, w)
            if ov is not None:
                ov.t = (k + 1) * DT
                ov.phase = ("stood" if e.stood and standing else "fell" if e.stood
                            else "standing" if standing else "up")
                ov.stood_at = e.t_stand if e.stood else None
                ov.active, ov.up_s, ov.stand_s = over, e.conflict_up_s, e.conflict_stand_s
            if hl is not None:
                hl.set(over)
            if viewer and not viewer.tick():
                viewer = None
            if video:
                video.tick()
        if hl is not None:
            hl.clear()
        e.success = stand_t >= hold and not e.blown
        e.end_pelvis = float(sim.d.qpos[sim.r + 2])
        e.end_pose_err = float(np.sqrt(np.mean((sim.d.qpos[sim.qadr] - default_q) ** 2)))
        e.end_dev = np.asarray(sim.d.qpos[sim.qadr] - default_q, dtype=float)
        eps.append(e)
        if video:
            saved = video.save(f"ep{ep + 1:02d}_{kind + '_' if fall else ''}"
                               f"{POSTURES[e.start]}")
            if saved:
                print(f"  saved {saved} in {video.dir}")
        state = ("stood up" if e.success else "fell again" if e.stood else "did not stand")
        conf = [f"{name} {pair_label(p)} {100 * dd:.1f} cm ({t_s:.2f} s)"
                for name, (p, dd), t_s in (
                    ("getting up", worst_pair(e.conflicts_up, sim.near_joint), e.conflict_up_s),
                    ("standing", worst_pair(e.conflicts_stand, sim.near_joint), e.conflict_stand_s))
                if dd > lim]
        print(f"  episode {ep + 1:3d}/{len(specs)}: {POSTURES[e.start]:10s} "
              f"(pelvis {start_z:4.2f} m) {state:13s}"
              + (f" at {e.t_stand:4.1f} s" if e.stood else "        ")
              + f"  head {e.head_s:4.2f} s, {e.head_peak:4.0f} N"
              + f"  self > 50 N {e.self_hard_s:4.2f} s, {e.self_peak:4.0f} N"
              + (f", in the air {e.air_s:.2f} s" if e.air_s > 0.0 else "")
              + ("  CONFLICT " + "; ".join(conf) if conf else "")
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
            + (", starts: face-down fall" if args.start == "facedown" and not bench else "")
            + (", benchmark starts: supine, prone, face-down fall" if bench else "")
            + (", links not colliding with each other" if nocoll else "")
            + (f", released as {args.released}" if args.released and not collapse else ""))
    print()
    lines = report(eps, groups, sat, f"sim2sim {pol_path.parent.name}/{pol_path.name}", mode,
                   self_flags)
    at = next((i + 1 for i, ln in enumerate(lines) if ln.startswith("at the end, standing")),
              len(lines))
    lines[at:at] = pose_report(eps, [r.name for r in rows])
    for line in lines:
        print(line)
    if measure:
        print()
        for line in conflict_report(eps, sim.near_joint, lim, collide=not nocoll):
            print(line)
        for line in group_report(eps, lim):
            print(line)
        if meter is not None:
            for line in proxy_report(eps, meter.lc.names, sim.near_joint, lim):
                print(line)
    if bench:
        def n_with(attr: str) -> int:
            return sum(worst_pair(getattr(e, attr), sim.near_joint)[1] > lim for e in eps)
        nconf = n_with("conflicts")
        rate = sum(e.success for e in eps) / len(eps)
        ok = nconf == 0 and rate == 1.0
        print(f"\nBENCHMARK: {nconf} of {len(eps)} scenarios with a conflict (getting up "
              f"{n_with('conflicts_up')}, standing {n_with('conflicts_stand')}; without what the "
              f"start already had: getting up {n_with('new_up')}, standing {n_with('new_stand')}), "
              f"stand rate {rate:.2f}: {'PASS' if ok else 'FAIL'}")
        if args.video:
            print(f"videos: {Path(args.video).expanduser()}  (two runs side by side: "
                  "scripts/side_by_side.py LEFT_DIR RIGHT_DIR)")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
