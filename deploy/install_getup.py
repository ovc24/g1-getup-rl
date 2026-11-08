#!/usr/bin/env python3
"""Add the get-up policy to g1_ctrl (Unitree's deploy controller), next to the walking one.

    # on the PC (needs onnxruntime: the training venv has it)
    python install_getup.py --deploy <unitree_rl_mjlab>/deploy \
        --getup-run <unitree_rl_mjlab>/logs/rsl_rl/g1_23dof_getup/<run> --auto
    # on the robot: the policy folder made on the PC (<deploy>/robots/g1_23dof/config/
    # policy/getup/v0), copied over (plain python3 is enough)
    python3 install_getup.py --deploy <robot>/deploy --policy-dir <copied>/v0
    # then, in both places:
    cd <deploy>/robots/g1_23dof/build && make -j4

What it changes in <deploy> (every file it touches is first copied to <file>.orig, once;
running it again changes nothing that is already done):
  robots/g1_23dof/include/State_GetUp.h, State_Fallen.h   (new FSM states, next to this
      script in g1_ctrl/)
  robots/g1_23dof/main.cpp       includes the two states
  robots/g1_23dof/src/State_RLBase.cpp
                                 where to go on a fall: config "on_fall" (default Passive)
  include/isaaclab/envs/mdp/terminations.h
                                 bad_orientation really checks the tilt (Unitree's returns
                                 false: without this no fall is ever detected)
  include/FSM/State_RLBase.h     the gains go to the motors of the policy joints
                                 (joint_ids_map); Unitree's writes them to motors 0..22
  robots/g1_23dof/config/config.yaml
      Velocity: on_fall: Fallen (walking -> fall detected -> Fallen)
      Fallen  (new): motors limp until the robot lies still; then GetUp, by itself
                     (auto_getup: true, --auto) or with RT + X (--manual)
      GetUp   (new): motors limp until the robot lies still (0.5 s, at most 3 s), then
                     the get-up policy; standing still in the walking pose -> Velocity;
                     not up after 10 s -> Fallen
      Passive: RT + X -> GetUp as well (the robot already lying when g1_ctrl starts)
  robots/g1_23dof/config/policy/getup/v0/exported/policy.onnx
  robots/g1_23dof/config/policy/getup/v0/params/deploy.yaml
      made from the policy's own metadata (joint order, default pose, action scale) and
      the walking policy's deploy.yaml (joint map, PD gains: the get-up was trained with
      the same ones; mjlab's metadata cannot give them for its DC-motor model): 5 inputs x
      5 steps (375 numbers), joint targets = walking pose + scale x action. Its first
      lines say which policy file it came from (path, md5).
Running it again with another --getup-run replaces the policy and copies the two states
again (a newer State_GetUp.h too); then build again.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATES = ("State_GetUp.h", "State_Fallen.h")

# Motor order of the G1 (unitree_sdk2 low-level command, 29 motors).
SDK_INDEX = {n: i for i, n in enumerate([
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
])}
# Get-up policy inputs (getup_env_cfg.make_observations, actor group) -> g1_ctrl names.
OBS_TERMS = (("base_ang_vel", "base_ang_vel", 3), ("projected_gravity", "projected_gravity", 3),
             ("joint_pos", "joint_pos_rel", None), ("joint_vel", "joint_vel_rel", None),
             ("actions", "last_action", None))

CONFIG_STATES = """    GetUp:
      id: 4
      type: GetUp
    Fallen:
      id: 6
      type: Fallen
"""
CONFIG_BLOCKS = """
  Fallen:
    transitions:
      Passive: LT + B.on_pressed
      GetUp: RT + X.on_pressed
    auto_getup: {auto}
    settle_s: 1.0
    still_gyro: 0.5
    still_joint_vel: 1.0
    getup_state: GetUp

  GetUp:
    transitions:
      Passive: LT + B.on_pressed
    policy_dir: config/policy/getup
    settle_s: 0.5
    max_settle_s: 3.0
    min_time_s: 2.0
    hold_s: 0.5
    max_time_s: 10.0
    max_tilt: 0.35
    max_gyro: 0.8
    max_leg_dev: 0.5
    on_success: Velocity
    on_timeout: Fallen
"""


class Edit:
    """Collects the new contents first; writes only when every step succeeded."""

    def __init__(self):
        self.files: dict[Path, str] = {}
        self.notes: list[str] = []

    def text(self, path: Path) -> str:
        if path not in self.files:
            if not path.is_file():
                raise SystemExit(f"not found: {path}")
            self.files[path] = path.read_text()
        return self.files[path]

    def set(self, path: Path, new: str, note: str) -> None:
        self.files[path] = new
        self.notes.append(note)

    def write(self) -> None:
        for path, new in self.files.items():
            if path.is_file() and path.read_text() == new:
                continue
            orig = path.with_name(path.name + ".orig")
            if path.is_file() and not orig.exists():
                shutil.copy2(path, orig)
            path.write_text(new)


def patch_main(ed: Edit, robot: Path) -> None:
    p = robot / "main.cpp"
    t = ed.text(p)
    if all(f'#include "{s}"' in t for s in STATES):
        ed.notes.append("main.cpp: already includes the states")
        return
    add = "".join(f'#include "{s}"\n' for s in STATES if f'#include "{s}"' not in t)
    anchor = '#include "State_Mimic.h"\n'
    if anchor in t:
        t = t.replace(anchor, anchor + add, 1)
    else:
        last = list(re.finditer(r'^#include .*\n', t, re.M))
        if not last:
            raise SystemExit(f"{p}: no #include lines")
        t = t[:last[-1].end()] + add + t[last[-1].end():]
    ed.set(p, t, "main.cpp: includes State_GetUp.h and State_Fallen.h")


def patch_rlbase_cpp(ed: Edit, robot: Path) -> None:
    p = robot / "src" / "State_RLBase.cpp"
    t = ed.text(p)
    if "on_fall" in t:
        ed.notes.append("State_RLBase.cpp: already has on_fall")
        return
    pat = re.compile(r'(?P<indent>[ \t]*)this->registered_checks\.emplace_back\(\s*std::make_pair\(\s*'
                     r'\[&\]\(\)->bool\{\s*return isaaclab::mdp::bad_orientation\(env\.get\(\), '
                     r'1\.0\);\s*\},\s*FSMStringMap\.right\.at\("Passive"\)')
    m = pat.search(t)
    if not m:
        raise SystemExit(f"{p}: the bad_orientation -> Passive check was not found (a different "
                         "version of Unitree's code?). Nothing was changed.")
    ind = m.group("indent")
    pre = (f'{ind}// where to go when the robot falls (config "on_fall", default Passive)\n'
           f'{ind}std::string on_fall = cfg["on_fall"] ? cfg["on_fall"].as<std::string>() : std::string("Passive");\n'
           f'{ind}if(!FSMStringMap.right.count(on_fall)) {{\n'
           f'{ind}    spdlog::warn("State_{{}}: on_fall state \'{{}}\' not found, using Passive", state_string, on_fall);\n'
           f'{ind}    on_fall = "Passive";\n'
           f'{ind}}}\n')
    block = m.group(0).replace('FSMStringMap.right.at("Passive")', 'FSMStringMap.right.at(on_fall)')
    t = t[:m.start()] + pre + block + t[m.end():]
    ed.set(p, t, "State_RLBase.cpp: on a fall -> config on_fall (default Passive)")


def patch_terminations(ed: Edit, deploy: Path) -> None:
    p = deploy / "include" / "isaaclab" / "envs" / "mdp" / "terminations.h"
    t = ed.text(p)
    live = re.search(r'^\s*return std::fabs\(std::acos\(-data\[2\]\)\) > limit_angle;', t, re.M)
    dead = re.search(r'^\s*return false;\s*$', t, re.M)
    if live and not dead:
        ed.notes.append("terminations.h: bad_orientation already checks the tilt")
        return
    t2 = re.sub(r'^(\s*)//\s*(return std::fabs\(std::acos\(-data\[2\]\)\) > limit_angle;)',
                r'    \2', t, count=1, flags=re.M)
    t2 = re.sub(r'^\s*return false;\s*\n', '', t2, count=1, flags=re.M)
    if not re.search(r'^\s*return std::fabs\(std::acos\(-data\[2\]\)\) > limit_angle;', t2, re.M):
        raise SystemExit(f"{p}: could not make bad_orientation check the tilt. Nothing was changed.")
    ed.set(p, t2, "terminations.h: bad_orientation checks the tilt (> 1 rad = fallen)")


def patch_rlbase_h(ed: Edit, deploy: Path) -> None:
    p = deploy / "include" / "FSM" / "State_RLBase.h"
    t = ed.text(p)
    if "joint_ids_map[i]" in t:
        ed.notes.append("State_RLBase.h: gains already go to the policy joints' motors")
        return
    old = "lowcmd->msg_.motor_cmd()[i]."
    if t.count(old) < 4:
        raise SystemExit(f"{p}: the gain loop of enter() was not found. Nothing was changed.")
    t = t.replace(old, "lowcmd->msg_.motor_cmd()[static_cast<int>(env->robot->data.joint_ids_map[i])].")
    ed.set(p, t, "State_RLBase.h: gains to the motors of the policy joints (joint_ids_map)")


def patch_rlbase_first_step(ed: Edit, deploy: Path) -> None:
    """enter() starts the policy thread, and until its first step run() writes the targets
    left from the last time (on a first start: 0 rad for every joint) for a few ms. One
    step in enter() first. Optional: skipped with a note if the code looks different."""
    p = deploy / "include" / "FSM" / "State_RLBase.h"
    t = ed.text(p)
    if "first targets before the thread" in t:
        ed.notes.append("State_RLBase.h: already computes the first targets in enter()")
        return
    m = re.search(r'^(?P<i>[ \t]*)env->robot->update\(\);\n(?=\s*// Start policy thread)', t, re.M)
    if not m:
        ed.notes.append("State_RLBase.h: (skipped the first-targets change: code differs)")
        return
    i = m.group("i")
    add = f"{i}env->reset();\n{i}env->step();  // first targets before the thread starts\n"
    ed.set(p, t[:m.end()] + add + t[m.end():],
           "State_RLBase.h: the first walking targets are computed in enter()")


def patch_config(ed: Edit, robot: Path, auto: bool | None) -> None:
    p = robot / "config" / "config.yaml"
    t = ed.text(p)
    done = []
    if "type: GetUp" not in t:
        anchor = re.search(r'^      type: RLBase\n', t, re.M)
        if not anchor:
            raise SystemExit(f"{p}: the Velocity entry (type: RLBase) of the FSM list was not "
                             "found. Nothing was changed.")
        t = t[:anchor.end()] + CONFIG_STATES + t[anchor.end():]
        done.append("states GetUp (4) and Fallen (6)")
    if "GetUp: RT + X.on_pressed" not in t.split("\n  FixStand:")[0]:
        anchor = re.search(r'^      FixStand: LT \+ up\.on_pressed\n', t, re.M)
        if not anchor:
            raise SystemExit(f"{p}: the Passive -> FixStand transition was not found. "
                             "Nothing was changed.")
        t = t[:anchor.end()] + "      GetUp: RT + X.on_pressed\n" + t[anchor.end():]
        done.append("Passive: RT + X -> GetUp")
    if "on_fall:" not in t:
        anchor = re.search(r'^    policy_dir: config/policy/velocity\s*\n', t, re.M)
        if not anchor:
            raise SystemExit(f"{p}: the Velocity policy_dir line was not found. Nothing was changed.")
        t = t[:anchor.end()] + "    on_fall: Fallen\n" + t[anchor.end():]
        done.append("Velocity: on_fall: Fallen")
    if not re.search(r'^  GetUp:\s*$', t, re.M):
        if not t.endswith("\n"):
            t += "\n"
        t += CONFIG_BLOCKS.format(auto="true" if auto else "false")
        done.append(f"Fallen and GetUp settings (auto_getup: {'true' if auto else 'false'})")
    elif auto is not None:
        t2 = re.sub(r'^(    auto_getup: )(true|false)\s*$', r'\g<1>' + ("true" if auto else "false"),
                    t, count=1, flags=re.M)
        if t2 != t:
            t = t2
            done.append(f"auto_getup: {'true' if auto else 'false'}")
    if done:
        ed.set(p, t, "config.yaml: " + "; ".join(done))
    else:
        ed.notes.append("config.yaml: already set up")


def _csv(meta: dict, key: str) -> list[str] | None:
    v = meta.get(key)
    return None if v is None else [s.strip() for s in str(v).split(",")]


def _floats(meta: dict, key: str) -> list[float] | None:
    v = _csv(meta, key)
    return None if v is None else [float(x) for x in v]


def onnx_path(src: Path) -> Path:
    src = src.expanduser()
    if src.is_dir():
        return src / "policy.onnx"
    if src.suffix == ".pt":
        print(f"[INFO] {src.name}: using {src.parent / 'policy.onnx'} (written by the last save)")
        return src.parent / "policy.onnx"
    return src


def make_policy_dir(onnx: Path, walk_yaml: Path, out: Path) -> list[str]:
    """policy.onnx + params/deploy.yaml for g1_ctrl, from the policy's metadata."""
    try:
        import onnxruntime as ort
        import yaml
    except ImportError as e:
        raise SystemExit(f"{e.name} is missing: run this on the PC, in the training venv "
                         "(or use --policy-dir with a folder made there)") from None
    if not onnx.is_file():
        raise SystemExit(f"no policy file: {onnx}")
    walk = yaml.safe_load(walk_yaml.read_text())
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    meta = dict(sess.get_modelmeta().custom_metadata_map)
    inp, outp = sess.get_inputs(), sess.get_outputs()
    notes = []

    names = _csv(meta, "joint_names")
    if not names:
        raise SystemExit(f"{onnx}: no joint_names in its metadata")
    n = len(names)
    missing = [x for x in names if x not in SDK_INDEX]
    if missing:
        raise SystemExit(f"joints without a G1 motor: {missing}")
    ids = [SDK_INDEX[x] for x in names]
    if ids != [int(x) for x in walk["joint_ids_map"]]:
        raise SystemExit(f"joint order differs from the walking policy:\n  get-up  {ids}\n"
                         f"  walking {walk['joint_ids_map']}")

    if len(inp) != 1 or inp[0].name != "obs":
        raise SystemExit(f"g1_ctrl feeds one input named 'obs'; this policy has "
                         f"{[i.name for i in inp]}")
    n_in = inp[0].shape[-1]
    n_out = outp[0].shape[-1]
    if n_out != n:
        raise SystemExit(f"the policy gives {n_out} actions for {n} joints")
    obs_names = _csv(meta, "observation_names")
    expected = [o for o, _, _ in OBS_TERMS]
    if obs_names and obs_names != expected:
        raise SystemExit(f"inputs differ from the get-up config:\n  policy {obs_names}\n"
                         f"  expected {expected}")
    per_step = sum(d if d else n for _, _, d in OBS_TERMS)
    if not isinstance(n_in, int) or n_in % per_step:
        raise SystemExit(f"input size {n_in} is not a multiple of {per_step} (one step)")
    history = n_in // per_step

    default = _floats(meta, "default_joint_pos")
    walk_default = [float(x) for x in walk["default_joint_pos"]]
    if default is None:
        default = walk_default
        notes.append("default pose: the walking one (not in the policy metadata)")
    elif max(abs(a - b) for a, b in zip(default, walk_default)) > 2e-3:
        raise SystemExit("the get-up default pose differs from the walking pose: the hand-over "
                         f"assumes the same one\n  get-up  {default}\n  walking {walk_default}")
    scale = _floats(meta, "action_scale")
    if scale is None or len(scale) != n:
        raise SystemExit(f"{onnx}: no action_scale (one per joint) in its metadata")
    ratio = [s / float(w) for s, w in zip(scale, walk["actions"]["JointPositionAction"]["scale"])]
    notes.append(f"action scale {min(ratio):.2f}-{max(ratio):.2f} x the walking one")
    if not 1.5 <= min(ratio) <= max(ratio) <= 2.5:
        notes.append("WARNING the get-up action scale is usually 2 x the walking one "
                     "(0.5 instead of 0.25 x peak torque / stiffness)")
    # PD gains: the walking ones. The get-up policy was trained with the same (Unitree's)
    # stiffness and damping, inside a DC-motor model that mjlab computes itself; its
    # MuJoCo actuators are plain motors, so the metadata says stiffness 1 and damping 0.
    gains = {k: [float(x) for x in walk[k]] for k in ("stiffness", "damping")}
    if min(gains["stiffness"]) < 5.0:
        raise SystemExit(f"{walk_yaml}: suspicious walking stiffness {gains['stiffness']}")
    meta_kp = _floats(meta, "joint_stiffness")
    if meta_kp and len(meta_kp) == n and min(meta_kp) >= 5.0:
        worst = max(abs(a - b) / b for a, b in zip(meta_kp, gains["stiffness"]))
        if worst > 0.02:
            notes.append(f"WARNING the policy's metadata stiffness differs from the walking one "
                         f"by up to {worst:.0%}; using the walking one")
    notes.append("gains: the walking ones, kp " + ", ".join(
        f"{x:g}" for x in sorted(set(round(v, 1) for v in gains["stiffness"]))))

    def flow(xs, fmt="{:.6g}") -> str:
        return "[" + ", ".join(fmt.format(x) for x in xs) + "]"

    lines = [
        "# g1_ctrl config of the get-up policy (made by install_getup.py; gains: the walking ones)",
        f"# from {onnx.resolve()} (md5 {hashlib.md5(onnx.read_bytes()).hexdigest()})",
        f"joint_ids_map: {flow(ids, '{}')}",
        f"step_dt: {float(walk['step_dt'])}",
        f"stiffness: {flow(gains['stiffness'])}",
        f"damping: {flow(gains['damping'])}",
        f"default_joint_pos: {flow(default)}",
        "actions:",
        "  JointPositionAction:",
        "    clip: null",
        "    joint_names: [.*]",
        f"    scale: {flow(scale)}",
        f"    offset: {flow(default)}",
        "    joint_ids: null",
        "observations:",
    ]
    for _, name, dim in OBS_TERMS:
        d = dim if dim else n
        lines += [f"  {name}:", "    params: {}", "    clip: null",
                  f"    scale: {flow([1.0] * d, '{:.1f}')}", f"    history_length: {history}"]
    (out / "params").mkdir(parents=True, exist_ok=True)
    (out / "exported").mkdir(parents=True, exist_ok=True)
    (out / "params" / "deploy.yaml").write_text("\n".join(lines) + "\n")
    for f in onnx.parent.glob(onnx.name + "*"):            # policy.onnx (+ policy.onnx.data)
        shutil.copy2(f, out / "exported" / f.name.replace(onnx.stem, "policy", 1))
    notes.insert(0, f"policy: {n_in} inputs = {per_step} x {history} steps, {n} joints; "
                    f"from {onnx}")
    return notes


def copy_policy_dir(src: Path, out: Path) -> Path:
    """g1_ctrl's part of a policy folder (exported/, params/deploy.yaml) -> out. A relative
    policy/<name> not found from here is taken from this repo."""
    src = src.expanduser()
    if not src.is_absolute() and not src.exists() and (HERE.parent / src).exists():
        src = HERE.parent / src
    for part in ("exported/policy.onnx", "params/deploy.yaml"):
        if not (src / part).is_file():
            raise SystemExit(f"{src} has no {part}")
    if src.resolve() == out.resolve():
        return src
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src / "exported", out / "exported")
    (out / "params").mkdir(parents=True)
    shutil.copy2(src / "params" / "deploy.yaml", out / "params" / "deploy.yaml")
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", required=True, help="the deploy folder of unitree_rl_mjlab")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--getup-run", help="get-up run folder, its policy.onnx or a model_<N>.pt")
    src.add_argument("--policy-dir", help="a getup policy folder already made (exported/, params/)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--auto", action="store_true", help="get up by itself after a fall")
    mode.add_argument("--manual", action="store_true", help="get up only with RT + X (default)")
    args = ap.parse_args()

    deploy = Path(args.deploy).expanduser().resolve()
    robot = deploy / "robots" / "g1_23dof"
    if not (robot / "main.cpp").is_file():
        raise SystemExit(f"{deploy} does not look like unitree_rl_mjlab/deploy "
                         "(no robots/g1_23dof/main.cpp)")
    for s in STATES:
        if not (HERE / "g1_ctrl" / s).is_file():
            raise SystemExit(f"missing {HERE / 'g1_ctrl' / s}")

    ed = Edit()
    patch_main(ed, robot)
    patch_rlbase_cpp(ed, robot)
    patch_terminations(ed, deploy)
    patch_rlbase_h(ed, deploy)
    patch_rlbase_first_step(ed, deploy)
    patch_config(ed, robot, True if args.auto else False if args.manual else None)

    policy_out = robot / "config" / "policy" / "getup" / "v0"
    policy_notes = []
    if args.getup_run:
        walk_yaml = robot / "config" / "policy" / "velocity" / "v0" / "params" / "deploy.yaml"
        policy_notes = make_policy_dir(onnx_path(Path(args.getup_run)), walk_yaml, policy_out)
    elif args.policy_dir:
        src_dir = copy_policy_dir(Path(args.policy_dir), policy_out)
        policy_notes = [f"policy folder copied from {src_dir}"]
    elif not (policy_out / "exported" / "policy.onnx").is_file():
        print("[WARNING] no get-up policy yet: give --getup-run (PC) or --policy-dir (robot)")

    ed.write()
    for s in STATES:
        shutil.copy2(HERE / "g1_ctrl" / s, robot / "include" / s)
    print(f"[OK] {deploy}")
    print(f"  include/State_GetUp.h, State_Fallen.h copied to {robot / 'include'}")
    for note in ed.notes + policy_notes:
        print(f"  {note}")
    if (policy_out / "exported" / "policy.onnx").is_file():
        print(f"  get-up policy: {policy_out}")
    print(f"\nNow build:  cd {robot / 'build'} && make -j4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
