#!/usr/bin/env python3
"""Copy a trained get-up policy into this repo, ready to publish and to run elsewhere
(g1_ctrl, Unitree's simulator, another simulator, the robot):

    cd <unitree_rl_mjlab> && source .venv/bin/activate
    python <this repo>/scripts/import_policy.py --run logs/rsl_rl/g1_23dof_getup/<run> --name getup_v5
    bash <this repo>/scripts/check_public.sh

It writes policy/<name>/:
    exported/policy.onnx         the network as g1_ctrl runs it, one file; its metadata
                                 keeps joints, default pose, action scale, inputs, gets the
                                 PD gains it was trained with (mjlab writes 1 and 0) and no
                                 path of this computer (run_path -> the run folder's name)
    params/deploy.yaml           its g1_ctrl config (deploy/install_getup.py --policy-dir)
    params/getup_options.json    what eval_getup.py / play_getup.py / sim2sim_getup.py need
                                 to rebuild the same environment (paths -> file names)
    model_<N>.pt                 the PyTorch checkpoint of the same save (eval, fine-tuning)
    README.md                    inputs, outputs, joints, gains: how to run it anywhere

The joint order and the PD gains come from g1_ctrl's walking deploy.yaml (--deploy, default
./deploy), as in deploy/install_getup.py. Needs onnx and onnxruntime (the training venv).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "deploy"))

OBS_DOC = {
    "base_ang_vel": ("3", "pelvis angular velocity, rad/s, pelvis frame (the IMU gyro)"),
    "projected_gravity": ("3", "gravity direction in the pelvis frame (unit vector; (0, 0, -1) upright)"),
    "joint_pos_rel": ("23", "joint angles minus the default pose below, rad"),
    "joint_vel_rel": ("23", "joint speeds, rad/s"),
    "last_action": ("23", "the network's previous output (0 at the start)"),
}


def last_checkpoint(run: Path) -> Path:
    def it(f: Path) -> int:
        m = re.search(r"(\d+)$", f.stem)
        return int(m.group(1)) if m else -1
    ckpts = sorted(run.glob("model_*.pt"), key=it)
    if not ckpts:
        raise SystemExit(f"no model_<N>.pt in {run}")
    return ckpts[-1]


def private_words() -> list[str]:
    f = REPO / "notes" / "private_words.txt"
    if not f.is_file():
        return []
    return [w.strip().lower() for w in f.read_text().splitlines()
            if w.strip() and not w.strip().startswith("#")]


def clean_value(key: str, value: str, run_name: str) -> str:
    """No local paths in what gets published: a path keeps only its last part."""
    if key == "run_path":
        return run_name
    if value.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", value):
        return Path(value).name
    return value


def clean_onnx(src: Path, dst: Path, run_name: str) -> list[str]:
    try:
        import onnx
    except ImportError:
        raise SystemExit("onnx is missing: run this in the training venv") from None
    model = onnx.load(str(src))                     # also loads policy.onnx.data, if any
    changed = []
    meta = [(p.key, p.value) for p in model.metadata_props]
    del model.metadata_props[:]
    for k, v in meta:
        nv = clean_value(k, v, run_name)
        if nv != v:
            changed.append(f"metadata {k}: '{v}' -> '{nv}'")
        e = model.metadata_props.add()
        e.key, e.value = k, nv
    docs = sum(1 for n in model.graph.node if n.doc_string)
    for n in model.graph.node:
        n.doc_string = ""
    if model.doc_string or model.graph.doc_string or docs:
        changed.append(f"doc strings removed ({docs} nodes)")
    model.doc_string = ""
    model.graph.doc_string = ""
    onnx.save_model(model, str(dst), save_as_external_data=False)
    return changed


def set_gains(path: Path, kp: list, kd: list) -> list[str]:
    """mjlab writes the MuJoCo actuator gains into joint_stiffness / joint_damping: 1 and 0
    for a policy trained with its DC-motor model (plain-motor actuators, the PD computed by
    mjlab). Write the gains it was trained with (deploy.yaml's), so that a tool taking the
    gains from the metadata does not run the motors limp."""
    import onnx
    model = onnx.load(str(path))
    want = {"joint_stiffness": ",".join(f"{float(x):.6g}" for x in kp),
            "joint_damping": ",".join(f"{float(x):.6g}" for x in kd)}
    changed, seen = [], set()
    for p in model.metadata_props:
        if p.key in want:
            seen.add(p.key)
            if p.value != want[p.key]:
                changed.append(f"metadata {p.key}: '{p.value[:24]}...' -> the training gains "
                               f"'{want[p.key][:24]}...'")
                p.value = want[p.key]
    for k in sorted(want.keys() - seen):
        e = model.metadata_props.add()
        e.key, e.value = k, want[k]
        changed.append(f"metadata {k}: added (the training gains)")
    onnx.save_model(model, str(path), save_as_external_data=False)
    return changed


def clean_options(src: Path, run_name: str) -> dict:
    opts = json.loads(src.read_text()) if src.is_file() else {}
    for k, v in list(opts.items()):
        if isinstance(v, str) and ("/" in v or "\\" in v):
            name = Path(v).name
            opts[k] = f"data/{name}" if k == "pool" else name
    opts["source_run"] = run_name
    return opts


def policy_readme(name: str, run_name: str, ckpt: Path, deploy: dict, joint_names: list[str],
                  obs_names: list[str], history: int, md5: str, results: str,
                  io: tuple[str, str] = ("obs", "actions")) -> str:
    kp, kd = deploy["stiffness"], deploy["damping"]
    scale = deploy["actions"]["JointPositionAction"]["scale"]
    default = deploy["default_joint_pos"]
    n = len(joint_names)
    per_step = sum(int(OBS_DOC[o][0]) if o in OBS_DOC else 0 for o in obs_names)
    rows = "\n".join(f"| {i} | `{j}` | {default[i]:+.3f} | {scale[i]:.4g} | {kp[i]:g} | {kd[i]:g} |"
                     for i, j in enumerate(joint_names))
    obs_rows = "\n".join(f"   | {i + 1} | `{o}` | {OBS_DOC.get(o, ('?', '?'))[0]} | {OBS_DOC.get(o, ('?', '?'))[1]} |"
                         for i, o in enumerate(obs_names))
    terms = [deploy["observations"][o] for o in obs_names]
    plain = all(tm.get("clip") is None and all(float(s) == 1.0 for s in (tm.get("scale") or [1.0]))
                for tm in terms)
    prep = ("No scaling, no clipping." if plain else
            "Each signal scaled and clipped as `params/deploy.yaml` says (`scale`, `clip`).")
    return f"""# {name}: get-up policy for the Unitree G1-23DoF

From training run `{run_name}`, checkpoint `{ckpt.name}`; `exported/policy.onnx` md5 `{md5}`.
{results}
## Files

| File | Use |
|---|---|
| `exported/policy.onnx` | the network (one input `{io[0]}`, one output `{io[1]}`): g1_ctrl, any ONNX runtime |
| `params/deploy.yaml` | its g1_ctrl config (`deploy/install_getup.py --policy-dir policy/{name}`) |
| `params/getup_options.json` | the environment it was trained in, for `eval_getup.py`, `play_getup.py`, `sim2sim_getup.py` |
| `{ckpt.name}` | PyTorch checkpoint (actor, critic, optimizer): `eval_getup.py --checkpoint policy/{name}/{ckpt.name}`, fine-tuning |

With this repo and unitree_rl_mjlab (main README, sections 1 and 6), from the unitree_rl_mjlab
folder with its venv:

```bash
python <this repo>/deploy/install_getup.py --deploy deploy --policy-dir policy/{name} --manual   # into g1_ctrl
python <this repo>/sim2sim_getup.py --policy policy/{name} --start collapse --episodes 20       # Unitree's MuJoCo model
python <this repo>/sim2sim_scenario.py --getup policy/{name} --out videos/                      # scenario video
```

## Running it anywhere

Every 20 ms (50 Hz):

1. Build the input `{io[0]}` (shape 1 x {per_step * history}): {len(obs_names)} signals, each over the last
   {history} steps, **one signal after the other, oldest step first** (all {history} steps of signal 1,
   then all {history} of signal 2, ...). {prep} When the policy starts, every
   step of the history is the first reading, and the last action is 0.

   | # | signal | size | what |
   |---|---|---|---|
{obs_rows}

2. `{io[1]} = policy({io[0]})` (shape 1 x {n}, joint order below).
3. Joint targets: `q_target = default + scale * {io[1]}`, held for the 20 ms and tracked by
   the motors' PD, `tau = kp * (q_target - q) - kd * dq` with the gains below (in training
   at 200 Hz, 4 physics steps per policy step; the robot's motors run it faster).

The ONNX metadata has the same values (`joint_names`, `default_joint_pos`, `action_scale`,
`joint_stiffness`, `joint_damping`), for tools that read them from there.

Common mistakes (any of them can make it fail):

- the history step by step instead of signal by signal, or newest first;
- angular velocity or gravity in the world frame, or from another body than the pelvis
  (the robot's root body);
- joints in the simulator's or the SDK's order instead of the order of the table;
- other PD gains (e.g. kp 1 and kd 0, what mjlab writes into the metadata of such a
  policy), or the PD only once per 20 ms instead of every physics step.

Without the motors' torque limits it may still get up in simulation, in a way the robot
cannot.

Start it with the robot lying still (g1_ctrl's GetUp state waits for that), and hand over to
the walking policy once it stands still in the walking pose (the default pose below), as
`deploy/g1_ctrl/State_GetUp.h` does. Trained with the motors' torque-speed curve and
Unitree's peak torques (88/139/50/25 N m by joint type). On the real robot, start as
`deploy/README.md` says (On the robot): on a mat, under the gantry with the rope slack,
started by hand, the damping state at hand.

| # | joint | default (rad) | scale | kp | kd |
|---|---|---|---|---|---|
{rows}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the training run folder")
    ap.add_argument("--name", required=True, help="folder name under policy/, e.g. getup_v5")
    ap.add_argument("--checkpoint", default=None, help="model_<N>.pt (default: the last one)")
    ap.add_argument("--deploy", default="deploy", help="g1_ctrl's deploy folder (walking deploy.yaml)")
    ap.add_argument("--results", default="", help="a line about how it does, for its README")
    args = ap.parse_args()

    run = Path(args.run).expanduser().resolve()
    ckpt = Path(args.checkpoint).expanduser() if args.checkpoint else last_checkpoint(run)
    src_onnx = run / "policy.onnx"
    if not src_onnx.is_file():
        raise SystemExit(f"no {src_onnx}")
    gap = src_onnx.stat().st_mtime - ckpt.stat().st_mtime
    if not 0.0 <= gap < 120.0:
        raise SystemExit(f"{src_onnx.name} was not written by the save of {ckpt.name} (every save "
                         "writes model_<N>.pt and then policy.onnx): import the last checkpoint")
    walk_yaml = (Path(args.deploy).expanduser() / "robots" / "g1_23dof" / "config" / "policy"
                 / "velocity" / "v0" / "params" / "deploy.yaml")
    if not walk_yaml.is_file():
        raise SystemExit(f"no walking deploy.yaml at {walk_yaml}: run from unitree_rl_mjlab or pass --deploy")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        raise SystemExit("--name: letters, digits, _ . - only")

    import install_getup as ig
    import yaml

    out = REPO / "policy" / args.name
    if out.exists():
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_onnx = Path(tmp) / "policy.onnx"
        changes = clean_onnx(src_onnx, tmp_onnx, run.name)
        notes = [n.replace(str(tmp_onnx), str(src_onnx)) for n in ig.make_policy_dir(tmp_onnx, walk_yaml, out)]
    dy = out / "params" / "deploy.yaml"
    made = yaml.safe_load(dy.read_text())
    changes += set_gains(out / "exported" / "policy.onnx", made["stiffness"], made["damping"])
    md5 = hashlib.md5((out / "exported" / "policy.onnx").read_bytes()).hexdigest()
    text = re.sub(r"^# from .*$", f"# from exported/policy.onnx (md5 {md5}), training run {run.name}",
                  dy.read_text(), count=1, flags=re.M)
    dy.write_text(text)
    opts = clean_options(run / "params" / "getup_options.json", run.name)
    (out / "params" / "getup_options.json").write_text(json.dumps(opts, indent=1) + "\n")
    shutil.copy2(ckpt, out / ckpt.name)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out / "exported" / "policy.onnx"), providers=["CPUExecutionProvider"])
    io = (sess.get_inputs()[0].name, sess.get_outputs()[0].name)
    meta = dict(sess.get_modelmeta().custom_metadata_map)
    joints = [s.strip() for s in meta.get("joint_names", "").split(",") if s.strip()]
    deploy = yaml.safe_load(text)
    obs_names = [k for k in deploy["observations"] if k not in ("scale_first", "use_gym_history")]
    history = next(iter(deploy["observations"].values()))["history_length"]
    results = f"\n{args.results}\n" if args.results else ""
    (out / "README.md").write_text(policy_readme(args.name, run.name, ckpt, deploy, joints,
                                                 obs_names, history, md5, results, io))

    words, hits = private_words(), []
    for f in sorted(out.rglob("*")):
        if f.is_file():
            data = f.read_bytes().lower()
            hits += [f"{f.relative_to(REPO)}: '{w}'" for w in words if w.encode() in data]
    print(f"policy/{args.name}/ from {run.name} ({ckpt.name}):")
    for line in changes + notes:
        print("  " + line)
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out)}  {f.stat().st_size / 1024:.0f} KB")
    if hits:
        print("PRIVATE WORDS FOUND (do not publish before fixing):\n  " + "\n  ".join(hits))
        return 1
    print("no private word found" if words else "no notes/private_words.txt: run scripts/check_public.sh")
    print(f"next: bash scripts/check_public.sh && git add policy/{args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
