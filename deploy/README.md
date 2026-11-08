# Get-up in g1_ctrl (walking + falling + getting up)

Two extra states for `g1_ctrl`, Unitree's controller in `unitree_rl_mjlab/deploy`, so that
the robot walks with the walking policy, goes limp when it falls, gets up with the get-up
policy and walks on:

```
Passive --L2+up--> FixStand --R2+A--> Velocity (walking policy)
                                         | tilt > 1 rad (a fall)
                                         v
              +--------------------- Fallen: motors limp (damping only)
              |                          | still for 1 s (auto_getup: true) or R2+X
              | not up after 10 s        v
              +--------------------- GetUp: limp until still for 0.5 s (at most 3 s),
                                         | then the get-up policy
                                         | upright, legs in the walking pose, still,
                                         | for 0.5 s (and at least 2 s after the start)
                                         v
                                     Velocity
Passive --R2+X--> GetUp   (the robot already lying on the ground)
L2+B: Passive from any of them (motors limp).
```

## Install

On the PC (needs onnxruntime, which the training venv has):

```bash
cd <unitree_rl_mjlab> && source .venv/bin/activate
python <this repo>/deploy/install_getup.py --deploy deploy \
    --getup-run logs/rsl_rl/g1_23dof_getup/<run> --auto
cd deploy/robots/g1_23dof/build && make -j4
```

Without training, with the policy of this repo: `--policy-dir policy/getup_v5` instead of
`--getup-run ...` (the folder as `policy/<name>`, found in this repo from any folder).

`--auto`: after a fall it gets up by itself once it lies still; `--manual` (default for a
new install): only with R2+X. The script (details in its docstring):

- copies `g1_ctrl/State_GetUp.h` and `State_Fallen.h` into `robots/g1_23dof/include/` and
  includes them in `main.cpp`;
- `State_RLBase.cpp`: on a fall the walking state goes to the state named `on_fall` in
  `config.yaml` (default Passive, as before);
- `terminations.h`: `bad_orientation` really checks the tilt (Unitree's returns false, so
  no fall is ever detected); `State_RLBase.h`: the gains go to the motors of the policy
  joints (Unitree's writes them to motors 0..22);
- `config.yaml`: the two states, their settings, Velocity `on_fall: Fallen`, Passive R2+X;
- `config/policy/getup/v0/`: the policy and its `deploy.yaml`, made from the policy's
  metadata (joint order, walking pose, action scale) and checked against the walking
  policy's: 5 inputs (pelvis angular velocity, gravity, joint angles minus the walking
  pose, joint speeds, last action) x the last 5 steps = 375 numbers, joint targets =
  walking pose + scale x action. PD gains: the walking ones, which the get-up was trained
  with. (mjlab writes stiffness 1 and damping 0 into the metadata of a policy trained
  with its DC-motor model, whose MuJoCo actuators are plain motors: with those the motors
  stay limp.)

Every file it changes is copied once to `<file>.orig`; running it again changes nothing
already done. Each policy's `deploy.yaml` starts with the path and md5 of the `policy.onnx`
it was made from.

**Updating** (a new get-up run, or a new version of the states): run the same command
again with the new `--getup-run`, then `make -j4`. It copies the two states again and
replaces the policy in `config/policy/getup/v0/`; `config.yaml` is left as it is (settings
added in a newer version take their defaults until you write them there). To go back to an
earlier policy, run it with that run.

On the robot (plain python3 is enough), with the policy folder made on the PC copied over:

```bash
python3 install_getup.py --deploy <robot>/deploy --policy-dir <copied>/v0 --manual
cd <robot>/deploy/robots/g1_23dof/build && make -j4
```

## Settings (config.yaml)

| state | key | default | meaning |
|---|---|---|---|
| Fallen | `auto_getup` | false | get up by itself once still |
| | `settle_s` | 1.0 | how long it must lie still (s) |
| | `still_gyro`, `still_joint_vel` | 0.5, 1.0 | "still": pelvis rad/s, every motor rad/s |
| GetUp | `settle_s` | 0.5 | motors limp until the robot has lain still this long (s), then the policy starts |
| | `max_settle_s` | 3.0 | start the policy anyway after this long (s) |
| | `still_gyro`, `still_joint_vel` | 0.5, 1.0 | "still", as in Fallen |
| | `min_time_s` | 2.0 | no hand-over before this (s from the policy start) |
| | `hold_s` | 0.5 | how long "standing" must hold (s) |
| | `max_tilt`, `max_gyro`, `max_leg_dev` | 0.35, 0.8, 0.5 | "standing": tilt (rad), pelvis rotation (rad/s), leg joints off the walking pose (rad) |
| | `max_time_s` | 10.0 | not standing by then (s from the policy start) -> `on_timeout` (Fallen) |

Why GetUp waits for the robot to lie still: the policy was trained starting from robots at
rest, and the buttons can be pressed (or Passive left) while the robot is still falling or
rolling. After Fallen with `auto_getup: true` the robot is already still, so this adds
only `settle_s`.

While it gets up, g1_ctrl prints once per second the tilt, the pelvis rotation, how far the
legs are from the walking pose and whether that counts as standing.

## Simulator first

`unitree_mujoco` (`simulate/`) with the G1-23DoF scene. Its scene limits the right ankle
motors to +-25 N m (left and training: 50); raise them before testing the get-up:

```bash
X=<unitree_rl_mjlab>/src/assets/robots/unitree_g1/xmls/scene_g1_23dof.xml
[ -f $X.orig ] || cp $X $X.orig
sed -i -E 's/(joint="right_ankle_(pitch|roll)_joint" ctrlrange=)"-25 25"/\1"-50 50"/' $X
```

Then: simulator, `g1_ctrl --network=lo`, gamepad. Lying test: in Passive release the band
(key 9) so the robot drops, then R2+X. Fall test: walk, release the band, push the robot
until it falls.

Pushing with the mouse (double-click the torso, Ctrl+right drag) is hard when the simulator
lags. `patch_unitree_mujoco.py` adds keyboard pushes to the simulator (a horizontal force
on the torso for 0.1 s, like the shoves of the walking fine-tune):

```bash
python3 <this repo>/deploy/patch_unitree_mujoco.py --unitree-mujoco <unitree_mujoco>
cd <unitree_mujoco>/simulate/build && make -j4
```

(`unitree_rl_mjlab` has its own copy of the simulator in `simulate/`: then pass the
`unitree_rl_mjlab` folder.)

| key (simulator window) | push |
|---|---|
| I / K | forward / backward (relative to where the robot faces) |
| J / L | left / right |
| = / - | force +-50 N (default 300 N) |
| P | automatic pushes on/off: every 4-6 s, random direction |

The simulator's terminal prints every push. To undo: copy `simulate/src/main.cc.orig`
back and build again.

A PlayStation-style gamepad on Linux, with unitree_mujoco's `joystick_type: "xbox"`, maps
A = cross, B = circle, X = triangle, Y = square (R2 = RT, L2 = LT): get up = R2 + triangle,
walk = R2 + cross, limp = L2 + circle.

For a video without lag, or the same test again and again, `sim2sim_scenario.py` (in this
repo's root) runs a scripted scenario with this installation's files and states in the
same model, off-screen (main README, section 5.1).

## On the robot

Start with `--manual`, the robot lying on a mat under the gantry with the rope slack, a
person at the remote with L2+B (limp) ready, and the first get-ups started with R2+X.
