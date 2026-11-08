# getup_v5: get-up policy for the Unitree G1-23DoF

From training run `2026-11-07_20-51-49`, checkpoint `model_1499.pt`; `exported/policy.onnx` md5 `1362b627024ed0fc016f916f9ccbab88`.

In simulation it stood up in every test episode: 600 in training's simulator (median 1.8 s) and 20 + 20 in Unitree's MuJoCo model (from the pool and from limp collapses). Not yet tried on the real robot.

## Files

| File | Use |
|---|---|
| `exported/policy.onnx` | the network (one input `obs`, one output `actions`): g1_ctrl, any ONNX runtime |
| `params/deploy.yaml` | its g1_ctrl config (`deploy/install_getup.py --policy-dir policy/getup_v5`) |
| `params/getup_options.json` | the environment it was trained in, for `eval_getup.py`, `play_getup.py`, `sim2sim_getup.py` |
| `model_1499.pt` | PyTorch checkpoint (actor, critic, optimizer): `eval_getup.py --checkpoint policy/getup_v5/model_1499.pt`, fine-tuning |

With this repo and unitree_rl_mjlab (main README, sections 1 and 6), from the unitree_rl_mjlab
folder with its venv:

```bash
python <this repo>/deploy/install_getup.py --deploy deploy --policy-dir policy/getup_v5 --manual   # into g1_ctrl
python <this repo>/sim2sim_getup.py --policy policy/getup_v5 --start collapse --episodes 20       # Unitree's MuJoCo model
python <this repo>/sim2sim_scenario.py --getup policy/getup_v5 --out videos/                      # scenario video
```

## Running it anywhere

Every 20 ms (50 Hz):

1. Build the input `obs` (shape 1 x 375): 5 signals, each over the last
   5 steps, **one signal after the other, oldest step first** (all 5 steps of signal 1,
   then all 5 of signal 2, ...). No scaling, no clipping. When the policy starts, every
   step of the history is the first reading, and the last action is 0.

   | # | signal | size | what |
   |---|---|---|---|
   | 1 | `base_ang_vel` | 3 | pelvis angular velocity, rad/s, pelvis frame (the IMU gyro) |
   | 2 | `projected_gravity` | 3 | gravity direction in the pelvis frame (unit vector; (0, 0, -1) upright) |
   | 3 | `joint_pos_rel` | 23 | joint angles minus the default pose below, rad |
   | 4 | `joint_vel_rel` | 23 | joint speeds, rad/s |
   | 5 | `last_action` | 23 | the network's previous output (0 at the start) |

2. `actions = policy(obs)` (shape 1 x 23, joint order below).
3. Joint targets: `q_target = default + scale * actions`, held for the 20 ms and tracked by
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
| 0 | `left_hip_pitch_joint` | -0.100 | 1.095 | 40.2 | 2.6 |
| 1 | `left_hip_roll_joint` | +0.000 | 0.701 | 99.1 | 6.3 |
| 2 | `left_hip_yaw_joint` | +0.000 | 1.095 | 40.2 | 2.6 |
| 3 | `left_knee_joint` | +0.300 | 0.701 | 99.1 | 6.3 |
| 4 | `left_ankle_pitch_joint` | -0.200 | 0.877 | 28.5 | 1.8 |
| 5 | `left_ankle_roll_joint` | +0.000 | 0.877 | 28.5 | 1.8 |
| 6 | `right_hip_pitch_joint` | -0.100 | 1.095 | 40.2 | 2.6 |
| 7 | `right_hip_roll_joint` | +0.000 | 0.701 | 99.1 | 6.3 |
| 8 | `right_hip_yaw_joint` | +0.000 | 1.095 | 40.2 | 2.6 |
| 9 | `right_knee_joint` | +0.300 | 0.701 | 99.1 | 6.3 |
| 10 | `right_ankle_pitch_joint` | -0.200 | 0.877 | 28.5 | 1.8 |
| 11 | `right_ankle_roll_joint` | +0.000 | 0.877 | 28.5 | 1.8 |
| 12 | `waist_yaw_joint` | +0.000 | 1.095 | 40.2 | 2.6 |
| 13 | `left_shoulder_pitch_joint` | +0.350 | 0.877 | 14.3 | 0.9 |
| 14 | `left_shoulder_roll_joint` | +0.180 | 0.877 | 14.3 | 0.9 |
| 15 | `left_shoulder_yaw_joint` | +0.000 | 0.877 | 14.3 | 0.9 |
| 16 | `left_elbow_joint` | +0.870 | 0.877 | 14.3 | 0.9 |
| 17 | `left_wrist_roll_joint` | +0.000 | 0.877 | 14.3 | 0.9 |
| 18 | `right_shoulder_pitch_joint` | +0.350 | 0.877 | 14.3 | 0.9 |
| 19 | `right_shoulder_roll_joint` | -0.180 | 0.877 | 14.3 | 0.9 |
| 20 | `right_shoulder_yaw_joint` | +0.000 | 0.877 | 14.3 | 0.9 |
| 21 | `right_elbow_joint` | +0.870 | 0.877 | 14.3 | 0.9 |
| 22 | `right_wrist_roll_joint` | +0.000 | 0.877 | 14.3 | 0.9 |
