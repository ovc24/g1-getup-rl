# G1-23DoF get-up policy (mjlab)

Reinforcement learning task that teaches the Unitree G1 (23 DoF) to stand up from the
ground, from any way it can lie after a fall (on its back, on its front, on a side, sitting,
crumpled), and to end standing still in the walking pose so that a walking policy can take
over. Built on [mjlab](https://github.com/mujocolab/mjlab) 1.2.0 and Unitree's
[unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) (same robot model,
gains and motors as Unitree's G1-23DoF walking task), trained with PPO (rsl_rl).

The reward design follows HoST, *Learning Humanoid Standing-up Control across Diverse
Postures* (Huang et al., RSS 2025, [paper](https://arxiv.org/abs/2502.08378),
[code](https://github.com/OpenRobotLab/HoST), MIT), which got the same robot up in the real
world. What is different here, and why, is in [How it works](#how-it-works).

Status:
- The first run (no pace) stood up from every posture in a kip-up-like burst (0.5 s from
  the back, knees at 83 % of their no-load speed). The **pace** and the torque/speed
  budgets (see [The pace](#the-pace)) fixed that.
- v4 (pace, budgets, pushes while standing) gets up in Unitree's own MuJoCo model of the
  robot (`sim2sim_getup.py`): every episode, in about 2.4 s, joints below half their
  no-load speed, no motor at its torque limit. It also runs inside Unitree's `g1_ctrl`
  next to the walking policy ([deploy/](deploy/README.md)): walking -> fall -> limp ->
  get up -> walking, in Unitree's simulator.
- Its style is not yet one for a real robot: legs swung high for momentum (from the front
  sometimes over the back, the head on the ground), wide splits, fast rolls; and it never
  started from a limp collapse (motors limp while it stands: the legs fold under it), the
  start of one attempt that stalled kneeling in `g1_ctrl`. From such collapses, in
  Unitree's model, it gets up in 19 of 20 episodes (2.2 s), but in 4 of 20 the head hits
  the ground with 700-900 N: Unitree's training model collides with a head sphere of 6 cm
  radius, the real head (and the mesh of Unitree's MuJoCo model) reaches up to 5.6 cm
  further, the face with the camera included. **v5** is a fine-tune for all of that
  ([3.1](#31-v5-style-fine-tune)).
- **v5** (1500 iterations from v4): 100 % of 600 evaluation episodes (v4 99.8 %), standing
  in 1.8 s (v4 2.3 s), no head hit (p90 0 N; v4 up to 353 N), a foot more than 0.2 m above
  the pelvis in 9 % of the episodes (v4 57 %). In Unitree's model: 100 % from the pool and
  from collapses, head contacts from collapses p90 53 N (v4 700 N). The sequence now: sit
  up with the hands behind, lift the hips (hands and feet on the ground), turn to face the
  ground, squat, stand; from the front or a side it first rolls over. Not yet tried on the
  real robot. The trained policy is in [`policy/getup_v5/`](policy/getup_v5/README.md)
  ([section 6](#6-trained-policy-to-run-elsewhere-policy)).
- v5 leans its arms on its own body while getting up: one part of the robot presses on
  another during a median 1.1 s of its ~2 s get-up (training's simulator), an upper arm on
  the torso up to ~700 N in Unitree's model. **v6** ([3.2](#32-v6-no-self-contacts),
  2000 iterations continuing v5 with a self-contact cost) keeps the arms off the body:
  99.7 % of 600 evaluation episodes (v5 99.7 % on the same pool), standing in 1.9 s,
  no head hit, self-contacts while getting up from a median of 1.1 s per episode to 0 (p90
  0.1 s; none at all in 45 % of the episodes). In Unitree's model: 100 % from the pool, from
  collapses and from collapses with the torque off; an upper arm on the torso ~0.03 s per
  get-up (v5 ~1.2 s). Left: short knocks (a hand on a hip, the pelvis on a heel, under
  0.1 s). Its style is one-sided (the right arm held out as a counterweight), equally
  good from either side. Not yet tried on the real robot. Trained policy:
  [`policy/getup_v6/`](policy/getup_v6/README.md).
- The self-collision benchmark (section 5: links not colliding with each other, a conflict
  is one link more than 1 cm into another) fails v6 in 15 of 15 starts, and almost only
  while standing: it stands with the arms turned in, a hand 7-9 cm inside the hip of
  Unitree's meshes for the whole stand (~7.6 s per 10 s episode); getting up, its conflicts
  last a median 0.06 s (v5: 3.6 s). Its self-contact cost only counted while getting up,
  and in the training model's capsules the hands only touch the hips. **v7**
  ([3.3](#33-v7-hands-off-the-hips), in training) measures the clearance of the arm links
  with stand-ins of Unitree's shapes, getting up and standing, and pays for it.

Everything is checked offline (`python3 checks/getup_check.py`, without a GPU).

## 1. Setup

Use the environment of `unitree_rl_mjlab` (the one that trains the walking policy). The
commands below assume the checkouts in `~/g1_ws`; any folder works.

```bash
cd ~/g1_ws && git clone https://github.com/ovc24/g1-getup-rl.git g1_getup
cd ~/g1_ws/unitree_rl_mjlab && source .venv/bin/activate      # every new terminal
```

## 2. Starting poses (once)

```bash
python ~/g1_ws/g1_getup/make_fallen_poses.py            # -> g1_getup/data/fallen_poses.npz
```

Drops the robot 6000 times in plain MuJoCo (CPU, a few minutes with `--workers 4`) and keeps
every resting pose. `train_getup.py` does this by itself if the file is missing.

How it is released (`--drops`, weights): `supine`, `prone`, `left_side`, `right_side`,
`random` (any orientation), `standing` (on its feet with a shove: it tips over), `collapse`
(standing in about the walking pose, barely moving, motors limp: knees and hips fold where
it stands and it ends up lying, mostly on its back or front, legs folded under or beside
it; a limp robot does not stay kneeling, nothing holds the hips), `collapse_off` (the same
with the torque off, no motor damping either: it folds completely, often with the arms
caught between the legs and the body). The default mix has neither; the v5 pool:

```bash
python ~/g1_ws/g1_getup/make_fallen_poses.py --out ~/g1_ws/g1_getup/data/fallen_poses_v5.npz \
    --drops supine=0.22,prone=0.22,left_side=0.09,right_side=0.09,random=0.08,standing=0.15,collapse=0.15
```

The v6 pool adds torque-off collapses:

```bash
python ~/g1_ws/g1_getup/make_fallen_poses.py --out ~/g1_ws/g1_getup/data/fallen_poses_v6.npz \
    --drops supine=0.2,prone=0.2,left_side=0.08,right_side=0.08,random=0.08,standing=0.14,collapse=0.11,collapse_off=0.11
```

It prints how the robot ended up (supine, prone, sides, upright = sitting or kneeling,
other) and how it was released. The v5 pool: supine 57 %, prone 34 %, each side 5 %, no
upright or other (with limp motors nothing stays sitting); 15 % of it comes from collapses.
Look at the poses before training (no policy, the motors just pull towards the walking
pose):

```bash
python ~/g1_ws/g1_getup/play_getup.py --zero            # open the viser link in a browser
```

## 3. Train

```bash
python ~/g1_ws/g1_getup/train_getup.py --num-envs 64 --iterations 5     # smoke test
python ~/g1_ws/g1_getup/train_getup.py                                   # 2048 envs, 6000 it.
python ~/g1_ws/g1_getup/train_getup.py --num-envs 4096                   # more GPU memory
```

Logs and checkpoints (`model_<N>.pt` every 100 iterations, `policy.onnx`) go to
`logs/rsl_rl/g1_23dof_getup/<date>/`. In tensorboard (`tensorboard --logdir logs/rsl_rl`):

| curve | what it means |
|---|---|
| `Metrics/getup/success` | share of episodes that end standing (>= 1 s) |
| `Metrics/getup/assist_N` | assist force on the torso; goes 200 N -> 0 as success rises |
| `Metrics/getup/reg_scale` | weight of the regularization rewards; 0.2 -> 1 |
| `Metrics/getup/time_to_stand_s` | when the first 1 s stand began (10 = never); with the 3 s pace it should settle around 2-3 s, not below 1 s |
| `Metrics/getup/head_ground_s`, `head_peak_N` | head (camera) on the ground: time, hardest hit |
| `Metrics/getup/airborne_s` | time with nothing on the ground (jumps); should be ~0 |
| `Metrics/getup/self_contact_s`, `self_peak_N` | one part of the robot pressing on another while getting up (> 1 N): time, strongest; v6 should bring it near 0 |
| `Metrics/getup/self_scale` | share of the self-contact cost in use: 0.05 -> 1 while the success rate stays above 90 %, down again if it falls below 80 % |
| `Metrics/getup/conflict_up_s`, `conflict_stand_s`, `conflict_cm` | an arm link more than 1 cm into the body in stand-ins of Unitree's shapes (the benchmark's measure, section 3.3): s per episode getting up and standing, deepest cm. Every run logs it; v7 should bring `conflict_stand_s` to ~0 |
| `Metrics/getup/self_stand_s` | one part of the robot pressing on another while standing (v8 cost), s per episode |
| `Metrics/getup/clear_scale` | share of the link-clearance cost in use (v7), as `self_scale` |
| `Episode_Reward/*` | every reward term |

How fast it may get up: `--pace 3` (default: seconds over which the task targets go from
lying to standing), `--torque-budget 0.8` (PD torque asked above 80 % of the peak torque
costs), `--speed-budget 0.6` (joint speed above 60 % of the no-load speed costs).
`--pace 0 --torque-budget 0 --speed-budget 0.9` is nearly the first version, which got up
in 0.5 s.

Pushes: `--push 0.4` (default) gives robots that are up on their feet, past the pace, a push
every 2-4 s (pelvis velocity change up to ±0.4 m/s sideways and forward, ±0.5 rad/s yaw), so
the policy learns to keep its balance standing, or to get up again if it falls. `--push 0`
turns them off. `eval_getup.py --push 0.4` measures it ("fell again").

Useful options: `--postures supine,prone` (only some starting postures), `--no-symmetry`,
`--no-realistic` (ideal motors), `--action-scale 0.25` (Unitree's walking value),
`--assist 0`, `--max-std 1.0`. To continue a stopped run as a new one:
`--init-checkpoint <run>/model_<N>.pt --assist-start <last assist_N> --reg-scale-start
<last reg_scale>` (the curriculum state is not in the checkpoint; from a run made before
the pace, only the policy is loaded and the critic starts new).

### 3.1 v5: style fine-tune

Style options (their defaults are the v5 values; runs before v5 had 0, 1 and off, and
`eval_getup.py`/`play_getup.py` rebuild every run with its own):

- `--feet-high 4`: feet above the pelvis cost 4 per m (summed over both feet, from 5 cm
  above the pelvis) per second: no legs swung up for momentum, no legs over the back.
- `--head-scale 2`: the head (camera) on the ground and head hits cost twice as much.
- joint friction (0-0.4 N m), damping (0-0.1 N m s/rad) and armature (x0.8-1.25)
  randomized per robot (`--no-joint-dr`: off). Real joints are not ideal; Unitree's
  MuJoCo model has 0.2, 0.05 and 0.01 in every joint.

Not an option, from v5 on for every run (evaluations of older runs included): the head
collision geom has the size of the real head (`getup/robot.py`, a capsule that covers the
`head_link` mesh within 0.5 cm), so the head penalties see the contacts the real head
would have. The fallen poses rest on the head too: make the pool again after this change
(the training warns about a pool made with the old head).

v5 starts from the last v4 checkpoint, without the assist force and with the full
regularization (where v4 ended), on the pool with the collapse releases (default posture
mix; classes the pool does not have are skipped):

```bash
python ~/g1_ws/g1_getup/train_getup.py \
    --init-checkpoint logs/rsl_rl/g1_23dof_getup/<v4 run>/model_5999.pt \
    --assist 0 --assist-start 0 --reg-scale-start 1 --iterations 1500 \
    --pool ~/g1_ws/g1_getup/data/fallen_poses_v5.npz --max-std 0.5
```

`--max-std 0.5` keeps the action noise from growing during the fine-tune (it prints where
the checkpoint's noise was). Compare v4 and v5 under the same conditions, the v5 pool and
joint randomization on for both (v4 was trained without it), also only from the collapses
(`--released collapse`), and in Unitree's model (`sim2sim_getup.py`, below):

```bash
P5=~/g1_ws/g1_getup/data/fallen_poses_v5.npz
python ~/g1_ws/g1_getup/eval_getup.py --checkpoint <v4 run>/model_5999.pt --pool $P5 --joint-dr on
python ~/g1_ws/g1_getup/eval_getup.py --checkpoint <v5 run>/model_1499.pt --pool $P5 --joint-dr on
python ~/g1_ws/g1_getup/eval_getup.py --checkpoint <v5 run>/model_1499.pt --pool $P5 --joint-dr on \
    --released collapse
```

Success and time to stand should stay; the "feet above the pelvis" and head lines should
drop.

### 3.2 v6: no self-contacts

v5 leans its arms on its body while getting up: in Unitree's model (`sim2sim_getup.py`) an
upper arm presses on the torso in almost every episode, up to ~700 N, and in a simulator
where the links do not collide with each other its arms go through the chest and its
hands into the hips. On the real robot that is an arm caught between the body and the
ground, or squeezed by a leg (hip and knee motors 88-139 N m, arm motors 25 N m). The
training had no cost for it; Unitree's walking task has one (`self_collisions`, physics
substeps with a self-contact above 10 N).

- `--self-collision 1` (default from v6; 0 for runs before): while the robot is not
  standing, every physics substep in which one part of the robot presses on another
  (strongest self-contact > 1 N; a link and its parent never collide) costs 1 per second:
  a contact all the time costs 4 per second, like the head on the ground. Standing is left
  out: in the walking pose the hands hang 2 mm from the hips (Unitree's capsules).
- The weight of that cost follows the success rate (`Metrics/getup/self_scale`): it
  starts at 5 %, rises to the full weight in 40 x num_envs episodes while more than 90 %
  of the episodes end standing, and falls twice as fast below 80 %. The first try used the
  full weight from the start and forgot how to stand: after 1300 iterations it sat up,
  head high, touching nothing, and never stood (success 0).
- The pool also has collapses with the torque off (`collapse_off`, section 2): the start
  where the arms are caught between the legs.

v6 continues the v5 run with `--resume`, not `--init-checkpoint` as v5 did from v4:
`--init-checkpoint` starts a new, empty optimizer (Adam), whose first updates move every
weight by about the learning rate whatever the gradient. A converged policy wobbles
(more self-contacts, joints at their limits) and has to find its way back: on the v5
pool it did in ~100 iterations, on the v6 pool three tries never stood again. `--resume`
also loads the optimizer state of v5. The cost reaches its full weight ~950 iterations
after the success rate passes 90 %, hence 2000 more (model_1500 ... model_3499):

```bash
python ~/g1_ws/g1_getup/train_getup.py \
    --resume logs/rsl_rl/g1_23dof_getup/<v5 run>/model_1499.pt \
    --assist 0 --assist-start 0 --reg-scale-start 1 --iterations 2000 \
    --pool ~/g1_ws/g1_getup/data/fallen_poses_v6.npz --max-std 0.5 --self-collision 1
```

The success of the first ~20 iterations of any run is low by construction: the first
episode of every robot starts at a random time and is cut short.

`Metrics/getup/self_contact_s` should fall towards 0 while `success` stays (v6: 0.03 s
and 100 % at the end, `self_scale` 1). Then compare v5 and v6 on the v6 pool
(`eval_getup.py` prints the self-contacts too) and in Unitree's model, also after
torque-off collapses:

```bash
P6=~/g1_ws/g1_getup/data/fallen_poses_v6.npz
python ~/g1_ws/g1_getup/eval_getup.py --checkpoint <v5 run>/model_1499.pt --pool $P6 --joint-dr on
python ~/g1_ws/g1_getup/eval_getup.py --checkpoint <v6 run>/model_1499.pt --pool $P6 --joint-dr on
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy <v6 run> --start collapse --limp-kd 0 --episodes 20
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy <v6 run> --episodes 60
```

Results (v5 and v6, same pool and seeds):

| | v5 | v6 |
|---|---|---|
| success, 600 episodes (training's simulator) | 99.7 % | 99.7 % |
| standing after (median) | 1.8 s | 1.9 s |
| self-contact while getting up, per episode: median, p90 | 1.14 s, 1.58 s | 0.00 s, 0.10 s |
| episodes with one / strongest (median) | 100 % / 289 N | 55 % / 40 N |
| Unitree's model, torque-off collapses: success, head p90 | 100 %, 160 N | 100 %, 1 N |
| Unitree's model, from the pool: upper arm on the torso per get-up | ~1.2 s | ~0.03 s |

### 3.3 v7: hands off the hips

The benchmark of section 5 measured v6 in Unitree's scene with the links not colliding
with each other: in all 15 starts a link went more than 1 cm into another, and in all 15
it was while standing, the right hand 7.3-8.7 cm inside the right hip (hip roll and
pitch links; the left hand 3-6 cm) for ~7.6 s of the 10 s episode. Getting up it was
nearly clean: conflicts in 12 of 15 starts, a median 0.06 s, mostly an upper arm 1-2 cm
into the torso. In the walking pose the hands are 2.6 cm from the hips (Unitree's convex
hulls; turning a shoulder 5 deg in already closes that), and v6 stands 5.7 deg RMS away
from it. Its costs did not see this: the self-contact cost counts only while getting up,
the standing pose reward (`stand_pose`, std 0.3 rad over all joints) barely notices a few
degrees in one shoulder, and in the training model the hands and hips are thin capsules
that only touch where Unitree's hulls overlap by centimetres.

v7 measures what the benchmark measures, in training (`getup/clearance.py`): the clearance
of each arm link (shoulder yaw, elbow, hand) to the torso, the pelvis, the legs and the
other arm (93 pairs), with stand-ins of the collision shapes of Unitree's deploy scene:
per shape, its convex hull wrapped in 26 planes and 32 points spread over it (made by
`scripts/make_clearance_shapes.py` from the scene's meshes, left and right mirror images).
Against the exact hull distances in 100 random poses it reads 0.5 cm closer near contact
(p10 1.4 cm closer, so it errs on the safe side) and found 151 of 152 conflicts deeper than
1 cm. In the walking pose every pair is at least 2.1 cm apart (2.45 cm exact).

- `--clearance 0.2` (default from v7; 0 for runs before): every pair closer than 1 cm
  costs 0.2 per cm and second, getting up and standing (v6 standing: ~30 cm summed, ~6 per
  second at full weight, against 8 per second for standing). The weight follows the
  success rate like the self-contact cost, from `--clearance-scale-start` (default 5 %).
- Every run logs `Metrics/getup/conflict_up_s`, `conflict_stand_s`, `conflict_cm`, and
  `eval_getup.py` reports the same over 600 episodes (with the pairs most often in a
  conflict), so the benchmark's number can be followed during training.

v7 continues the v6 run (`--resume`), the self-contact cost at full weight from the start
as v6 ended, the clearance cost from 25 % (moving the hands out costs the get-up nothing):

```bash
python ~/g1_ws/g1_getup/train_getup.py --resume <v6 run>/model_3498.pt \
    --assist 0 --assist-start 0 --reg-scale-start 1 --iterations 1500 \
    --pool ~/g1_ws/g1_getup/data/fallen_poses_v6.npz --max-std 0.5 \
    --self-collision 1 --self-scale-start 1 --clearance 0.2 --clearance-scale-start 0.25
```

`conflict_stand_s` should fall from ~7 s to ~0 and `success` stay near 1; `clear_scale`
reaches 1 after ~630 iterations above 90 % success. Then the benchmark (section 5) on the
last checkpoint, `model_4997.pt`.

Result (v7, `model_4997.pt`): in the training's simulator no conflict while standing in
600 episodes (getting up: under 0.1 s, mostly an upper arm 2-4 cm on the torso), 100 %
standing in 1.9 s. In Unitree's scene with the links colliding: no conflict standing, but
the right hand leans on the right hip with ~25 N for the whole stand; with the links
passing through each other (the benchmark) that push takes it 8.7 cm in, 15 of 15. The
clearance measures where the hand is, not where it is pushed.

v8: the self-contact cost also while standing (`--self-standing 1`, default from v8),
continuing v7:

```bash
python ~/g1_ws/g1_getup/train_getup.py --resume <v7 run>/model_4997.pt \
    --assist 0 --assist-start 0 --reg-scale-start 1 --iterations 1000 \
    --pool ~/g1_ws/g1_getup/data/fallen_poses_v6.npz --max-std 0.5 \
    --self-collision 1 --self-scale-start 1 --clearance 0.2 --clearance-scale-start 1 \
    --self-standing 1
```

`Metrics/getup/self_stand_s` (self-contact time while standing) should go to ~0; then the
benchmark on `model_5996.pt`.

### 3.4 What a run saves

Next to the checkpoints, in `<run>/params/`: `env.yaml` and `agent.yaml` (the whole
configuration), `getup_options.json` (what `eval_getup.py`/`play_getup.py` need to rebuild
the environment), `run_info.json` (date, exact command, starting checkpoint, git commit of
this repo and of `unitree_rl_mjlab` and whether they had uncommitted changes, md5 of the
pool), a copy of the pool and `code/` (the training script and `getup/` as they ran).

## 4. Evaluate and watch

```bash
python ~/g1_ws/g1_getup/eval_getup.py --checkpoint logs/rsl_rl/g1_23dof_getup/<run>/model_<N>.pt
python ~/g1_ws/g1_getup/eval_getup.py --checkpoint ... --postures prone --clean
python ~/g1_ws/g1_getup/play_getup.py --checkpoint ... [--postures supine]
```

`eval_getup.py` runs one 10 s episode per env (600) with the mean action, no assist force,
sensor noise and randomization on (`--clean`: off) and prints, per starting posture:
success, stood up at some point, time to stand (median, p90), fell again after standing,
head on the ground (s) and the hardest head hit (N); in how many episodes a foot went more
than 0.2 m above the pelvis (legs swung up). Then how violent the movement is:
peak joint speeds against the motors' no-load speeds, peak pelvis vertical and angular
speeds, time with nothing on the ground, and how often a motor is at its peak torque
while getting up and while standing, with the joints that are there most. Before a real
robot: time to stand of a few seconds, joint peaks well below 90 % of no-load, no flight,
motors at their peak only briefly and never while standing.

## 5. Sim2sim: Unitree's MuJoCo model (CPU)

```bash
RUN=logs/rsl_rl/g1_23dof_getup/<run>            # the run folder: its policy.onnx
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy $RUN                   # 60 episodes
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy $RUN --view --episodes 5
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy $RUN --postures prone
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy $RUN --episodes 6 --video videos/
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy $RUN --start collapse --limp-kd 3 --episodes 20
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy $RUN --start collapse --limp-kd 0 --episodes 20
```

`--start collapse`: each episode the robot stands in the walking pose and the motors go
limp, as when the damping state is chosen while it stands in `g1_ctrl` (`--limp-kd 3`: the
damping of g1_ctrl's Passive and Fallen states; `--limp-kd 0`: the torque off, the robot
folds completely, often with the arms between the legs); once it lies still, the policy
runs.
`--released collapse` (or `standing`, ...): only pool poses released that way.

Runs `policy.onnx` (what the robot runs) in Unitree's own model of the robot
(`scene_g1_23dof.xml`, the one of unitree_mujoco and `g1_ctrl`), on the CPU, one episode
after the other, with the same report table as `eval_getup.py`. Each episode a pool pose
(random heading) is put just above the floor and settles with limp motors (damping only)
until still, then the policy runs for 10 s. Same as in training: the 375 inputs (built as
mjlab does: terms, noise, history), PD gains, torque limits and curve, action scale,
standing test. Different, on purpose: the collision shapes (Unitree's scene collides with
the convex hulls of the link meshes and 4 small spheres per foot; training: capsules and
spheres), armature 0.01, damping 0.05 and dry friction 0.2 N m in every joint, contact
friction 1.0, CPU MuJoCo with 2 ms steps, no randomization. The right ankle motors are
limited to +-25 N m in that scene (training 50): raised unless `--keep-scene-limits`.
The self-collision benchmark (what the policy would do to the robot's own body: the
links do not collide with each other, a conflict is one link more than 1 cm into another,
from supine, prone and a face-down fall; it fails if any episode has one or does not end
standing):

```bash
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy $RUN --benchmark 5
```

The conflicts are split into getting up and standing (with the time with a conflict in
each), and starts already in a conflict (links that went into each other during the limp
fall, before the policy) are counted apart: the report and the BENCHMARK line also give the
counts without those conflicts (each until it first clears; the benchmark itself still
fails on them). Also: the closest arm-body pairs in the walking
pose, the conflicts per part of the body (each hand against the hips and legs, the arms
against the torso: starts, time, deepest, per phase), the joints furthest from the walking
pose at the end (a shoulder roll marked "arm in" or "arm out"), and the training's measure
(section 3.3) on the same steps next to the exact one. To watch it, two policies side by side from
the same starts (the same `--seed` gives the same poses, headings, falls and camera):

```bash
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy policy/getup_v5 --benchmark 5 --video videos/bench_v5
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy policy/getup_v6 --benchmark 5 --video videos/bench_v6
python ~/g1_ws/g1_getup/scripts/side_by_side.py videos/bench_v5 videos/bench_v6 --out videos/v5_vs_v6
```

Each video has a header with the policy (`--label`, default the policy folder's name) and
the start, "getting up" / "STOOD UP at X s", and the time with a conflict so far, getting
up and standing; the links in a conflict at that moment are drawn red and named at the
bottom with their depth. `videos/v5_vs_v6/` gets one video per start and `all.mp4` with
all of them one after the other. `--view` shows the red links live (and the text, if the
MuJoCo viewer of that version can show text). Pictures are 800x600 (`--video-size`).

It also reports the peak pelvis speeds, the time with nothing on the ground and the
self-contacts: how long one part of the robot presses on another (> 5 N, and > 50 N, about
what an arm motor can push back with), while getting up and while standing, the hardest
one, which pairs of links, and how often the start pose was already pressing. Pairs one
joint apart (pelvis - hip roll, torso - shoulder roll) overlap around their joint in this
model's convex shapes: they are listed and marked but left out of the totals; pairs that
already touch in the walking pose are marked too. Needs `onnxruntime` (`uv pip install
onnxruntime`).

### 5.1 Scenario videos: walking, pushes, falls, getting up

`sim2sim_scenario.py` runs a scripted scenario with the files and the state machine of
`g1_ctrl` (installed with `deploy/install_getup.py`) in the same Unitree model, and
records it without a window: no lag (it does not run in real time), every frame drawn,
the state and the events written on the video.

```bash
python ~/g1_ws/g1_getup/sim2sim_scenario.py --out videos/            # what g1_ctrl has installed
python ~/g1_ws/g1_getup/sim2sim_scenario.py --walk logs/rsl_rl/<walking run> --out videos/
python ~/g1_ws/g1_getup/sim2sim_scenario.py --print-script > scenario.txt   # edit, then
python ~/g1_ws/g1_getup/sim2sim_scenario.py --script scenario.txt --out videos/
```

It reads `deploy/robots/g1_23dof/config/`: `config.yaml` (states and settings), the
walking and the get-up policy (`exported/policy.onnx` + `params/deploy.yaml`, the version
folder `g1_ctrl` would pick). `--walk`/`--getup`: another `policy.onnx` with the same
`deploy.yaml`, or a policy folder with its own (`--getup policy/getup_v5`). The policies run as in `g1_ctrl` (a Python port of its observation terms,
history, clip and scale, actions, gait phase), the states as installed (fall at a tilt
of 1 rad, Fallen, GetUp waiting until still, hand-over when standing). The default
scenario (82 s): standing, pushes from the four sides, walking with a push, turning, a
push too strong (fall, then the automatic get-up), motors off (collapse, get-up buttons),
hung from the elastic band and dropped (get-up buttons). Events: `walk`, `cmd VX VY WZ`,
`push DIR N [S]`, `limp`, `getup`, `band H`, `release`, `end` (details: `--help`). It
writes `scenario_<date>.mp4` (960x540, 25 fps) and `.txt` (events, transitions, every
get-up attempt with its time and hardest head contact).

## 6. Trained policy, to run elsewhere (`policy/`)

`policy/<name>/` holds a trained policy, ready for `g1_ctrl`, Unitree's simulator, another
simulator or the robot, without training it again (`policy/getup_v6/`: v6, the one to
use; `policy/getup_v5/`: v5, which leans its arms on its body):

| File | Use |
|---|---|
| `exported/policy.onnx` | the network: one input `obs` (375 numbers), one output `actions` (23); its metadata: joint names, default pose, action scale, the PD gains it was trained with |
| `params/deploy.yaml` | its `g1_ctrl` config |
| `params/getup_options.json` | the environment it was trained in (`eval_getup.py`, `play_getup.py`, `sim2sim_getup.py`) |
| `model_<N>.pt` | the PyTorch checkpoint of the same save (evaluation, fine-tuning) |
| `README.md` | how to run it in any simulator or controller: the inputs (order, history), the outputs, the joints with default pose, action scale, kp and kd |

With unitree_rl_mjlab (section 1); `policy/<name>` is found in this repo from any folder:

```bash
cd ~/g1_ws/unitree_rl_mjlab && source .venv/bin/activate
# into g1_ctrl, then build it (deploy/README.md)
python ~/g1_ws/g1_getup/deploy/install_getup.py --deploy deploy --policy-dir policy/getup_v6 --manual
# Unitree's MuJoCo model, without g1_ctrl (section 5)
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy policy/getup_v6 --start collapse --episodes 20
python ~/g1_ws/g1_getup/sim2sim_getup.py --policy policy/getup_v6     # from the pool (make it first)
# the scenario video with g1_ctrl's states (section 5.1)
python ~/g1_ws/g1_getup/sim2sim_scenario.py --getup policy/getup_v6 --out videos/
# the training environment (GPU; the pool of getup_options.json, data/<file>, made first)
python ~/g1_ws/g1_getup/eval_getup.py --checkpoint ~/g1_ws/g1_getup/policy/getup_v6/model_3498.pt
```

The pool is not in the repo (it is made in a few minutes): for v6 the v6 one, with the
command of section 2, written to `data/fallen_poses_v6.npz`, the file its
`getup_options.json` names (v5: `data/fallen_poses_v5.npz`).

A run becomes such a folder with `scripts/import_policy.py` (in the training venv): it
copies the last checkpoint and its `policy.onnx`, writes the `deploy.yaml` as
`deploy/install_getup.py` does, removes the paths of the training computer from the ONNX
metadata and the options, and writes the README:

```bash
python ~/g1_ws/g1_getup/scripts/import_policy.py --run logs/rsl_rl/g1_23dof_getup/<run> --name getup_v6
bash ~/g1_ws/g1_getup/scripts/check_public.sh
```

## How it works

**Episodes.** 10 s at 50 Hz. Each starts from a random pose of the pool (random heading and
position), at rest. The pool comes from dropping the robot in MuJoCo with the training scene:
released lying on the back / front / sides (+-25 deg), in a random orientation, standing
on its feet with a shove (it tips over and collapses), or standing still with the motors
limp (it crumples where it stands); joints at a random mix of the walking
pose and a random pose (`poses.JOINT_SAMPLE_RANGES`), motors limp (damping only), holding the
pose, or holding then going limp; kept once everything is still. Starting at rest from a pool
instead of dropping the robot in every episode (HoST) means no wasted falling time and no
spawn-height computation on the GPU. Training mix: supine 30 %, prone 30 %, each side 12 %,
sitting/kneeling 8 %, other 8 % (`--postures`).

**Policy.** Inputs, all available on the robot: pelvis IMU angular velocity, gravity in the
pelvis frame, joint angles minus the walking pose, joint speeds, last action (75 numbers),
for the last 5 steps (375). Outputs: joint position targets around the walking pose, PD at
Unitree's gains, like the walking policy, but with twice the action scale (0.5 x peak torque
/ stiffness rad per unit): getting up needs the peak torques often. The critic also sees
pelvis velocity, pelvis/head/foot heights, which bodies touch the ground, the head contact
force, the assist force and where the pace schedule is.

**Rewards** (per second; mjlab multiplies by dt):

| group | term | weight | when |
|---|---|---|---|
| task | head height x pelvis uprightness (HoST) | 3.0 | always; targets paced |
| task | pelvis height | 1.5 | always; target paced |
| standing | still (no pelvis speed, no rotation) | 1.0 | standing, from 3 s on |
| standing | walking pose (exp of mean squared joint error) | 1.5 | standing, from 3 s on |
| standing | both feet on the ground | 1.0 | standing, from 3 s on |
| style | head (camera) on the ground | -4.0 (v4: -2.0) | always |
| style | head impact above 50 N | -8.0 per 100 N (v4: -4.0) | always |
| style | feet above the pelvis: Σ (foot height - pelvis height - 5 cm), if positive | -4.0 per m (v4: 0) | always |
| style | waist yaw > 0.9 rad, hip yaw > 0.8, hip roll > 0.6 from the walking pose | -2.0 per rad | always |
| style | nothing on the ground (a jump) | -5.0 | after the first 0.1 s |
| regularization | feet sliding: Σ (horizontal foot speed)² of the feet on the ground | -1.0 | curriculum |
| regularization | action rate, joint acceleration, speed, torque, limits | Unitree-like | curriculum |
| regularization | speed budget: (joint speed - 0.6 x no-load speed)^2, if positive | -2.0 | curriculum |
| regularization | torque budget: (PD torque asked / peak torque - 0.8)^2, if positive | -5.0 | curriculum |
| — | simulation blew up | -100 | episode end |

"Standing" = pelvis above 0.62 m, head above 1.0 m, pelvis tilt under 32 deg, nothing but the
feet on the ground. Once the pace has reached standing, lying earns ~0.1/s, sitting up ~2,
kneeling or squatting ~3.7, standing 8: the standing terms make standing worth twice a
squat, so that the policy does not stop halfway (`checks/getup_check.py` prints the table).

### The pace

Without it, every second spent standing pays 8, so standing up 0.1 s earlier is worth more
than all the regularization of the whole movement costs, and the policy learns the fastest
way up there is. The first run did exactly that: up in half a second, knees near their
no-load speed. Penalizing speed harder only moves the optimum to "as fast as the penalty
allows".

With `--pace 3`, the targets of the task rewards rise linearly with the time since the
episode began: head from 0.15 to 1.10 m, uprightness from 0 to 0.95, pelvis from 0.10 to
0.70 m, over 3 s; the standing rewards count from 3 s on. Being ahead of that schedule
pays exactly as much as being on it (at 1.5 s, kneeling upright and standing both earn
4.5/s), falling behind costs (still lying at 1.5 s: 1.5/s). Standing up earlier earns
nothing, so the regularization decides how fast to move: calmly, with the motors within
their budgets. The critic sees the phase of the schedule; the policy does not, it stays a
pure feedback controller and needs no clock on the robot.

**Curriculum.** From the success rate of the episodes that end (moving average over
~num_envs episodes): above 50 %, the assist force (HoST's pulling force: 200 N up on the
torso, ~60 % of the robot's weight, while the pelvis is within 37 deg of upright) decreases
to 0 within 10 x num_envs episodes and never comes back; above 30 %, the regularization
weights grow from 20 % to 100 % within 20 x num_envs episodes. Early on the robot dares to
move; at the end it gets up without help and smoothly.

**Robot model.** Unitree's G1-23DoF model and actuators, with three changes: every
collision geom has friction (`condim=3`; Unitree's walking config makes all but the feet
frictionless, fine for walking, not when hands and knees push on the ground), a DC motor
torque-speed curve (the torque falls to zero at the motor's no-load speed), and (from v5)
a head collision geom of the real head's size (Unitree's: a 6 cm sphere, up to 5.6 cm
smaller than the head; here a capsule of 7 cm radius around an axis 38-43 cm above the
torso link). Randomized once per env:
friction 0.35-1.1 (all geoms), torso center of mass +-3 cm, torso mass -1..+2 kg, encoder
bias, PD gains +-15 %, motor strength +-10 %; from v5 also joint dry friction 0-0.4 N m,
joint damping 0-0.1 N m s/rad, rotor armature x0.8-1.25. Sensor noise as in the walking
task.
Left-right mirror augmentation (`getup/symmetry.py`): every sample is also learned mirrored,
so falls to the left and to the right are the same problem.

**Differences from HoST:** one policy for all starting postures (HoST trains one per
posture/terrain), a single critic instead of one per reward group (the regularization
curriculum plays that role), absolute joint targets around the walking pose (deployable with
Unitree's `g1_ctrl` as it is) instead of targets relative to the current angles, Unitree's
low PD gains, a pool of resting poses instead of a drop at every episode start, head contact
penalties. HoST keeps the motion slow by bounding its relative targets (which bounds the
torque); with absolute targets that bound would not limit the torque, so here the pace and
the torque and speed budgets do that job.

## Files

| file | what |
|---|---|
| `getup/robot.py` | robot config: collisions with friction, DC motor curve, action scale |
| `getup/poses.py` | fallen-pose pool: drops (CPU MuJoCo), posture classes, file format |
| `getup/pool_maker.py` | builds the training scene on the CPU and makes the pool |
| `getup/getup_env_cfg.py` | the environment: scene, sensors, observations, actions, rewards, events |
| `getup/getup_mdp.py` | reward, observation and termination functions, reset event |
| `getup/getup_term.py` | episode bookkeeping, assist force, curriculum |
| `getup/symmetry.py` | left-right mirror (with observation history) |
| `getup/run_options.py` | a run's options (`getup_options.json`), for eval and play |
| `getup/provenance.py` | `run_info.json` and the copy of the code in every run |
| `getup/train_utils.py` | action-noise cap (`--max-std`) |
| `train_getup.py`, `eval_getup.py`, `play_getup.py`, `make_fallen_poses.py` | scripts |
| `sim2sim_getup.py`, `getup/sim2sim_core.py` | sim2sim in Unitree's MuJoCo model |
| `getup/sim2sim_video.py`, `scripts/side_by_side.py` | the videos' header and red links; two runs side by side |
| `getup/clearance.py`, `getup/clearance_shapes.py` | link clearance with stand-ins of Unitree's collision shapes (v7 cost, metrics) |
| `scripts/make_clearance_shapes.py` | makes `clearance_shapes.py` from Unitree's scene and checks it against the exact hulls (needs scipy) |
| `sim2sim_scenario.py`, `getup/scenario_core.py` | scripted scenario videos with `g1_ctrl`'s files and states |
| `checks/getup_check.py`, `checks/sim2sim_core_check.py`, `checks/scenario_check.py`, `checks/sim2sim_video_check.py`, `checks/clearance_check.py` | offline checks (numpy only; the video one also Pillow and imageio-ffmpeg; the clearance one, with scipy and `UNITREE_RL_MJLAB`, also in the robot's poses) |
| `deploy/` | the get-up in Unitree's `g1_ctrl`: states, installer, keyboard pushes for Unitree's simulator ([README](deploy/README.md)) |
| `policy/` | trained policies, ready to run elsewhere (section 6) |
| `scripts/import_policy.py`, `scripts/check_public.sh` | a run into `policy/`; nothing private before a push |

## Troubleshooting

- **Out of GPU memory**: `--num-envs 1024`. Lying robots have many contacts; the scene keeps
  room for 128 contacts and 600 constraint rows per env (`sim.nconmax`, `sim.njmax` in
  `getup_env_cfg.py`). If mjlab warns that contacts or constraints overflowed, raise them.
- **`success` stays near 0 after ~1000 iterations**: look at `head_up` and `pelvis_height` in
  `Episode_Reward`: if they do not grow, the robot does not even sit up; try
  `--postures supine` first, or a larger `--entropy-coef`.
- **The assist force stops decreasing**: the policy needs the help; let it train longer.
  Check that the final policy works without it: `eval_getup.py` runs with 0 N.
- **Gets up in under a second, joints near no-load speed** (kip-up): a run without the pace
  (`--pace 0`, or made before the option existed). Train with the pace.
- **Still too fast with the pace** (time to stand well below the pace): a longer pace
  (`--pace 4`), a lower `--speed-budget` (0.5) or `--torque-budget` (0.7).
- **Never stands within the episode with the pace** (`success` stays low while sitting
  and kneeling work): a shorter pace or higher budgets; look at which postures fail with
  `eval_getup.py`.
- **A motor at its peak while standing** (`eval_getup.py` names the joint): usually an arm
  pressed against the body; the torque budget penalizes it. If it stays, raise the
  `torque_budget` weight in `getup_env_cfg.py`.

## Next steps

- v6 (section 3.2), then the real robot: lying on a mat under the gantry, get-ups started
  by hand (`deploy/README.md`).
- A calmer, more human sequence (roll to the front, hands and knees, one knee, stand),
  e.g. from a reference motion.
- Protective falls: a policy for the fall itself (arms out, head off the ground), as a
  state between walking and getting up.

## References

- T. Huang et al., *Learning Humanoid Standing-up Control across Diverse Postures*, RSS 2025.
  [arXiv:2502.08378](https://arxiv.org/abs/2502.08378), [code](https://github.com/OpenRobotLab/HoST) (MIT).
- X. He et al., *Learning Getting-Up Policies for Real-World Humanoid Robots* (HumanUP), RSS
  2025. [arXiv:2502.12152](https://arxiv.org/abs/2502.12152).
- mjlab: https://github.com/mujocolab/mjlab; unitree_rl_mjlab:
  https://github.com/unitreerobotics/unitree_rl_mjlab.

## License

MIT ([LICENSE](LICENSE)), the trained policies in `policy/` included. mjlab and
unitree_rl_mjlab (Apache-2.0) are used, not included: the robot model, the actuators and
`g1_ctrl` come from a checkout of unitree_rl_mjlab.
