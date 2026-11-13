#!/usr/bin/env python3
"""Checks of the video parts of sim2sim_getup.py:  python3 checks/sim2sim_video_check.py

The header of the frames (getup/sim2sim_video.py), the red links (on a stand-in model),
and, if imageio-ffmpeg is installed, the mp4 writer and scripts/side_by_side.py on
made-up videos. Needs numpy and Pillow; MuJoCo is not needed."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from getup.sim2sim_video import (HEADER, GREEN, RED, Highlighter, Mp4Writer,  # noqa: E402
                                 Overlay, draw_frame, pair_label, policy_label, read_mp4,
                                 short_link, side_by_side)

OK = True


def check(cond, msg):
    global OK
    print(("  ok   " if cond else "  FAIL ") + msg)
    OK &= bool(cond)


def near(img, rgb, tol=40):
    """How many pixels are close to this colour."""
    d = np.abs(img[..., :3].astype(int) - np.array(rgb)[None, None, :]).max(axis=-1)
    return int((d < tol).sum())


print("names")
check(short_link("right_wrist_roll_rubber_hand") == "right hand"
      and short_link("left_hip_roll_link") == "left hip roll" and short_link("torso_link") == "torso"
      and short_link("pelvis") == "pelvis", "short link names")
check(pair_label("left_hip_roll_link - left_wrist_roll_rubber_hand") == "left hip roll - left hand",
      "pair label")
check(policy_label(Path("policy/getup_v6/exported/policy.onnx")) == "getup_v6"
      and policy_label(Path("logs/rsl_rl/g1_23dof_getup/2026-11-08_22-20-06/policy.onnx"))
      == "2026-11-08_22-20-06" and policy_label(Path("a/b/model.onnx")) == "model",
      "policy label: the policy folder, else the run folder")

print("header")
ov = Overlay(label="getup_v6", measure=True, limit_cm=1.0)
ov.new_episode("getup_v6 · prone 2/5")
check(ov.badge()[0] == "motors limp" and ov.t is None, "limp before the policy")
ov.t, ov.phase = 0.8, "up"
ov.up_s, ov.stand_s = 0.12, 0.0
ov.active = {"right_hip_roll_link - right_wrist_roll_rubber_hand": 0.079,
             "right_shoulder_yaw_link - torso_link": 0.021}
check(ov.badge()[0] == "getting up" and "getting up 0.12 s" in ov.counter()
      and ov.now_lines()[0] == "right hip roll - right hand  7.9 cm", "getting up, deepest first")
img = np.full((480, 640, 3), 90, dtype=np.uint8)
f1 = draw_frame(img, ov)
check(f1.shape == (480 + HEADER, 640, 3) and f1.dtype == np.uint8, f"frame size {f1.shape}")
head, scene = f1[:HEADER], f1[HEADER:]
check(near(head, (255, 255, 255)) > 150 and near(head, (205, 205, 205)) > 100,
      "the header has the title (white) and the counter (grey)")
check(near(scene[-90:], RED) > 150, "the conflicts of this moment in red at the bottom")
check(np.array_equal(scene[:300], img[:300]), "the picture itself is not covered above the box")
ov.phase, ov.stood_at, ov.active = "stood", 1.9, {}
f2 = draw_frame(img, ov)
check(near(f2[:HEADER], GREEN) > 100 and near(f2[HEADER:], RED) == 0,
      "STOOD UP in green, no red text without a conflict")
ov.measure = False
check(ov.counter() == "", "no counter when conflicts are not measured")
ov.new_episode("x " * 200)
check(draw_frame(img[:240, :320], ov).shape == (240 + HEADER, 320, 3), "a long title is cut, small frames work")

print("red links (stand-in model)")
bodies = ["world", "pelvis", "right_hip_roll_link", "right_wrist_roll_rubber_hand", "torso_link"]
m = SimpleNamespace(ngeom=6, geom_bodyid=np.array([0, 1, 2, 2, 3, 4]),
                    geom_rgba=np.tile(np.array([0.7, 0.7, 0.7, 1.0], np.float32), (6, 1)))
m.geom_rgba[3, 3] = 0.0                        # a hidden shape stays hidden
m.body = lambda name: SimpleNamespace(id=bodies.index(name))
orig = m.geom_rgba.copy()
hl = Highlighter(m)
hl.set({"right_hip_roll_link - right_wrist_roll_rubber_hand": 0.08})
red = [g for g in range(6) if m.geom_rgba[g, 0] > 0.9 and m.geom_rgba[g, 1] < 0.2]
check(red == [2, 3, 4] and m.geom_rgba[3, 3] == 0.0 and m.geom_rgba[2, 3] == 1.0,
      f"both links red (all their shapes), alpha kept: {red}")
hl.set({"pelvis - torso_link": 0.02})
red = [g for g in range(6) if m.geom_rgba[g, 0] > 0.9 and m.geom_rgba[g, 1] < 0.2]
check(red == [1, 5] and np.array_equal(m.geom_rgba[2:5], orig[2:5]), f"the others back: {red}")
hl.clear()
check(np.array_equal(m.geom_rgba, orig), "all colours back")

print("mp4 and side by side")
try:
    import imageio_ffmpeg  # noqa: F401
    have_ffmpeg = True
except ImportError:
    have_ffmpeg = False
    print("  (imageio-ffmpeg not installed here: skipped)")
if have_ffmpeg:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "a").mkdir()
    (tmp / "b").mkdir()
    for folder, n, col in (("a", 30, (200, 40, 40)), ("b", 45, (40, 40, 200))):
        for name in ("ep01_supine.mp4", "ep02_prone.mp4"):
            w = Mp4Writer(tmp / folder / name, 25)
            for i in range(n):
                fr = np.zeros((120, 160, 3), np.uint8)
                fr[:] = col
                fr[:, :i * 3] = 255 if folder == "a" else 0
                w.add(fr)
            check(w.close(), f"{folder}/{name}: {n} frames written")
    fps, size, frames = read_mp4(tmp / "a" / "ep01_supine.mp4")
    got = list(frames)
    check(abs(fps - 25) < 0.01 and size == (160, 120) and len(got) == 30,
          f"read back: {fps} fps, {size}, {len(got)} frames")
    out = Mp4Writer(tmp / "pair.mp4", 25)
    k = side_by_side(tmp / "a" / "ep01_supine.mp4", tmp / "b" / "ep01_supine.mp4", [out])
    out.close()
    _, size, frames = read_mp4(tmp / "pair.mp4")
    got = list(frames)
    last = got[-1].astype(int)
    check(k == 45 and len(got) == 45 and size == (160 + 8 + 160, 120),
          f"the longer one sets the length, 8 px bar: {k} frames, {size}")
    check(last[:, 164].min() > 230 and last[60, 150, 0] > 150 and last[60, 150, 2] < 100
          and last[60, 318, 2] > 150 and last[60, 318, 0] < 100,
          "left red (its last frame kept), white bar, right blue")
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "side_by_side.py"), str(tmp / "a"),
                        str(tmp / "b"), "--out", str(tmp / "ab")], capture_output=True, text=True)
    print("    " + r.stdout.strip().replace("\n", "\n    "))
    ok = r.returncode == 0 and (tmp / "ab" / "all.mp4").is_file() and (tmp / "ab" / "ep02_prone.mp4").is_file()
    n_all = len(list(read_mp4(tmp / "ab" / "all.mp4")[2])) if ok else 0
    check(ok and n_all == 90, f"side_by_side.py: one video per start and all.mp4 ({n_all} frames)")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("ALL OK" if OK else "SOME CHECKS FAILED")
sys.exit(0 if OK else 1)
