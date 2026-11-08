#!/usr/bin/env python3
"""Keyboard pushes for Unitree's simulator (unitree_mujoco, C++ ``simulate``).

Pushing the robot with the mouse (double-click the torso, Ctrl+right drag) is hard when the
simulator lags. After this patch, with the simulator window focused:

    I / K       push forward / backward  (relative to where the robot faces)
    J / L       push left / right
    = / -       push force +-50 N (default 300 N, 50..1500)
    P           automatic pushes on/off: every 4-6 s, random direction, same force

A push is a horizontal force on the torso for 0.1 s, like the shoves of the walking
fine-tune (``eval/push_test.py`` in the walking repo): 300 N for 0.1 s changes the speed of
the whole robot by about 0.9 m/s, but it hits the torso high up. The simulator prints
every push in its terminal. The elastic band keys (7, 8, 9) and Backspace (reset) keep
working.

    python3 patch_unitree_mujoco.py --unitree-mujoco ~/unitree_mujoco
    cd ~/unitree_mujoco/simulate/build && make -j4

unitree_rl_mjlab has its own copy of the simulator (simulate/, same main.cc anchors): pass
that folder instead, --unitree-mujoco <unitree_rl_mjlab>.

It edits ``simulate/src/main.cc`` (copied once to ``main.cc.orig``); running it again
changes nothing. To undo: copy ``main.cc.orig`` back and build again. Standard library
only.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

MARK = "KeyboardPush"

PUSH_CODE = r"""

// ---- keyboard pushes on the torso (added by patch_unitree_mujoco.py) ----------------------
// I / K / J / L: push forward / backward / left / right (relative to where the robot faces)
// for 0.1 s; = / -: force +-50 N; P: automatic pushes every 4-6 s in a random direction.
#include <algorithm>
#include <cmath>
#include <random>
class KeyboardPush
{
public:
  void Request(int dir)  // 0 forward, 1 left, 2 backward, 3 right
  {
    std::lock_guard<std::mutex> lock(mtx_);
    request_ = dir;
  }
  void ChangeForce(double delta)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    newton_ = std::min(1500.0, std::max(50.0, newton_ + delta));
    std::printf("push force: %.0f N\n", newton_);
  }
  void ToggleAuto()
  {
    std::lock_guard<std::mutex> lock(mtx_);
    auto_ = !auto_;
    next_auto_ = -1.0;
    std::printf("automatic pushes: %s\n", auto_ ? "on (every 4-6 s, random direction)" : "off");
  }
  // Physics thread, right before each mj_step (sim mutex held). link6 = 6 x torso body id.
  void Apply(const mjModel *m, mjData *d, int link6)
  {
    if (!m || !d || link6 <= 0)
      return;
    std::lock_guard<std::mutex> lock(mtx_);
    if (d->time < last_time_)  // reset (Backspace)
    {
      steps_left_ = 0;
      clear_ = true;
      next_auto_ = -1.0;
    }
    last_time_ = d->time;
    if (auto_ && steps_left_ == 0 && request_ < 0)
    {
      if (next_auto_ < 0.0)
        next_auto_ = d->time + 4.0 + 2.0 * uniform_(rng_);
      else if (d->time >= next_auto_)
      {
        request_ = static_cast<int>(4.0 * uniform_(rng_)) % 4;
        next_auto_ = -1.0;
      }
    }
    if (request_ >= 0)
    {
      const int body = link6 / 6;
      const double heading = std::atan2(d->xmat[9 * body + 3], d->xmat[9 * body + 0]);
      const double angle = heading + request_ * 1.5707963267948966;
      f_[0] = newton_ * std::cos(angle);
      f_[1] = newton_ * std::sin(angle);
      steps_left_ = std::max(1, static_cast<int>(std::lround(0.1 / m->opt.timestep)));
      static const char *names[4] = {"forward", "left", "backward", "right"};
      std::printf("push %s: %.0f N for 0.1 s (t = %.1f s)\n", names[request_], newton_, d->time);
      request_ = -1;
    }
    if (steps_left_ > 0 || clear_)
    {
      // absolute values: the elastic band (if on) + this push; when the push ends, the band only
      const bool band = param::config.enable_elastic_band == 1 && elastic_band.enable_;
      const bool on = steps_left_ > 0;
      for (int k = 0; k < 3; ++k)
        d->xfrc_applied[link6 + k] = (band ? elastic_band.f_[k] : 0.0) + (on && k < 2 ? f_[k] : 0.0);
      if (on)
      {
        --steps_left_;
        clear_ = steps_left_ == 0;
      }
      else
        clear_ = false;
    }
  }

private:
  std::mutex mtx_;
  double newton_ = 300.0;
  int request_ = -1;
  int steps_left_ = 0;
  bool clear_ = false;
  bool auto_ = false;
  double next_auto_ = -1.0;
  double last_time_ = 0.0;
  double f_[2] = {0.0, 0.0};
  std::mt19937 rng_{12345};
  std::uniform_real_distribution<double> uniform_{0.0, 1.0};
};
inline KeyboardPush keyboard_push;
// ---- end of keyboard pushes ----------------------------------------------------------------
"""

KEY_CODE = r"""
    // keyboard pushes (added by patch_unitree_mujoco.py)
    switch (key)
    {
    case GLFW_KEY_I: keyboard_push.Request(0); break;
    case GLFW_KEY_J: keyboard_push.Request(1); break;
    case GLFW_KEY_K: keyboard_push.Request(2); break;
    case GLFW_KEY_L: keyboard_push.Request(3); break;
    case GLFW_KEY_EQUAL: case GLFW_KEY_KP_ADD: keyboard_push.ChangeForce(50.0); break;
    case GLFW_KEY_MINUS: case GLFW_KEY_KP_SUBTRACT: keyboard_push.ChangeForce(-50.0); break;
    case GLFW_KEY_P: keyboard_push.ToggleAuto(); break;
    default: break;
    }"""

BAND_ANCHOR = "inline ElasticBand elastic_band;"
STEP_RE = re.compile(r"^(?P<indent>[ \t]*)mj_step\(m, d\);", re.M)
KEY_RE = re.compile(r"void user_key_cb\(GLFWwindow\s*\*\s*window,\s*int key,\s*int scancode,\s*int act,"
                    r"\s*int mods\)\s*\{\s*if\s*\(act\s*==\s*GLFW_PRESS\)\s*\{")


def patch(text: str) -> tuple[str, list[str]]:
    """The patched main.cc and what was done; raises ValueError if an anchor is missing."""
    notes = []
    if text.count(BAND_ANCHOR) != 1:
        raise ValueError(f"'{BAND_ANCHOR}' not found once (found {text.count(BAND_ANCHOR)})")
    text = text.replace(BAND_ANCHOR, BAND_ANCHOR + PUSH_CODE)
    notes.append("KeyboardPush class after the elastic band")

    steps = list(STEP_RE.finditer(text))
    if not steps:
        raise ValueError("no 'mj_step(m, d);' found")
    text = STEP_RE.sub(lambda mt: f"{mt['indent']}keyboard_push.Apply(m, d, param::config.band_attached_link);\n"
                                  f"{mt['indent']}mj_step(m, d);", text)
    notes.append(f"push applied before {len(steps)} mj_step call(s)"
                 + ("" if len(steps) == 2 else "  (Unitree's file has 2: check the result)"))

    km = KEY_RE.search(text)
    if not km:
        raise ValueError("keyboard callback 'user_key_cb' with 'if (act==GLFW_PRESS) {' not found")
    text = text[:km.end()] + KEY_CODE + text[km.end():]
    notes.append("keys I/J/K/L, =/-, P in user_key_cb")
    return text, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--unitree-mujoco", required=True,
                    help="the folder with simulate/src/main.cc: unitree_mujoco, or "
                         "unitree_rl_mjlab (it has its own copy of the simulator)")
    args = ap.parse_args()
    src = Path(args.unitree_mujoco).expanduser() / "simulate" / "src" / "main.cc"
    if not src.is_file():
        print(f"not found: {src}")
        return 1
    text = src.read_text()
    if MARK in text:
        print(f"{src}: already patched (keyboard pushes); nothing changed")
        return 0
    try:
        new, notes = patch(text)
    except ValueError as e:
        print(f"{src}: {e}. Unitree's file may have changed; nothing written.")
        return 1
    orig = src.with_name(src.name + ".orig")
    if not orig.exists():
        shutil.copy2(src, orig)
    src.write_text(new)
    for n in notes:
        print("  " + n)
    print(f"patched {src} (original in {orig.name}). Build again:\n"
          f"  cd {src.parent.parent / 'build'} && make -j4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
