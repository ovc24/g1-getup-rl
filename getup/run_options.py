"""Options of a training run (saved by train_getup.py in <run>/params/getup_options.json),
so that eval_getup.py and play_getup.py rebuild the same environment."""
from __future__ import annotations

import json
from pathlib import Path

from .poses import POSTURES


def parse_postures(text: str | None, ap=None) -> dict[str, float] | None:
    """'supine,prone' or 'supine=2,prone=1,left_side=0.5' -> posture weights."""
    if not text:
        return None
    w = {}
    for item in text.split(","):
        k, _, v = item.partition("=")
        k = k.strip()
        if k not in POSTURES:
            msg = f"unknown posture '{k}' (names: {', '.join(POSTURES)})"
            if ap is not None:
                ap.error(msg)
            raise ValueError(msg)
        w[k] = float(v) if v else 1.0
    return w


def load_run_options(checkpoint: str) -> dict:
    path = Path(checkpoint).expanduser().resolve().parent / "params" / "getup_options.json"
    if not path.exists():
        print(f"[WARNING] no {path}: using the defaults (action scale 0.5, realistic motors, "
              "history 5); pass --action-scale / --no-realistic if the run used others")
        return {}
    return json.loads(path.read_text())


# Runs made before the pace existed (their options file has no "pace_s"): no pace, no
# torque budget, speed penalty from 90 % of the no-load speed.
LEGACY_PACE = {"pace_s": 0.0, "torque_budget": 0.0, "speed_budget": 0.9}
PACE_KEYS = tuple(LEGACY_PACE)


def pace_options(opts: dict) -> dict[str, float]:
    """pace_s, torque_budget, speed_budget of a run (make_getup_env_cfg keywords)."""
    if "pace_s" not in opts:
        return dict(LEGACY_PACE)
    return {k: float(opts.get(k, LEGACY_PACE[k])) for k in PACE_KEYS}


def describe_pace(p: dict[str, float]) -> str:
    if p["pace_s"] <= 0.0 and p["torque_budget"] <= 0.0:
        return "no pace (a run from before the pace option)"
    return (f"pace {p['pace_s']:g} s, torque budget {100 * p['torque_budget']:.0f} %, "
            f"speed budget {100 * p['speed_budget']:.0f} %")


# Runs made before v5 (their options file has no "feet_high"): no cost for the feet above
# the pelvis, head costs x1, no joint friction/damping/armature randomization. Runs made
# before v6 (no "self_collision"): self-contacts cost nothing; before v7 (no "clearance"):
# no link-clearance cost.
LEGACY_STYLE = {"feet_high": 0.0, "head_scale": 1.0, "joint_dr": False, "self_collision": 0.0,
                "clearance": 0.0, "self_standing": 0.0}


def resolve_pool(path) -> "Path":
    """A pool path as given; a relative one that does not exist from here is looked for
    in this repo (a published policy's getup_options.json says data/<file>). A data/<file>
    found nowhere is reported where it belongs, in this repo."""
    p = Path(str(path)).expanduser()
    if not p.is_absolute() and not p.exists():
        in_repo = Path(__file__).resolve().parent.parent / p
        if in_repo.exists() or p.parts[:1] == ("data",):
            return in_repo
    return p


def style_options(opts: dict) -> dict:
    """feet_high, head_scale, joint_dr, self_collision, clearance of a run
    (make_getup_env_cfg keywords)."""
    if "feet_high" not in opts:
        return dict(LEGACY_STYLE)
    return {"feet_high": float(opts["feet_high"]),
            "head_scale": float(opts.get("head_scale", 1.0)),
            "joint_dr": bool(opts.get("joint_dr", False)),
            "self_collision": float(opts.get("self_collision", 0.0)),
            "clearance": float(opts.get("clearance", 0.0)),
            "self_standing": float(opts.get("self_standing", 0.0))}


def describe_style(s: dict) -> str:
    sc = float(s.get("self_collision", 0.0))
    cl = float(s.get("clearance", 0.0))
    return (f"feet above the pelvis cost {s['feet_high']:g}, head costs x{s['head_scale']:g}, "
            f"joint friction/damping/armature {'randomized' if s['joint_dr'] else 'fixed'}, "
            + (f"self-contacts while getting up cost {sc:g}" if sc > 0 else
               "self-contacts free")
            + (f", arm links closer than 1 cm to the body cost {cl:g} per cm" if cl > 0 else
               ", link clearance free")
            + (f", self-contacts standing cost x{float(s.get('self_standing', 0.0)):g}"
               if float(s.get("self_standing", 0.0)) > 0 else ""))
