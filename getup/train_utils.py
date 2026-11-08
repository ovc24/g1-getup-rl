"""Small PPO helpers for train_getup.py (rsl_rl 5.0.1 as used by mjlab 1.2.0)."""
from __future__ import annotations

import math

import torch


def _find_std(module) -> tuple[torch.nn.Parameter | None, bool]:
    """The action-noise parameter of an actor (rsl_rl: its Gaussian distribution's
    std_param or log_std_param; older versions: std / log_std on the module)."""
    for owner in (getattr(module, "distribution", None), module):
        for attr, is_log in (("std_param", False), ("log_std_param", True),
                             ("std", False), ("log_std", True)):
            p = getattr(owner, attr, None) if owner is not None else None
            if isinstance(p, torch.nn.Parameter):
                return p, is_log
    return None, False


def install_std_limit(runner, max_std: float, min_std: float = 0.05) -> float | None:
    """After every PPO update, clamp the action noise (std of each joint) to
    [min_std, max_std]. With an entropy bonus the noise can keep growing where it costs
    little (e.g. the arms) until the motions get violent. runner: the rsl_rl runner (or
    its .alg). Returns the largest std before the first clamp (e.g. where the loaded
    checkpoint ended), None if no std parameter was found."""
    alg = getattr(runner, "alg", runner)
    param, is_log = None, False
    for m in (getattr(alg, "actor", None), getattr(alg, "policy", None)):
        if m is not None:
            param, is_log = _find_std(m)
            if param is not None:
                break
    if param is None and hasattr(runner, "get_inference_policy"):
        try:
            param, is_log = _find_std(runner.get_inference_policy())
        except Exception:  # noqa: BLE001
            param = None
        if hasattr(runner, "train_mode"):
            runner.train_mode()             # get_inference_policy switched to eval mode
    if param is None:
        return None
    with torch.no_grad():
        before = float(param.max().exp() if is_log else param.max())
    lo, hi = float(min_std), float(max_std)
    if is_log:
        lo, hi = math.log(lo), math.log(hi)
    orig_update = alg.update

    def update(*args, **kwargs):
        out = orig_update(*args, **kwargs)
        with torch.no_grad():
            param.clamp_(min=lo, max=hi)
        return out

    alg.update = update
    with torch.no_grad():
        param.clamp_(min=lo, max=hi)
    return before
