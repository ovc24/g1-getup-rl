"""Link clearance in training and evaluation, with stand-ins of the collision shapes of
Unitree's deploy scene: how far each arm link is from the body, as the self-collision
benchmark measures it in that scene (sim2sim_getup.py --benchmark), not with the
training model's own capsules (a hand that only touches the training model's hip
capsule can be 8 cm inside the hip of Unitree's meshes).

The stand-ins (clearance_shapes.py, made by scripts/make_clearance_shapes.py): per
collision geom, its convex hull wrapped in 26 planes (a 26-DOP) and 32 points spread over
it, in its body's frame. Clearance of a pair of links: for the points of one shape, the
largest signed distance to the planes of the other (negative inside: how deep), the
smallest over the points; both ways; the smallest over the shapes of the two links.
Negative = one link that far into the other. Near contact it reads ~0.5 cm closer than the
exact hulls (p10 -1.4 cm), so it errs on the safe side; in random poses it found all the
conflicts deeper than 1 cm.

Pairs (clearance_shapes.PAIRS): an arm link (shoulder yaw, elbow, hand) and the torso,
the pelvis, a leg link or an arm link of the other arm. In the walking pose all are at
least 2.45 cm apart (the stand-ins: 2.09 cm)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from . import clearance_shapes as cs

CONFLICT_M = 0.01      # a conflict: one link more than 1 cm into another (the benchmark)
MARGIN_M = 0.01        # the cost starts 1 cm before contact (training, getup_env_cfg.py)
_CHUNK = 8             # shapes measured against one shape at a time (GPU memory)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternions w, x, y, z -> (..., 3, 3) rotation matrices."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)


def pair_name(pair) -> str:
    return " - ".join(sorted(pair))


def check_scene(scene: Path) -> str | None:
    """None if the stand-ins were made from this scene file, else a warning."""
    try:
        digest = hashlib.sha1(Path(scene).read_bytes()).hexdigest()[:12]
    except OSError:
        return None
    if digest == cs.SCENE_SHA1:
        return None
    return (f"the link-clearance stand-ins (getup/clearance_shapes.py, scene sha1 "
            f"{cs.SCENE_SHA1}) were made from another {Path(scene).name} (sha1 {digest}): "
            f"make them again (needs scipy): python scripts/make_clearance_shapes.py --scene {scene}")


class LinkClearance:
    """clearance(pos, quat) -> (N, pairs) signed clearance in m, pairs = self.pairs.
    pos (N, B, 3) and quat (N, B, 4, w x y z) of the bodies in self.bodies, world frame."""

    def __init__(self, device=None, shapes=cs.SHAPES, pairs=cs.PAIRS, dirs=cs.DIRS):
        self.pairs = [tuple(p) for p in pairs]
        self.names = [pair_name(p) for p in self.pairs]
        self.bodies = sorted({b for p in self.pairs for b in p})
        bidx = {b: i for i, b in enumerate(self.bodies)}
        shapes = [s for s in shapes if s["body"] in bidx]
        missing = sorted(set(self.bodies) - {s["body"] for s in shapes})
        if missing:
            raise ValueError(f"no stand-in shapes for {missing}")
        self.shape_body = [bidx[s["body"]] for s in shapes]
        f32 = torch.float32
        pts = torch.tensor([[list(p) for p in s["points"]] for s in shapes], dtype=f32,
                           device=device)                                   # (S, P, 3)
        self.points_t = pts.transpose(1, 2)                                 # (S, 3, P)
        self.offsets = torch.tensor([list(s["offsets"]) for s in shapes], dtype=f32,
                                    device=device)                          # (S, K)
        self.dirs_t = torch.tensor([list(d) for d in dirs], dtype=f32,
                                   device=device).transpose(0, 1)           # (3, K)
        self.sb = torch.tensor(self.shape_body, dtype=torch.long, device=device)
        of: dict[int, list[int]] = {}
        for i, b in enumerate(self.shape_body):
            of.setdefault(b, []).append(i)
        evals = []                       # (source shape, target shape, pair)
        for q, (a, b) in enumerate(self.pairs):
            for sa in of[bidx[a]]:
                for sb in of[bidx[b]]:
                    evals += [(sa, sb, q), (sb, sa, q)]
        self.n_evals = len(evals)
        by_target: dict[int, tuple[list[int], list[int]]] = {}
        for e, (s, t, _) in enumerate(evals):
            src, cols = by_target.setdefault(t, ([], []))
            src.append(s)
            cols.append(e)
        self.groups = []                 # (target shape, its body, sources, result columns)
        for t, (src, cols) in sorted(by_target.items()):
            for k in range(0, len(src), _CHUNK):
                self.groups.append((t, self.shape_body[t],
                                    torch.tensor(src[k:k + _CHUNK], dtype=torch.long, device=device),
                                    torch.tensor(cols[k:k + _CHUNK], dtype=torch.long, device=device)))
        per_pair: list[list[int]] = [[] for _ in self.pairs]
        for e, (_, _, q) in enumerate(evals):
            per_pair[q].append(e)
        width = max(len(c) for c in per_pair)
        self.reduce = torch.tensor([c + [self.n_evals] * (width - len(c)) for c in per_pair],
                                   dtype=torch.long, device=device)        # (Q, L), pad: +inf

    def __call__(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        return self.measure(pos, quat_to_mat(quat))

    def measure(self, pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        """The same with rotation matrices rot (N, B, 3, 3), body frame -> world."""
        n = pos.shape[0]
        world = ((rot[:, self.sb] @ self.points_t).transpose(-1, -2)
                 + pos[:, self.sb].unsqueeze(2))                          # (N, S, P, 3)
        vals = torch.full((n, self.n_evals + 1), float("inf"), device=pos.device,
                          dtype=pos.dtype)
        for t, bt, src, cols in self.groups:
            local = ((world[:, src] - pos[:, bt].reshape(n, 1, 1, 3))
                     @ rot[:, bt].unsqueeze(1))                            # (N, M, P, 3)
            vals[:, cols] = (local @ self.dirs_t - self.offsets[t]).amax(dim=-1).amin(dim=-1)
        return vals[:, self.reduce].amin(dim=-1)                          # (N, Q)
