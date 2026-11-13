#!/usr/bin/env python3
"""Two runs of sim2sim_getup.py --video next to each other, as in the team's benchmark
video: for every epNN_*.mp4 in both folders, LEFT | RIGHT with a white bar between them
(the shorter one stays on its last frame), plus all.mp4 with every pair one after the
other.

    G=<this repo>
    python $G/sim2sim_getup.py --policy <run v5> --label getup_v5 --benchmark 5 --video videos/bench_v5
    python $G/sim2sim_getup.py --policy <run v6> --label getup_v6 --benchmark 5 --video videos/bench_v6
    python $G/scripts/side_by_side.py videos/bench_v5 videos/bench_v6 --out videos/v5_vs_v6

The same --seed (default 0) gives both runs the same starts (same poses, headings and
falls), so each pair shows the two policies from the same start and the same camera.
Needs imageio-ffmpeg (in the mjlab environment: it writes the videos of training)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from getup.sim2sim_video import Mp4Writer, read_mp4, side_by_side  # noqa: E402


class SameSize:
    """A writer that skips frames of another size (all.mp4 needs one size)."""

    def __init__(self, writer: Mp4Writer):
        self.w, self.skipped = writer, 0

    def add(self, frame):
        if self.w.size is not None and (frame.shape[1], frame.shape[0]) != self.w.size:
            self.skipped += 1
            return
        self.w.add(frame)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left", help="folder of the run shown on the left")
    ap.add_argument("right", help="folder of the run shown on the right")
    ap.add_argument("--out", default=None,
                    help="output folder (default: <left>_vs_<right> next to the right one)")
    ap.add_argument("--no-all", action="store_true", help="no all.mp4")
    ap.add_argument("--sep", type=int, default=8, help="width of the white bar (px)")
    args = ap.parse_args(argv)

    left, right = Path(args.left).expanduser(), Path(args.right).expanduser()
    for d in (left, right):
        if not d.is_dir():
            raise SystemExit(f"no folder {d}")
    out = (Path(args.out).expanduser() if args.out
           else right.parent / f"{left.name}_vs_{right.name}")
    names = sorted(p.name for p in left.glob("*.mp4") if p.name != "all.mp4")
    pairs = [n for n in names if (right / n).is_file()]
    missing = sorted(set(names) ^ {p.name for p in right.glob("*.mp4") if p.name != "all.mp4"})
    if not pairs:
        raise SystemExit(f"no video with the same name in {left} and {right} (same --benchmark "
                         "and --seed in both runs give the same names)")
    out.mkdir(parents=True, exist_ok=True)
    fps, _, _ = read_mp4(left / pairs[0])
    every = None if args.no_all else SameSize(Mp4Writer(out / "all.mp4", int(round(fps))))
    for n in pairs:
        w = Mp4Writer(out / n, int(round(fps)))
        k = side_by_side(left / n, right / n, [w] + ([every] if every else []), sep=args.sep)
        w.close()
        print(f"  {n}: {k} frames")
    if every is not None:
        every.w.close()
        if every.skipped:
            print(f"  (all.mp4: {every.skipped} frames of another size left out)")
    print(f"{len(pairs)} videos side by side in {out}"
          + ("" if args.no_all else f", and all of them one after the other in {out / 'all.mp4'}"))
    if missing:
        print("only in one of the folders (not compared): " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
