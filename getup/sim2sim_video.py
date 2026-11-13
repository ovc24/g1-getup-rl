"""Pictures of the sim2sim runs (no MuJoCo here, so checks/ can test them): the header of
each video frame (the policy, the start, what the robot is doing, the link conflicts so
far), the links in a conflict drawn red, an mp4 writer that streams the frames, and two
runs side by side (scripts/side_by_side.py).

The header is a black band above the picture, as in the team's benchmark video:
    getup_v6 · prone 2/5                                        STOOD UP at 1.9 s
    link conflicts > 1 cm:  getting up 0.12 s,  standing 3.40 s              t 5.2 s
and at the bottom of the picture, in red, the pairs of links in a conflict at that moment
(those two links are drawn red on the robot too)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HEADER = 76                       # px of the black band above the picture
WHITE, GRAY, LIGHT = (255, 255, 255), (150, 150, 150), (205, 205, 205)
GREEN, AMBER, RED = (90, 225, 110), (255, 190, 60), (255, 75, 60)
RED_RGB = (0.95, 0.10, 0.08)      # links in a conflict (the alpha of each shape is kept)
SCENARIO_TEXT = {"supine": "supine", "prone": "prone", "facedown": "face-down fall",
                 "collapse": "limp collapse", "pool": "pool pose"}


def short_link(name: str) -> str:
    """right_wrist_roll_rubber_hand -> right hand; left_hip_roll_link -> left hip roll."""
    n = name.replace("_rubber_hand", "_hand").replace("wrist_roll_hand", "hand")
    if n.endswith("_link"):
        n = n[:-len("_link")]
    return n.replace("_", " ")


def pair_label(pair: str, sep: str = " - ") -> str:
    """'left_hip_roll_link - left_wrist_roll_rubber_hand' -> 'left hip roll - left hand'."""
    return sep.join(short_link(p) for p in pair.split(" - "))


def policy_label(pol_path: Path) -> str:
    """policy/<name>/exported/policy.onnx -> <name>; <run>/policy.onnx -> <run>."""
    pol_path = Path(pol_path)
    if pol_path.parent.name == "exported":
        return pol_path.parent.parent.name
    if pol_path.name == "policy.onnx":
        return pol_path.parent.name
    return pol_path.stem


@dataclass
class Overlay:
    """What the header shows; the episode loop keeps it up to date."""
    label: str = ""                   # the policy
    measure: bool = False             # link conflicts measured
    limit_cm: float = 1.0
    title: str = ""
    t: float | None = None            # s since the policy started; None while limp
    phase: str = "limp"               # limp, up, standing (< 1 s so far), stood, fell
    stood_at: float | None = None     # when the stand of >= 1 s began, s
    up_s: float = 0.0                 # time with a conflict: getting up, standing (s)
    stand_s: float = 0.0
    active: dict = field(default_factory=dict)     # pair -> depth (m) beyond the limit now

    def new_episode(self, title: str) -> None:
        self.title, self.t, self.phase, self.stood_at = title, None, "limp", None
        self.up_s = self.stand_s = 0.0
        self.active = {}

    def badge(self) -> tuple[str, tuple]:
        if self.phase == "limp":
            return "motors limp", GRAY
        if self.phase == "stood":
            return f"STOOD UP at {self.stood_at:.1f} s", GREEN
        if self.phase == "standing":
            return "standing", GREEN
        if self.phase == "fell":
            return "fell again", RED
        return "getting up", AMBER

    def counter(self) -> str:
        if not self.measure:
            return ""
        return (f"link conflicts > {self.limit_cm:g} cm:  getting up {self.up_s:.2f} s,  "
                f"standing {self.stand_s:.2f} s")

    def now_lines(self, sep: str = " - ", top: int = 3) -> list[str]:
        items = sorted(self.active.items(), key=lambda kv: -kv[1])
        out = [f"{pair_label(p, sep)}  {100 * d:.1f} cm" for p, d in items[:top]]
        if len(items) > top:
            out.append(f"+ {len(items) - top} more")
        return out


_FONTS: dict = {}


def font(size: int, bold: bool = False):
    """DejaVu Sans (Ubuntu, or matplotlib's copy), else Pillow's own font."""
    key = (size, bold)
    if key in _FONTS:
        return _FONTS[key]
    from PIL import ImageFont
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    paths = [name, "/usr/share/fonts/truetype/dejavu/" + name]
    try:
        import matplotlib
        paths.append(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name))
    except Exception:  # noqa: BLE001
        pass
    f = None
    for p in paths:
        try:
            f = ImageFont.truetype(p, size)
            break
        except Exception:  # noqa: BLE001
            continue
    if f is None:
        try:
            f = ImageFont.load_default(size=size)
        except TypeError:             # Pillow < 10.1
            f = ImageFont.load_default()
    _FONTS[key] = f
    return f


def _arrow(f) -> str:
    """The pair separator: an arrow with DejaVu, a dash with other fonts."""
    return " ↔ " if "DejaVu" in str(getattr(f, "path", "")) else " - "


def _fit(dr, text: str, f, width: float) -> str:
    if dr.textlength(text, font=f) <= width:
        return text
    while text and dr.textlength(text + "…", font=f) > width:
        text = text[:-1]
    return text + "…"


def draw_frame(img: np.ndarray, ov: Overlay) -> np.ndarray:
    """The picture with the header band above it and the conflicts of this moment."""
    from PIL import Image, ImageDraw
    img = np.ascontiguousarray(np.asarray(img)[..., :3], dtype=np.uint8)
    h, w = img.shape[:2]
    canvas = Image.new("RGB", (w, h + HEADER), (0, 0, 0))
    canvas.paste(Image.fromarray(img), (0, HEADER))
    dr = ImageDraw.Draw(canvas, "RGBA")
    dr.rectangle((0, HEADER - 2, w, HEADER - 1), fill=(70, 70, 70))     # the band's edge
    big, mid, bold = font(22, True), font(16), font(17, True)
    badge, col = ov.badge()
    bw = dr.textlength(badge, font=bold)
    dr.text((w - 12 - bw, 12), badge, font=bold, fill=col)
    room = w - 36 - bw
    for size in (22, 20, 18):              # a long title: a smaller font, then cut
        big = font(size, True)
        if dr.textlength(ov.title, font=big) <= room:
            break
    dr.text((12, 9 + (22 - size) // 2), _fit(dr, ov.title, big, room), font=big, fill=WHITE)
    tt = f"t {ov.t:.1f} s" if ov.t is not None else ""
    tw = dr.textlength(tt, font=mid) if tt else 0.0
    if tt:
        dr.text((w - 12 - tw, 46), tt, font=mid, fill=GRAY)
    c = ov.counter()
    if c:
        dr.text((12, 46), _fit(dr, c, mid, w - 36 - tw), font=mid, fill=LIGHT)
    lines = ov.now_lines(_arrow(bold))
    if lines:
        lh = 23
        y1 = h + HEADER - 8
        y0 = y1 - 10 - lh * len(lines)
        bx = min(max(dr.textlength(s, font=bold) for s in lines) + 22, w - 12)
        dr.rectangle((6, y0, 6 + bx, y1), fill=(0, 0, 0, 165))
        for i, s in enumerate(lines):
            dr.text((17, y0 + 6 + i * lh), _fit(dr, s, bold, w - 34), font=bold, fill=RED)
    return np.asarray(canvas)


class Highlighter:
    """Draws the links of the pairs in a conflict red by changing the model's shape colours
    (so the MuJoCo viewer shows it too), and puts the colours back."""

    def __init__(self, model, rgb=RED_RGB):
        self.m = model
        self.rgba0 = np.array(model.geom_rgba, copy=True)
        self.rgb = np.asarray(rgb, dtype=self.rgba0.dtype)
        self.geoms: dict[int, list[int]] = {}
        for g in range(int(model.ngeom)):
            self.geoms.setdefault(int(model.geom_bodyid[g]), []).append(g)
        self.ids: dict[str, int] = {}
        self.on: set[int] = set()

    def _id(self, name: str) -> int:
        if name not in self.ids:
            self.ids[name] = int(self.m.body(name).id)
        return self.ids[name]

    def set(self, pairs) -> None:
        bodies = {self._id(n) for p in pairs for n in p.split(" - ")}
        for b in self.on - bodies:
            for g in self.geoms.get(b, ()):
                self.m.geom_rgba[g] = self.rgba0[g]
        for b in bodies - self.on:
            for g in self.geoms.get(b, ()):
                self.m.geom_rgba[g, :3] = self.rgb
        self.on = bodies

    def clear(self) -> None:
        self.set(())


class Mp4Writer:
    """Streams frames to an mp4 (H.264, plays on phones and in browsers) with the ffmpeg of
    imageio-ffmpeg, so a long video does not have to fit in memory."""

    def __init__(self, path: Path, fps: int):
        self.path, self.fps, self.w, self.size = Path(path), fps, None, None

    def add(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
        h, w = frame.shape[:2]
        if self.w is None:
            import imageio_ffmpeg
            self.size = (w, h)
            self.w = imageio_ffmpeg.write_frames(str(self.path), (w, h), fps=self.fps,
                                                 macro_block_size=2, ffmpeg_log_level="error")
            self.w.send(None)
        if (w, h) != self.size:
            raise ValueError(f"frame {w}x{h}, the video is {self.size[0]}x{self.size[1]}")
        self.w.send(frame)

    def close(self) -> bool:
        if self.w is None:
            return False
        self.w.close()
        self.w = None
        return True


def read_mp4(path: Path):
    """(fps, (width, height), frame iterator) of an mp4, read as a stream."""
    import imageio_ffmpeg
    gen = imageio_ffmpeg.read_frames(str(path))
    meta = next(gen)
    w, h = meta["size"]

    def frames():
        for raw in gen:
            yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    return float(meta.get("fps") or 25.0), (int(w), int(h)), frames()


def side_by_side(left: Path, right: Path, writers, sep: int = 8) -> int:
    """Left and right video next to each other (a white bar between them), frame by frame;
    the shorter one stays on its last frame. Each frame goes to every writer. Returns the
    number of frames."""
    _, (wl, hl), fl = read_mp4(left)
    _, (wr, hr), fr = read_mp4(right)
    h = max(hl, hr)
    h += h % 2
    width = wl + sep + wr
    width += width % 2
    a = b = None
    n = 0
    while True:
        xa, xb = next(fl, None), next(fr, None)
        if xa is None and xb is None:
            break
        a = xa if xa is not None else a
        b = xb if xb is not None else b
        frame = np.zeros((h, width, 3), dtype=np.uint8)
        frame[:, wl:wl + sep] = 255
        if a is not None:
            frame[:hl, :wl] = a
        if b is not None:
            frame[:hr, wl + sep:wl + sep + wr] = b
        for wtr in writers:
            wtr.add(frame)
        n += 1
    return n


def tmp_name(out_dir: Path) -> Path:
    return Path(out_dir) / f".recording-{os.getpid()}.mp4"
