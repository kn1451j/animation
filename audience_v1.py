"""
Turn the v1 audience clips into RGBA line-art sequences.

v2 and v3 shipped as PNG sequences with a real alpha channel; v1 exists only as
H.264 (`audience-level-N-noise.mp4`, yuv420p), which has no alpha — it is black
ink on white paper. So the alpha is keyed back out of luminance here: paper goes
fully transparent, ink goes fully opaque, and the RGB is flattened to black to
match how every other hand-drawn sequence in this project ships.

Levels stay the nested, bottom-left-aligned crops they were in v3 (630x346,
1136x590, 1394x748), so the renderer's world placement is unchanged.

Writes:
    renders/audience/l1       level 1
    renders/audience/l2       level 2
    renders/audience/l3_full  level 3, un-split — audience_split.py reads this

Usage:  python3 audience_v1.py && python3 audience_split.py    (from repo root)
Needs:  pillow numpy, ffmpeg on PATH
"""

import os
import shutil
import subprocess
import numpy as np
from PIL import Image

SRC = "/Users/katerinanikiforova/Documents/katias-stuff/stuff/audience"
LEVELS = [
    ("audience-level-1-noise.mp4", "renders/audience/l1"),
    ("audience-level-2-noise.mp4", "renders/audience/l2"),
    ("audience-level-3-noise.mp4", "renders/audience/l3_full"),
]
TMP = "/tmp/_aud_v1"

# Luma keying. Anything at or above PAPER is background; at or below INK is
# solid line. The gap between them carries the antialiasing, which matters —
# these are 1-2 px strokes and a hard threshold would alias them badly.
PAPER = 232.0
INK = 60.0


def key_frame(path):
    rgb = np.array(Image.open(path).convert("RGB")).astype(np.float32)
    luma = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    a = np.clip((PAPER - luma) / (PAPER - INK), 0.0, 1.0)
    out = np.zeros((*a.shape, 4), np.uint8)      # RGB stays black
    out[:, :, 3] = (a * 255).astype(np.uint8)
    return out


def main():
    for clip, outdir in LEVELS:
        if os.path.isdir(TMP):
            shutil.rmtree(TMP)
        os.makedirs(TMP)
        subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(SRC, clip),
                        f"{TMP}/%03d.png"], check=True)
        frames = sorted(f for f in os.listdir(TMP) if f.endswith(".png"))

        os.makedirs(outdir, exist_ok=True)
        for old in os.listdir(outdir):            # v3 frames from a previous run
            if old.endswith(".png"):
                os.remove(os.path.join(outdir, old))

        opaque = 0
        for i, f in enumerate(frames):
            a = key_frame(os.path.join(TMP, f))
            opaque += int((a[:, :, 3] > 16).sum())
            Image.fromarray(a).save(os.path.join(outdir, f"frame-{i:03d}.png"))
        h, w = a.shape[:2]
        print(f"{clip} -> {outdir}: {len(frames)} frames, {w}x{h}, "
              f"mean ink {opaque/len(frames)/(w*h):.4f}")

    shutil.rmtree(TMP, ignore_errors=True)
    print("\nnow run: python3 audience_split.py")


if __name__ == "__main__":
    main()
