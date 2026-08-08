"""
Rebuild the title-screen ant frames with a smooth top edge and no bottom gap.

vids/moving_objects/ants/filter/ is the motion-isolated footage the title used
to draw: a wavy band of forest floor with black above AND below it. The title
draws it shifted down so the lower black is pushed off-screen, but the band's
bottom edge sits at y 633-815 — above the shifted screen bottom on the left and
centre — which leaves a black gap along the bottom of the terminal.

This regenerates the sequence from the unmasked source clip, fading it out along
a horizon of our own rather than the mask's. The mask is not used at all now: it
was a motion-isolation pass, so its top edge landed exactly where ant motion
stopped registering — which is to say it ran through the topmost ants and cut
them in half, and it was lumpy and unstable frame to frame besides. A plain arc
is both cleaner and entirely under our control.

The horizon never moves, so there is nothing to flicker, and it is drawn well
above the ant trail (whose per-column top measures y 279 at the 5th percentile)
so no ant is ever sliced by it.

Output is RGB on black, matching what SpriteSeq/the title renderer expect — the
title draws ants with blending off, so the black simply stays black.

Usage:  python3 ants_smooth.py         (from the repo root)
Needs:  pillow numpy, ffmpeg on PATH
"""

import os
import shutil
import subprocess
import numpy as np
from PIL import Image

SRC_CLIP = "vids/moving_objects/ants/0001-0123.mp4"   # unmasked original
OUT = "renders/ants_smooth"
TMP = "/tmp/_ants_src"

# The renderer draws this texture shifted down by ANT_SHIFT_Y, so texture y and
# screen y differ by this much. Kept here only so the numbers printed below are
# in the terms you actually see. Must match main.cpp.
SHIFT_Y = 345

# The horizon: a plain arc, peaking at PEAK_Y in the middle and falling DOME px
# to either edge. PEAK_Y is bounded above by the title text, which ends near
# screen y 390 — allow FEATHER/2 on top of PEAK_Y before comparing.
PEAK_Y = 146        # texture y at the top of the arc (screen 491)
DOME = 300          # px the arc falls from centre to either edge

FEATHER = 80        # px of vertical fade across the horizon. Wide enough to read
                    # as the ground dissolving into the dark rather than a cut.
                    # It is centred on the curve, reaching FEATHER/2 either side.


def extract_source():
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP)
    subprocess.run(["ffmpeg", "-v", "error", "-i", SRC_CLIP, f"{TMP}/%04d.png"],
                   check=True)
    return sorted(f for f in os.listdir(TMP) if f.endswith(".png"))


def horizon(w):
    """Symmetric arc across the frame, in texture rows."""
    x = np.linspace(0.0, 1.0, w)
    return PEAK_Y + DOME * (1.0 - np.sin(np.pi * x))


def main():
    names = extract_source()
    if not names:
        raise SystemExit(f"{SRC_CLIP} gave no frames")

    os.makedirs(OUT, exist_ok=True)
    alpha = None
    for i, name in enumerate(names):
        src = np.array(Image.open(os.path.join(TMP, name)).convert("RGB"))
        h, w = src.shape[:2]
        if alpha is None:
            e = horizon(w)
            # 0 above the curve, 1 below, smoothstepped across FEATHER px. The
            # horizon never moves, so this ramp is built once and reused.
            ys = np.arange(h)[:, None]
            t = np.clip((ys - (e[None, :] - FEATHER / 2.0)) / FEATHER, 0.0, 1.0)
            alpha = (t * t * (3.0 - 2.0 * t))[:, :, None]
            print(f"horizon: peak screen y {e.min()+SHIFT_Y:.0f} "
                  f"(fading from {e.min()+SHIFT_Y-FEATHER/2:.0f}), "
                  f"edges {e.max()+SHIFT_Y:.0f}, dome {DOME} px, feather {FEATHER} px")

        out = (src.astype(np.float32) * alpha).astype(np.uint8)
        Image.fromarray(out).save(os.path.join(OUT, name))
        if i % 25 == 0:
            print(f"  {name}")

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nwrote {len(names)} frames to {OUT}/")


if __name__ == "__main__":
    main()
