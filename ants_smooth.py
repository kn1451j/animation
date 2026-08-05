"""
Rebuild the title-screen ant frames with a smooth top edge and no bottom gap.

vids/moving_objects/ants/filter/ is the motion-isolated footage: a wavy band of
forest floor with black above AND below it. The title draws it shifted down so
the lower black is pushed off-screen, but the band's bottom edge sits at
y 633-815 — above the shifted screen bottom on the left and centre — which
leaves a black gap along the bottom of the terminal.

This regenerates the sequence from the unmasked source clip, keeping only the
band's TOP edge as a mask and filling everything beneath it with the original
footage. The edge is smoothed hard along x (the raw per-column boundary is
jagged from the isolation pass) and feathered vertically, so it reads as a
round horizon rather than a cut.

The edge is computed ONCE for the whole sequence, not per frame. The camera is
locked off, so the horizon has no business moving — but the isolation pass drops
and regains whole regions between frames, which swung the per-frame edge by up
to 180 px in a single step and read as a flicker. Taking the per-column median
across every frame throws those dropouts out and leaves the shape they agree on.
Nothing is lost by fixing it: the pixels below the edge come from the unmasked
clip either way, so the mask only ever chose where to stop.

Output is RGB on black, matching what SpriteSeq/the title renderer expect — the
title draws ants with blending off, so the black simply stays black.

Usage:  python3 ants_smooth.py         (from the repo root)
Needs:  pillow numpy scipy, ffmpeg on PATH
"""

import os
import shutil
import subprocess
import numpy as np
from PIL import Image
from scipy import ndimage

MASK_DIR = "vids/moving_objects/ants/filter"     # supplies the band's top edge
SRC_CLIP = "vids/moving_objects/ants/0001-0123.mp4"   # unmasked original
OUT = "renders/ants_smooth"
TMP = "/tmp/_ants_src"

EDGE_SMOOTH = 241   # px window for smoothing the top edge along x — bigger = rounder
FEATHER = 8         # px of vertical fade across the edge — just enough to
                    # antialias the curve; wider reads as a dark haze on the ground
DARK = 10           # luma at or below this counts as masked-out black


def extract_source(n_frames):
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", SRC_CLIP,
         "-frames:v", str(n_frames), f"{TMP}/%04d.png"],
        check=True)
    return sorted(f for f in os.listdir(TMP) if f.endswith(".png"))


def raw_edge(mask):
    """First masked row per column; NaN where the column is empty top to
    bottom, so those columns abstain from the median instead of voting 0."""
    idx = np.argmax(mask, axis=0).astype(np.float64)
    idx[~mask.any(axis=0)] = np.nan
    return idx


def stable_edge(masks):
    """One horizon for the whole sequence: the per-column median of the raw
    edges, hole-filled, then smoothed hard along x."""
    raw = np.array([raw_edge(m) for m in masks])       # frames x width
    with np.errstate(invalid="ignore"):
        idx = np.nanmedian(raw, axis=0)
    h, w = masks[0].shape
    has = ~np.isnan(idx)
    if not has.any():
        return np.full(w, h / 2.0)
    xs = np.arange(w)
    idx = np.interp(xs, xs[has], idx[has])             # nearest-fill the holes
    idx = ndimage.uniform_filter1d(idx, size=EDGE_SMOOTH, mode="nearest")
    return ndimage.gaussian_filter1d(idx, sigma=EDGE_SMOOTH / 6.0, mode="nearest")


def main():
    names = sorted(f for f in os.listdir(MASK_DIR)
                   if f.endswith(".png") and not f.startswith("._"))
    if not names:
        raise SystemExit(f"no frames in {MASK_DIR}")
    src_names = extract_source(len(names))
    if len(src_names) < len(names):
        raise SystemExit(f"{SRC_CLIP} gave {len(src_names)} frames, need {len(names)}")

    masks = [np.array(Image.open(os.path.join(MASK_DIR, n)).convert("L")) > DARK
             for n in names]
    edge = stable_edge(masks)
    mh, mw = masks[0].shape
    del masks

    os.makedirs(OUT, exist_ok=True)
    scaled = None
    for i, name in enumerate(names):
        src = np.array(Image.open(os.path.join(TMP, src_names[i])).convert("RGB"))
        h, w = src.shape[:2]
        if scaled is None:                         # keep the edge in source pixels
            scaled = edge * (h / mh) if (mh, mw) != (h, w) else edge
            if (mh, mw) != (h, w):
                scaled = np.interp(np.linspace(0, mw - 1, w), np.arange(mw), scaled)
            # 0 above the edge, 1 below, smoothstepped across FEATHER px. The
            # edge never moves, so this ramp is built once and reused.
            ys = np.arange(h)[:, None]
            t = np.clip((ys - (scaled[None, :] - FEATHER / 2.0)) / FEATHER, 0.0, 1.0)
            alpha = (t * t * (3.0 - 2.0 * t))[:, :, None]
            print(f"edge y {scaled.min():.0f}-{scaled.max():.0f}, feather {FEATHER} px")

        out = (src.astype(np.float32) * alpha).astype(np.uint8)
        Image.fromarray(out).save(os.path.join(OUT, name))
        if i % 25 == 0:
            print(f"  {name}")

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nwrote {len(names)} frames to {OUT}/")


if __name__ == "__main__":
    main()
