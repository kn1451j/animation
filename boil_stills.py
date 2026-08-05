"""
Bake the five hand-drawn "noise" loops down to one still each.

renders/man_noise and renders/audience/{l1,l2,l3_back,l3_front} ship as 100
frames apiece, but the frames are not animation — measured against frame 0, ink
sits a median of 0 px and a p90 of 1 px away (max 3-5 px). That is boil: one
drawing being redrawn, which FS_BOIL already reproduces at runtime and was
calibrated against renders/man_noise in the first place. So 500 frames of
1920x1080-class RGBA buy nothing that a single still plus the shader does not.

The stills come from the clean (un-noised) masters in katias-stuff, which are
~2x the baked resolution and land within 1.5 px of the baked canvases once
scaled. They are downscaled to exactly the baked dimensions so main.cpp's
placement math — all of which reads texW/texH — is untouched.

Level 3 is split into back/front so the bug can pass between the near and far
ink, using the same whole-blob rule as audience_split.py. Splitting one still
instead of 100 frames also fixes a real artefact: the per-frame blob labelling
reassigned ink between the two layers as the drawing boiled, which measured as a
p90 displacement of 20 px and a max of 142 px on l3_back — ink visibly popping
from one layer to the other. A single split cannot do that.

Outputs (RGB left black, shape in alpha, matching every other hand-drawn asset):
    renders/stills/man.png          328x560
    renders/stills/aud_l1.png       630x346
    renders/stills/aud_l2.png       1136x590
    renders/stills/aud_l3_back.png  1136x590
    renders/stills/aud_l3_front.png 1394x748

Usage:  python3 boil_stills.py        (from the repo root)
Needs:  pillow numpy scipy
"""

import os
import numpy as np
from PIL import Image
from scipy import ndimage

STUFF = "/Users/katerinanikiforova/Documents/katias-stuff/stuff/audience"
OUT = "renders/stills"

# Clean master -> baked canvas size the renderer expects.
PLAIN = [
    ("man-clean-alpha.png",        "man.png",    (328, 560)),
    ("audience-level-1-alpha.png", "aud_l1.png", (630, 346)),
    ("audience-level-2-alpha.png", "aud_l2.png", (1136, 590)),
]

L3_SRC = "audience-level-3-alpha.png"
L3_SIZE = (1394, 748)
SPLIT_RECT = (1136, 590)   # same rect audience_split.py cuts on
ALPHA_T = 16               # same threshold as main.cpp's centroid helper


def load_alpha(name, size):
    """Clean master's alpha, downscaled to the baked canvas size."""
    im = Image.open(os.path.join(STUFF, name)).convert("RGBA")
    a = im.split()[3].resize(size, Image.LANCZOS)
    return np.array(a)


def save_alpha(alpha, path):
    """Write RGB-black / alpha-shape RGBA, the hand-drawn asset convention."""
    h, w = alpha.shape
    out = np.zeros((h, w, 4), np.uint8)
    out[:, :, 3] = alpha
    Image.fromarray(out).save(path)
    print(f"  {path}  {w}x{h}  ink {float((alpha > ALPHA_T).mean() * 100):.2f}%")


def main():
    os.makedirs(OUT, exist_ok=True)

    for src, dst, size in PLAIN:
        save_alpha(load_alpha(src, size), os.path.join(OUT, dst))

    # Level 3: assign each connected ink blob to back or front exactly once.
    a3 = load_alpha(L3_SRC, L3_SIZE)
    h, w = a3.shape
    rw, rh = SPLIT_RECT
    oy = h - rh                      # top edge of the level-2 rect in the l3 canvas

    ink = a3 > ALPHA_T
    lab, n = ndimage.label(ink, structure=np.ones((3, 3), bool))
    back_mask = np.zeros((h, w), bool)
    for cid, sl in enumerate(ndimage.find_objects(lab), start=1):
        ys, xs = sl
        if xs.stop <= rw and ys.start >= oy:
            back_mask[sl] |= lab[sl] == cid
    front_mask = ink & ~back_mask
    print(f"  l3: {n} blobs, back {int(back_mask.sum())} px, front {int(front_mask.sum())} px")

    back = np.where(back_mask, a3, 0)
    save_alpha(back[oy:, :rw], os.path.join(OUT, "aud_l3_back.png"))
    save_alpha(np.where(front_mask, a3, 0), os.path.join(OUT, "aud_l3_front.png"))


if __name__ == "__main__":
    main()
