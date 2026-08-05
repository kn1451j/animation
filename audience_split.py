"""
Split the level-3 audience sequence into two layers so the bug clip can be
composited between them.

The three audience levels are nested, bottom-left-aligned crops of one master
drawing: level 1 (630x346) sits inside level 2 (1136x590) at (0, 244), and
level 2 sits inside level 3 (1394x748) at (0, 158). "Level-2 content" is
therefore the bottom-left L2_W x L2_H rectangle of a level-3 frame.

Cutting on that rectangle alone would slice figures in half wherever the bug
passes the boundary, so we assign whole connected ink blobs instead: a blob
goes BACK if its bounding box lies entirely inside the level-2 rect, else
FRONT. In practice the shoulder lines of the receding rows form one connected
chain that pokes above the rect, so the effective boundary lands just past the
nearest three figures — that is intentional, and slides with SPLIT_RECT below.

back + front recomposite to exactly the source frame: every opaque pixel lands
in exactly one of the two outputs.

Usage:  python3 audience_split.py         (from the repo root)
Needs:  pillow numpy scipy
"""

import os
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = "renders/audience/l3_full"      # written by audience_v1.py
OUT_BACK = "renders/audience/l3_back"
OUT_FRONT = "renders/audience/l3_front"

# Level-2 crop rect inside the level-3 canvas, as (width, height) anchored to
# the bottom-left corner. Shrink/grow to move the layer boundary.
SPLIT_RECT = (1136, 590)

ALPHA_T = 16  # same opacity threshold main.cpp's centroid helper uses


def main():
    os.makedirs(OUT_BACK, exist_ok=True)
    os.makedirs(OUT_FRONT, exist_ok=True)

    names = sorted(f for f in os.listdir(SRC) if f.endswith(".png") and not f.startswith("._"))
    if not names:
        raise SystemExit(f"no PNGs in {SRC}")

    rw, rh = SPLIT_RECT
    conn = np.ones((3, 3), bool)  # 8-connectivity: diagonal ink still counts as one stroke
    n_back_total = n_front_total = 0

    for i, name in enumerate(names):
        rgba = np.array(Image.open(os.path.join(SRC, name)).convert("RGBA"))
        h, w = rgba.shape[:2]
        oy = h - rh  # top edge of the level-2 rect within the level-3 canvas

        ink = rgba[:, :, 3] > ALPHA_T
        lab, n = ndimage.label(ink, structure=conn)

        back_mask = np.zeros((h, w), bool)
        for cid, sl in enumerate(ndimage.find_objects(lab), start=1):
            ys, xs = sl
            if xs.stop <= rw and ys.start >= oy:
                back_mask[sl] |= lab[sl] == cid

        # Any pixel below the alpha threshold is invisible in both layers, so
        # splitting on `ink` alone is enough to recomposite exactly.
        front_mask = ink & ~back_mask
        n_back_total += int(back_mask.sum())
        n_front_total += int(front_mask.sum())

        back = rgba.copy()
        back[~back_mask] = 0
        Image.fromarray(back[oy:, :rw]).save(os.path.join(OUT_BACK, name))

        front = rgba.copy()
        front[~front_mask] = 0
        Image.fromarray(front).save(os.path.join(OUT_FRONT, name))

        if i % 20 == 0:
            print(f"{name}: {n} blobs, back {back_mask.sum()} px, front {front_mask.sum()} px")

    print(f"\nwrote {len(names)} frames")
    print(f"  {OUT_BACK}  ({rw}x{rh})  {n_back_total} ink px")
    print(f"  {OUT_FRONT} (full canvas) {n_front_total} ink px")


if __name__ == "__main__":
    main()
