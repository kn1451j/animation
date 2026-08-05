"""
Build the exit screen's four still textures.

The exit screen is a "vignette sphere" — the backyard-tree drawing from the
katias-stuff site, circularly masked with a soft fade, ringed by the circle
hand-drawn on the layout sketch. The sphere is a white field with the tree dark
on top of it; everything outside stays black. These outputs carry shape in the
alpha channel only — RGB is left white and the colour comes from the shader's
tint, so the same texture serves as white disc or black ink.

The tree ships as a clean/noise pair, which is what the site's `doodle-boil`
flips between at 8 fps. Both are masked with the identical vignette so the two
line up exactly and the boil reads as the line being redrawn, not as the whole
sphere shifting.

Outputs (SIZE x SIZE, RGBA), all sharing one circle centre and radius so the
renderer can place them with a single square rect:
    renders/exit_screen/disc.png        filled circle — the sphere's white field
    renders/exit_screen/tree_clean.png
    renders/exit_screen/tree_noise.png
    renders/exit_screen/rim.png

Usage:  python3 exit_screen_assets.py        (from the repo root)
Needs:  pillow numpy scipy
"""

import os
import numpy as np
from PIL import Image
from scipy import ndimage

STUFF = "/Users/katerinanikiforova/Documents/katias-stuff/stuff/drawings"
SKETCH = "/Users/katerinanikiforova/Downloads/IMG_3817.HEIC"
OUT = "renders/exit_screen"

SIZE = 1024          # square texture edge
RIM_R = 0.98         # drawn circle radius, as a fraction of SIZE/2
VIG_INNER = 0.55     # tree ink is untouched inside this radius…
VIG_OUTER = 0.98     # …and has faded to nothing by this one

SKETCH_BLUR = 201    # local-threshold window for the sketch photo
SKETCH_T = 18        # how far below local paper brightness counts as ink


def load_rgba(path):
    if path.lower().endswith(".heic"):
        png = "/tmp/_exit_sketch.png"
        if os.system(f'sips -s format png "{path}" --out {png} >/dev/null 2>&1') != 0:
            raise SystemExit(f"could not convert {path}")
        path = png
    return np.array(Image.open(path).convert("RGBA"))


def radial(size):
    """Distance from centre, normalised so 1.0 is the inscribed circle."""
    c = (size - 1) / 2.0
    y, x = np.ogrid[:size, :size]
    return np.sqrt((x - c) ** 2 + (y - c) ** 2) / c


def cover_square(alpha, size):
    """Scale so the drawing covers the square, cropping the long axis, kept
    centred on the ink so the trunk lands near the middle."""
    h, w = alpha.shape
    s = size / min(h, w)
    im = Image.fromarray(alpha).resize((int(round(w * s)), int(round(h * s))), Image.LANCZOS)
    a = np.array(im)
    h2, w2 = a.shape
    ys, xs = np.nonzero(a > 16)
    cx = int(xs.mean()) if len(xs) else w2 // 2
    cy = int(ys.mean()) if len(ys) else h2 // 2
    x0 = min(max(cx - size // 2, 0), max(w2 - size, 0))
    y0 = min(max(cy - size // 2, 0), max(h2 - size, 0))
    return a[y0:y0 + size, x0:x0 + size]


def white_with_alpha(alpha):
    out = np.zeros((*alpha.shape, 4), np.uint8)
    out[:, :, :3] = 255
    out[:, :, 3] = alpha
    return Image.fromarray(out)


def build_tree(name, src):
    a = cover_square(load_rgba(src)[:, :, 3], SIZE)
    r = radial(SIZE)
    # Smoothstep from fully opaque at VIG_INNER to gone at VIG_OUTER.
    t = np.clip((r - VIG_INNER) / (VIG_OUTER - VIG_INNER), 0.0, 1.0)
    fade = 1.0 - (t * t * (3.0 - 2.0 * t))
    out = (a.astype(np.float32) * fade).astype(np.uint8)
    white_with_alpha(out).save(os.path.join(OUT, name))
    print(f"  {name}: {int((out > 16).sum())} ink px")


def build_rim():
    """Pull the hand-drawn circle off the sketch photo: local-threshold the ink,
    then take the largest blob whose bbox is roughly square — that's the circle,
    not the writing or the squiggle."""
    g = np.array(Image.open("/tmp/_exit_sketch.png").convert("L")).astype(np.float32)
    ink = (ndimage.uniform_filter(g, size=SKETCH_BLUR) - g) > SKETCH_T
    lab, n = ndimage.label(ink, structure=np.ones((3, 3)))
    best, best_px = None, 0
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if min(h, w) == 0 or not (0.85 < w / h < 1.18):
            continue
        px = int((lab[sl] == i).sum())
        if px > best_px:
            best, best_px = (i, sl), px
    if best is None:
        raise SystemExit("no circular blob found in the sketch")
    i, sl = best
    ys, xs = sl
    print(f"  rim blob: {best_px} px, bbox x{xs.start}-{xs.stop} y{ys.start}-{ys.stop}")
    crop = (lab[sl] == i).astype(np.uint8) * 255

    # Fit the circle's bbox to a RIM_R-radius disc inside the square canvas.
    d = int(round(SIZE * RIM_R))
    im = Image.fromarray(crop).resize((d, d), Image.LANCZOS)
    canvas = np.zeros((SIZE, SIZE), np.uint8)
    off = (SIZE - d) // 2
    canvas[off:off + d, off:off + d] = np.array(im)
    white_with_alpha(canvas).save(os.path.join(OUT, "rim.png"))
    print(f"  rim.png: {int((canvas > 16).sum())} ink px")


def build_disc():
    """Solid circle just inside the rim — the exit screen's sphere is a white
    field on black, with the tree drawn dark on top of it. Antialiased at the
    edge so it doesn't stair-step at the size it's drawn."""
    r = radial(SIZE)
    edge = 2.0 / (SIZE / 2)          # ~2 px of feather
    a = np.clip((RIM_R - r) / edge, 0.0, 1.0)
    white_with_alpha((a * 255).astype(np.uint8)).save(os.path.join(OUT, "disc.png"))
    print(f"  disc.png: r={RIM_R:.2f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    load_rgba(SKETCH)   # also caches the converted PNG for build_rim
    print("disc:")
    build_disc()
    print("tree:")
    build_tree("tree_clean.png", f"{STUFF}/3808-clean-alpha.png")
    build_tree("tree_noise.png", f"{STUFF}/3808-noise-alpha.png")
    print("rim:")
    build_rim()
    print(f"\nwrote 4 x {SIZE}x{SIZE} textures to {OUT}/")


if __name__ == "__main__":
    main()
