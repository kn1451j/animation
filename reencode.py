"""
Re-encode the PNG sprite sequences as video.

Startup used to decode ~2,286 PNGs through stb_image, which costs ~36 ms a
frame even at -O2 and dominated the wait before the title screen appeared. The
same frame out of a video stream costs ~1 ms. Nothing is deferred: main.cpp's
SpriteSeq::loadVideo() still uploads every frame to the GPU before init()
returns, so this is the same eager load with a faster front end.

Codec per sequence, and why:

  ants_smooth  opaque everywhere, so it ships with no alpha channel at all, and
               it is the one sequence here that is real footage rather than
               drawing. Lossy h264 is right for it — a lossless libx264rgb
               encode was bit-exact but took 126 ms a frame to decode, three
               times slower than the PNGs it was meant to replace. Lossless
               entropy coding on photographic noise is the worst case for the
               decoder, and there is nothing to protect: this plays as a dim
               texture under the terminal text.

  exit         flat colour behind a shaped alpha. qtrle (QuickTime Animation)
               is RLE, so it is bit-exact — the same pixels the PNGs held.

  smooth_bug   97.6% fully transparent, but the source PNGs still carry the
               original footage in RGB underneath that transparency: 66,794
               distinct colours under alpha=0. RLE cannot compress noise, which
               is why a straight qtrle encode came out at 748 MB — larger than
               the PNGs. Zeroing RGB wherever alpha is 0 collapses it. Those
               pixels are invisible either way; the blend never reads them.

  waves        dense, coloured, and soft: 59.9% of pixels are partially
               transparent and only 129 distinct values hide under the
               transparent ones, so the zeroing trick buys nothing here. It
               stays a plain qtrle — big on disk, but bit-exact and ~20x faster
               to decode than the PNGs, which is the thing being bought.

The walk/turn/pose sprites take the same lossless qtrle as exit, for the same
reason — they are ~99.3% empty, which is the best case RLE has.

Usage:  python3 reencode.py [name ...]      (from the repo root; no args = all)
Needs:  ffmpeg + ffprobe on PATH, pillow, numpy.
"""

import os
import subprocess
import sys
import glob

OUT = "renders/seq"

QTRLE = ["-c:v", "qtrle", "-pix_fmt", "rgba"]

# name -> (source glob, output file, extra ffmpeg args, zero_rgb_under_alpha)
SEQS = {
    "ants_smooth": ("renders/ants_smooth/*.png", "ants_smooth.mp4",
                    ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"], False),
    "exit":        ("exit/*.png", "exit.mov", QTRLE, False),
    "smooth_bug":  ("vids/moving_objects/bug/smooth_bug/*.png", "smooth_bug.mov",
                    QTRLE, True),
    "waves":       ("renders/background_waves_staggered/*.png", "waves.mov",
                    QTRLE, False),
}

# The walk/turn/pose sprites: ten directories, 692 frames, all of them
# 1920x1080 and ~99.3% fully transparent with clean RGB underneath. That is the
# best case RLE has, so they all take the same lossless qtrle treatment.
# back_pose is a single frame and stays a PNG — not worth a container.
for _d in ["walk_right/shake_head", "walk_left/shake_head",
           "walk_right/left_foot",  "walk_right/right_foot",
           "walk_left/left_foot",   "walk_left/right_foot",
           "walk_right/turn_right", "walk_left/turn_left",
           "center_sprite",         "enter_right"]:
    SEQS[_d] = (f"renders/{_d}/*.png", _d.replace("/", "_") + ".mov", QTRLE, False)


def frames(pattern):
    """Sorted, AppleDouble-free — the same order SpriteSeq::load walks."""
    return sorted(p for p in glob.glob(pattern)
                  if not os.path.basename(p).startswith("._"))


def encode(paths, args, dst, zero):
    """
    Decode the PNGs here and pipe raw RGBA into ffmpeg.

    Letting ffmpeg glob the directory itself is faster but hands frame ordering
    and frame *count* to a demuxer. The concat demuxer in particular emits its
    own timestamps, and ffmpeg's default frame-rate conversion then drops frames
    to hit the output rate — an early version of this script turned 123 ants
    frames into 104 and said nothing about it. Feeding rawvideo makes the count
    exact by construction: one array written, one frame encoded.

    zero=True clears RGB wherever alpha is 0. Those pixels are never blended in,
    but the noise they carry is what stops RLE from compressing.
    """
    import numpy as np
    from PIL import Image

    w, h = Image.open(paths[0]).size
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
         "-framerate", "30", "-i", "-", *args, "-an", dst],
        stdin=subprocess.PIPE)
    for p in paths:
        a = np.array(Image.open(p).convert("RGBA"))
        if zero:
            a[:, :, :3][a[:, :, 3] == 0] = 0
        proc.stdin.write(a.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed on {dst}")


def check_frames(dst, expected):
    """A sequence that silently loses frames plays back wrong, so refuse it."""
    got = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", dst],
        capture_output=True, text=True).stdout.strip()
    if got != str(expected):
        raise SystemExit(f"{dst}: encoded {got} frames, expected {expected}")


def main():
    os.makedirs(OUT, exist_ok=True)
    wanted = sys.argv[1:] or list(SEQS)
    for name in wanted:
        if name not in SEQS:
            raise SystemExit(f"unknown sequence {name!r}; know {list(SEQS)}")
        pattern, out, args, zero = SEQS[name]
        paths = frames(pattern)
        if not paths:
            print(f"  {name}: no PNGs at {pattern} — skipped")
            continue
        dst = os.path.join(OUT, out)
        encode(paths, args, dst, zero)
        check_frames(dst, len(paths))
        src_mb = sum(os.path.getsize(p) for p in paths) / 1e6
        print(f"  {name}: {len(paths)} frames, "
              f"{src_mb:.0f} MB PNG -> {os.path.getsize(dst)/1e6:.0f} MB {out}")


if __name__ == "__main__":
    main()
