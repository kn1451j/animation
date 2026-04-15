"""
motion_isolate.py  (v9)
-----------------------
Tracks a small moving blob via SimpleBlobDetector on optical flow magnitude.
- Dual-polarity detection (normal + inverted flow map)
- Direction-clamped anchor: velocity can decelerate to zero but never reverse
- Fixed-size circular mask (--dilate radius)
- On miss: anchor stays put
- Outputs per-frame PNGs + tracked.mp4 (LOCKED frames only)

Usage:
    python motion_isolate.py ./vids/moving_objects/bug.mp4 ./vids/moving_objects/bug/
    python motion_isolate.py ... --dilate 80 --debug

Dependencies:
    pip install opencv-python-headless numpy tqdm
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ─────────────────────────────────────────────────────────────────────────────
# Blob detector
# ─────────────────────────────────────────────────────────────────────────────

def make_blob_detector(min_area: float, max_area: float) -> cv2.SimpleBlobDetector:
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold        = 150
    p.maxThreshold        = 255
    p.thresholdStep       = 10
    p.minRepeatability    = 1
    p.filterByColor       = True
    p.blobColor           = 255
    p.filterByArea        = True
    p.minArea             = float(min_area)
    p.maxArea             = float(max_area)
    p.filterByCircularity = False
    p.filterByConvexity   = False
    p.filterByInertia     = False
    return cv2.SimpleBlobDetector_create(p)


# ─────────────────────────────────────────────────────────────────────────────
# Blob helpers
# ─────────────────────────────────────────────────────────────────────────────

def closest_blob(keypoints, anchor_xy, max_dist):
    ax, ay = anchor_xy
    best_kp, best_d = None, float("inf")
    for kp in keypoints:
        d = ((kp.pt[0] - ax) ** 2 + (kp.pt[1] - ay) ** 2) ** 0.5
        if d < max_dist and d < best_d:
            best_d, best_kp = d, kp
    return best_kp


def best_blob_either_polarity(mag_norm, detector, anchor_xy, max_dist):
    """Try normal and inverted flow map. Returns (keypoint|None, inverted:bool)."""
    kp_n = closest_blob(detector.detect(mag_norm),       anchor_xy, max_dist)
    kp_i = closest_blob(detector.detect(255 - mag_norm), anchor_xy, max_dist)

    if kp_n is None and kp_i is None:
        return None, False
    if kp_n is None:
        return kp_i, True
    if kp_i is None:
        return kp_n, False

    dn = ((kp_n.pt[0] - anchor_xy[0])**2 + (kp_n.pt[1] - anchor_xy[1])**2) ** 0.5
    di = ((kp_i.pt[0] - anchor_xy[0])**2 + (kp_i.pt[1] - anchor_xy[1])**2) ** 0.5
    if abs(dn - di) < 10:
        return (kp_i, True) if kp_i.size > kp_n.size else (kp_n, False)
    return (kp_i, True) if di < dn else (kp_n, False)


def clamp_direction(proposed, current, last_delta):
    """
    Move anchor toward `proposed` but don't allow direction reversal.
    Per axis: if the proposed move opposes last_delta, clamp that axis to zero.
    """
    dx = proposed[0] - current[0]
    dy = proposed[1] - current[1]

    # If last delta had a direction, new delta must not oppose it
    if last_delta[0] != 0:
        if (dx * last_delta[0]) < 0:   # opposite sign → clamp
            dx = 0.0
    if last_delta[1] != 0:
        if (dy * last_delta[1]) < 0:
            dy = 0.0

    return (current[0] + dx, current[1] + dy), (dx, dy)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "_debug"
    if args.debug:
        debug_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {args.video}")

    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps       = cap.get(cv2.CAP_PROP_FPS) or 24.0
    ret, prev = cap.read()
    if not ret:
        raise IOError("Cannot read first frame.")

    h, w      = prev.shape[:2]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    smooth_mag = np.zeros((h, w), dtype=np.float32)
    detector   = make_blob_detector(args.min_area, args.max_area)

    vid_path = output_dir / "tracked.mp4"
    vwriter  = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Tracker state
    frame_cx, frame_cy = w / 2.0, h / 2.0
    anchor      = (frame_cx, frame_cy)
    locked      = False
    ever_locked = False
    misses      = 0
    total_miss  = 0

    init_r   = min(w, h) * args.init_radius_frac
    relost_r = args.max_jump * 1.5

    print(f"Processing {total} frames | "
          f"blob area=[{args.min_area},{args.max_area}]px | "
          f"max_jump={args.max_jump}px")

    it = range(1, total)
    if HAS_TQDM:
        it = tqdm(it, unit="frame")

    for idx in it:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            flow=None, pyr_scale=0.5, levels=3,
            winsize=13, iterations=3,
            poly_n=5, poly_sigma=1.1, flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        smooth_mag = args.smooth * mag + (1.0 - args.smooth) * smooth_mag
        mag_norm   = cv2.normalize(smooth_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if locked:
            search_r = args.max_jump
        elif ever_locked:
            search_r = relost_r
        else:
            search_r = init_r

        best_kp, inverted = best_blob_either_polarity(mag_norm, detector, anchor, search_r)
        is_locked_frame   = False

        if best_kp is not None:
            anchor      = best_kp.pt
            locked      = True
            ever_locked = True
            misses      = 0
            is_locked_frame = True

            if args.debug:
                polarity = "INV" if inverted else "NRM"
                print(f"  frame {idx+1:04d} [LOCKED/{polarity}]"
                      f"  centre=({anchor[0]:.0f},{anchor[1]:.0f})")
        else:
            total_miss += 1
            if locked:
                # Stay put — no velocity extrapolation
                misses += 1
                if args.debug:
                    print(f"  frame {idx+1:04d} [MISS={misses}] anchor held at "
                          f"({anchor[0]:.0f},{anchor[1]:.0f})")
                if misses >= args.miss_limit:
                    locked = False
                    misses = 0
                    if args.debug:
                        print(f"  frame {idx+1:04d}  LOST — searching from {anchor}")

        # Debug heatmap
        if args.debug:
            display_map = (255 - mag_norm) if (best_kp is not None and inverted) else mag_norm
            vis = cv2.applyColorMap(display_map, cv2.COLORMAP_TURBO)
            cv2.circle(vis, (int(anchor[0]), int(anchor[1])), int(search_r), (255, 255, 255), 1)
            if best_kp is not None:
                cv2.circle(vis, (int(anchor[0]), int(anchor[1])), 5, (0, 0, 255), -1)
            cv2.imwrite(str(debug_dir / f"_dbg_{idx+1:04d}.png"), vis)

        # Write output whenever we have a lock or are holding position
        if ever_locked:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (int(anchor[0]), int(anchor[1])), args.dilate, 255, -1)
            mask_3ch  = np.stack([mask] * 3, axis=2)
            out_frame = np.where(mask_3ch, frame, 0).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"frame_{idx+1:06d}.png"), out_frame)
            if is_locked_frame:
                vwriter.write(out_frame)

        prev_gray = gray

    cap.release()
    vwriter.release()
    if total_miss:
        print(f"  ⚠  {total_miss} frames had no qualifying blob.")
    print(f"\n✓ Done — frames in: {output_dir.resolve()}")
    print(f"   Video (LOCKED only): {vid_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Track a small high-contrast moving blob.",
    )
    p.add_argument("video")
    p.add_argument("output_dir")
    p.add_argument("--min-area",         type=float, default=4.0)
    p.add_argument("--max-area",         type=float, default=50.0)
    p.add_argument("--max-jump",         type=int,   default=59)
    p.add_argument("--init-radius-frac", type=float, default=0.25)
    p.add_argument("--miss-limit",       type=int,   default=12)
    p.add_argument("--dilate",           type=int,   default=80)
    p.add_argument("--smooth",           type=float, default=0.65)
    p.add_argument("--debug",            action="store_true")
    args = p.parse_args()
    process(args)

if __name__ == "__main__":
    main()