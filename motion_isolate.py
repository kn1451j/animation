"""
motion_isolate.py  (v10)
------------------------
Tracks a small blob via SimpleBlobDetector on a per-frame cue map.
- Cue is optical-flow magnitude by default, or a whiteness score (--cue white)
  for handheld footage where the camera moves as much as the subject
- Dual-polarity detection (normal + inverted flow map; flow cue only)
- Anchor smoothed by median-of-last-N + EMA (kills per-frame detection jitter)
- Per-frame mask: fixed-radius radial vignette (soft edges), lightly warped by the
  thresholded flow silhouette, then EMA-blended across frames (aligned to the
  current anchor) so the mask shape stays roughly constant size and only waves
  around the edges from frame to frame
- On miss: anchor stays put, mask falls back to the vignette at the held anchor
- Boil: the smoothed mask's rim is warped by value noise, re-rolled every
  --boil-hold frames, so the cut-out reads as a redrawn edge instead of a clean
  disc. Only the mask is warped — the footage under it is never resampled,
  because the alpha is what selects from a frame that is valid everywhere.
- Outputs per-frame PNGs to {output_dir}/frames/ + tracked.mp4 (LOCKED frames only)

Usage:
    python motion_isolate.py ./vids/moving_objects/bug.mp4 ./vids/moving_objects/bug/
    python motion_isolate.py ... --dilate 80 --debug

Dependencies:
    pip install opencv-python-headless numpy tqdm scipy
"""

import cv2
import numpy as np
import argparse
from collections import namedtuple, deque
from pathlib import Path

TrajEntry = namedtuple(
    "TrajEntry",
    "idx cx cy is_locked mask_crop crop_x0 crop_y0",
)

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


def whiteness(frame, v_min: int, s_max: int) -> np.ndarray:
    """Where the frame is bright AND unsaturated, as a smooth 0-255 map.

    An alternative cue to optical flow, for footage where the camera moves as
    much as the subject does: a handheld walk over a trail puts more flow in the
    sliding ground than in the animal, so the flow map's brightest blob is the
    background. Colour has no such problem — a white butterfly against orange
    dirt is separable frame by frame, with no dependence on what moved.

    It is a hard threshold rather than a graded score on purpose. Sunlit dirt is
    bright and only moderately saturated, so a graded `V * (1 - S)` leaves a few
    hundred speckles per frame for the detector to pick from, and the nearest one
    to the anchor beats the real subject. Thresholding, opening away the
    speckles, and blurring what survives leaves the detector a map with peaks
    only where something is actually white.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > v_min) & (hsv[..., 1] < s_max)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(m, (21, 21), 0)


def best_blob_either_polarity(mag_norm, detector, anchor_xy, max_dist,
                              try_inverted=True):
    """Try normal and inverted cue map. Returns (keypoint|None, inverted:bool)."""
    kp_n = closest_blob(detector.detect(mag_norm),       anchor_xy, max_dist)
    # Only the flow cue is two-sided. On a whiteness map the inverse is "every
    # saturated dark thing", i.e. the entire forest floor.
    kp_i = (closest_blob(detector.detect(255 - mag_norm), anchor_xy, max_dist)
            if try_inverted else None)

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
# Per-frame detection mask (dilation of thresholded flow silhouette)
# ─────────────────────────────────────────────────────────────────────────────

def detection_mask(mag_norm: np.ndarray, inverted: bool, anchor: tuple,
                   kp_size: float, det_thresh: int, dilate: int,
                   h: int, w: int) -> np.ndarray:
    """Threshold the flow-magnitude map (matching the blob-detector polarity),
    keep only the connected component that contains the anchor, then
    morphologically dilate by `dilate` px with a circular kernel.

    Returns a full-frame uint8 mask (0/255). The dilation happens on the
    per-frame bug silhouette, so the mask shape follows the detection instead
    of always being a fixed-radius circle.
    """
    det_mag = (255 - mag_norm) if inverted else mag_norm
    det_binary = (det_mag > det_thresh).astype(np.uint8) * 255

    # Confine to a reasonable neighborhood of the anchor so we don't grab
    # unrelated moving regions (camera shake, other objects).
    confine_r = int(max(kp_size, 10.0) * 2 + dilate)
    confine = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(confine, (int(anchor[0]), int(anchor[1])), confine_r, 255, -1)
    det_binary = cv2.bitwise_and(det_binary, confine)

    # Extract only the connected component containing the anchor
    n, labels, stats, _ = cv2.connectedComponentsWithStats(det_binary, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), dtype=np.uint8)

    ax = int(np.clip(anchor[0], 0, w - 1))
    ay = int(np.clip(anchor[1], 0, h - 1))
    anchor_lbl = int(labels[ay, ax])
    if anchor_lbl == 0:
        # Anchor fell on background — use the closest non-zero pixel's component
        ys_nz, xs_nz = np.nonzero(det_binary)
        if xs_nz.size == 0:
            return np.zeros((h, w), dtype=np.uint8)
        d2 = (xs_nz - ax) ** 2 + (ys_nz - ay) ** 2
        j = int(np.argmin(d2))
        anchor_lbl = int(labels[ys_nz[j], xs_nz[j]])
        if anchor_lbl == 0:
            return np.zeros((h, w), dtype=np.uint8)

    component = np.where(labels == anchor_lbl, 255, 0).astype(np.uint8)

    k = 2 * dilate + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(component, kernel)


# ─────────────────────────────────────────────────────────────────────────────
# Edge boil
# ─────────────────────────────────────────────────────────────────────────────

def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def boil_field(h: int, w: int, cell: float, amp: float, seed: int,
               octaves: int, gain: float):
    """Two smooth random displacement maps (dx, dy) in pixels.

    A coarse random grid resized up is value noise by another name, and cv2 does
    the interpolation in C. `cell` is the grid pitch in pixels, so it sets the
    wavelength of the wobble; `amp` is its half-range.

    Octaves matter more than amplitude for how much the *shape* changes. One
    octave mostly slides the whole silhouette around — raising its amplitude
    gives a rounder blob in a different place, not a different outline. Halving
    the cell each octave adds detail that the boundary can actually bend around,
    so `gain` near 1.0 (rather than the usual fBm 0.5) is what buys variance.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        acc, norm, c, g = np.zeros((h, w), np.float32), 0.0, cell, 1.0
        for _o in range(max(1, octaves)):
            gh = max(2, int(round(h / c)) + 1)
            gw = max(2, int(round(w / c)) + 1)
            n  = (rng.random((gh, gw), dtype=np.float32) - 0.5) * 2.0
            acc += g * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
            norm += g
            c = max(4.0, c / 2.0)
            g *= gain
        out.append(acc / norm * amp)
    return out


def boil_mask(mask: np.ndarray, cell: float, amp: float, seed: int,
              octaves: int, gain: float) -> np.ndarray:
    """Warp the mask's soft rim by value noise, holding the middle still.

    The gate is the mask's own alpha — full displacement out at the feather and
    beyond it, none in the opaque core — so the silhouette waves without the
    interior sliding around. Ungated, the whole disc would swim.
    """
    if amp <= 0.0:
        return mask
    h, w = mask.shape
    dx, dy = boil_field(h, w, cell, amp, seed, octaves, gain)
    gate   = 1.0 - smoothstep(0.20, 0.80, mask.astype(np.float32) / 255.0)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(mask, xs + dx * gate, ys + dy * gate,
                     cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing post-process
# ─────────────────────────────────────────────────────────────────────────────

def smooth_pass(args, video_path: str, output_dir: Path,
                trajectory: list, fps: float, h: int, w: int) -> None:
    """Fit a cubic spline through a subset of trajectory frames and re-emit
    each frame with the masked bug content pasted at the smoothed position.

    Outputs:
        - {output_dir}/smooth_bug/frame_{idx+1:06d}.png   (RGBA, transparent BG)
        - {output_dir}/smooth_tracked.mp4                 (BGR over black)
    """
    from scipy.interpolate import CubicSpline

    N = len(trajectory)
    if N < 4:
        print("  ⚠  trajectory too short for cubic spline; skipping smoothing pass.")
        return

    T  = np.array([t.idx for t in trajectory], dtype=np.float64)
    xs = np.array([t.cx  for t in trajectory], dtype=np.float64)
    ys = np.array([t.cy  for t in trajectory], dtype=np.float64)

    K = max(4, min(args.smooth_knots, N))
    knot_idx = np.unique(np.linspace(0, N - 1, K, dtype=int))
    if len(knot_idx) < 4:
        print("  ⚠  not enough distinct knots; skipping smoothing pass.")
        return

    cs_x = CubicSpline(T[knot_idx], xs[knot_idx], bc_type="natural")
    cs_y = CubicSpline(T[knot_idx], ys[knot_idx], bc_type="natural")
    xs_s = cs_x(T)
    ys_s = cs_y(T)

    smooth_dir = output_dir / "smooth_bug"
    smooth_dir.mkdir(exist_ok=True)
    smooth_vid = output_dir / "smooth_tracked.mp4"
    vw = cv2.VideoWriter(str(smooth_vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"smooth_pass: cannot reopen {video_path}")

    cur = -1
    frame = None
    it = range(N)
    if HAS_TQDM:
        it = tqdm(it, unit="frame", desc="smoothing")

    for i in it:
        entry = trajectory[i]
        while cur < entry.idx:
            ret, frame = cap.read()
            cur += 1
            if not ret:
                cap.release(); vw.release()
                print("  ⚠  smooth_pass: video ended before trajectory did.")
                return

        # Reconstruct full-frame mask from the stored per-frame detection crop
        mask = np.zeros((h, w), dtype=np.uint8)
        if entry.mask_crop is not None:
            mh, mw = entry.mask_crop.shape
            y0, x0 = entry.crop_y0, entry.crop_x0
            mask[y0:y0+mh, x0:x0+mw] = entry.mask_crop
        else:
            cv2.circle(mask, (int(entry.cx), int(entry.cy)), args.dilate, 255, -1)

        # Boil the rim. Held for --boil-hold frames a beat: a fresh field every
        # frame is noise, not a redrawn line.
        mask = boil_mask(mask, args.boil_cell, args.boil_px,
                         seed=entry.idx // max(1, args.boil_hold),
                         octaves=args.boil_octaves, gain=args.boil_gain)

        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        bgra[..., 3] = mask

        dx = float(xs_s[i] - entry.cx)
        dy = float(ys_s[i] - entry.cy)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(
            bgra, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        cv2.imwrite(str(smooth_dir / f"frame_{entry.idx+1:06d}.png"), shifted)

        alpha   = shifted[..., 3:4].astype(np.float32) / 255.0
        bgr_out = (shifted[..., :3].astype(np.float32) * alpha).astype(np.uint8)
        if entry.is_locked:
            vw.write(bgr_out)

    cap.release()
    vw.release()
    print(f"✓ Smoothed frames in: {smooth_dir.resolve()}")
    print(f"   Smoothed video:    {smooth_vid.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

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
    anchor       = (frame_cx, frame_cy)
    ema_anchor   = (frame_cx, frame_cy)
    anchor_hist  = deque(maxlen=max(1, args.anchor_median))
    last_kp_size = 20.0
    locked       = False
    ever_locked  = False
    misses       = 0
    total_miss   = 0
    trajectory   = []  # TrajEntry per ever_locked frame, for smooth_pass

    # Soft-mask temporal blend state
    prev_blended        = None  # full-frame uint8 soft mask from previous frame
    prev_blended_anchor = None  # (x, y) associated with prev_blended

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
        if args.cue == "white":
            cue = whiteness(frame, args.white_v, args.white_s).astype(np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray,
                flow=None, pyr_scale=0.5, levels=3,
                winsize=13, iterations=3,
                poly_n=5, poly_sigma=1.1, flags=0,
            )
            cue, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Same temporal EMA either way — it steadies a flickering detection
        # without moving where the peak is.
        smooth_mag = args.smooth * cue + (1.0 - args.smooth) * smooth_mag
        mag_norm   = cv2.normalize(smooth_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if locked:
            search_r = args.max_jump
        elif ever_locked:
            search_r = relost_r
        else:
            search_r = init_r

        best_kp, inverted = best_blob_either_polarity(
            mag_norm, detector, anchor, search_r,
            try_inverted=(args.cue == "flow"))
        is_locked_frame   = False

        if best_kp is not None:
            raw_pt = (float(best_kp.pt[0]), float(best_kp.pt[1]))

            if not locked:
                # Fresh lock (initial or re-lock after lost): snap smoothers to raw
                anchor_hist.clear()
                anchor_hist.append(raw_pt)
                ema_anchor = raw_pt
                prev_blended = None
                prev_blended_anchor = None
            else:
                anchor_hist.append(raw_pt)
                med_x = float(np.median([p[0] for p in anchor_hist]))
                med_y = float(np.median([p[1] for p in anchor_hist]))
                a = args.anchor_ema
                ema_anchor = (
                    a * med_x + (1.0 - a) * ema_anchor[0],
                    a * med_y + (1.0 - a) * ema_anchor[1],
                )

            anchor          = ema_anchor
            last_kp_size    = float(best_kp.size)
            locked          = True
            ever_locked     = True
            misses          = 0
            is_locked_frame = True

            if args.debug:
                polarity = "INV" if inverted else "NRM"
                print(f"  frame {idx+1:04d} [LOCKED/{polarity}]"
                      f"  centre=({anchor[0]:.0f},{anchor[1]:.0f})"
                      f"  raw=({raw_pt[0]:.0f},{raw_pt[1]:.0f})")
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
            # Per-frame binary detection silhouette (or fallback disc if we missed)
            if best_kp is not None:
                det_binary = detection_mask(
                    mag_norm, inverted, anchor, last_kp_size,
                    args.det_thresh, args.dilate, h, w,
                )
                if cv2.countNonZero(det_binary) == 0:
                    det_binary = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(det_binary, (int(anchor[0]), int(anchor[1])), args.dilate, 255, -1)
            else:
                det_binary = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(det_binary, (int(anchor[0]), int(anchor[1])), args.dilate, 255, -1)

            feather = max(int(args.mask_feather), 1)
            kdim    = 2 * feather + 1
            det_soft = cv2.GaussianBlur(det_binary, (kdim, kdim), 0).astype(np.float32) / 255.0

            # Fixed-radius radial vignette at the anchor: soft disc, stable size
            disc = np.zeros((h, w), dtype=np.float32)
            cv2.circle(disc, (int(anchor[0]), int(anchor[1])),
                       max(args.dilate - feather, 1), 1.0, -1)
            vignette = cv2.GaussianBlur(disc, (kdim, kdim), 0)

            # Let the detection only lightly warp the vignette's edge (size stays put)
            warp_s   = float(args.mask_warp)
            mask_raw = vignette * (1.0 - warp_s * (1.0 - det_soft))
            mask_raw = np.clip(mask_raw, 0.0, 1.0)
            mask_now = (mask_raw * 255.0).astype(np.uint8)

            # Temporal EMA across frames, aligned to the current anchor so motion
            # doesn't smear the shape (purely shape-blending, not position-blending)
            if prev_blended is not None and prev_blended_anchor is not None:
                dx = float(anchor[0] - prev_blended_anchor[0])
                dy = float(anchor[1] - prev_blended_anchor[1])
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                prev_aligned = cv2.warpAffine(
                    prev_blended, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
                a_blend = float(args.mask_blend)
                mask = cv2.addWeighted(mask_now, a_blend, prev_aligned, 1.0 - a_blend, 0.0)
            else:
                mask = mask_now.copy()

            prev_blended        = mask
            prev_blended_anchor = (float(anchor[0]), float(anchor[1]))

            # Apply the soft mask as a real alpha multiply (not a hard np.where)
            alpha_f   = (mask.astype(np.float32) / 255.0)[..., None]
            out_frame = (frame.astype(np.float32) * alpha_f).astype(np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{idx+1:06d}.png"), out_frame)
            if is_locked_frame:
                vwriter.write(out_frame)

            bx, by, bw_, bh_ = cv2.boundingRect(mask)
            if bw_ > 0 and bh_ > 0:
                mask_crop = mask[by:by+bh_, bx:bx+bw_].copy()
                crop_x0, crop_y0 = int(bx), int(by)
            else:
                mask_crop = None
                crop_x0 = crop_y0 = 0
            trajectory.append(TrajEntry(
                idx, float(anchor[0]), float(anchor[1]), is_locked_frame,
                mask_crop, crop_x0, crop_y0,
            ))

        prev_gray = gray

    cap.release()
    vwriter.release()
    if total_miss:
        print(f"  ⚠  {total_miss} frames had no qualifying blob.")
    print(f"\n✓ Done — frames in: {frames_dir.resolve()}")
    print(f"   Video (LOCKED only): {vid_path.resolve()}")

    if not args.skip_smooth and trajectory:
        smooth_pass(args, args.video, output_dir, trajectory, fps, h, w)
    elif not trajectory:
        print("  ⚠  no trajectory recorded; skipping smoothing pass.")


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
    p.add_argument("--cue",              choices=("flow", "white"), default="flow",
                   help="What the blob detector looks at. 'flow' is optical-flow "
                        "magnitude — right when the camera is steadier than the "
                        "subject. 'white' scores bright, desaturated pixels, for "
                        "handheld footage where camera motion swamps the flow map.")
    p.add_argument("--white-v",          type=int,   default=175,
                   help="--cue white: minimum HSV value counted as bright.")
    p.add_argument("--white-s",          type=int,   default=55,
                   help="--cue white: maximum HSV saturation counted as white.")
    p.add_argument("--min-area",         type=float, default=4.0)
    p.add_argument("--max-area",         type=float, default=50.0)
    p.add_argument("--max-jump",         type=int,   default=59)
    p.add_argument("--init-radius-frac", type=float, default=0.25)
    p.add_argument("--miss-limit",       type=int,   default=12)
    p.add_argument("--dilate",           type=int,   default=130,
                   help="Morphological dilation radius (px) applied to the per-frame detection silhouette.")
    p.add_argument("--det-thresh",       type=int,   default=150,
                   help="Flow-magnitude threshold (0-255) used to extract the bug silhouette before dilation.")
    p.add_argument("--mask-feather",     type=int,   default=25,
                   help="Gaussian feather radius (px) for the vignette + detection softening.")
    p.add_argument("--mask-warp",        type=float, default=0.45,
                   help="Fraction of the vignette's edge the per-frame detection may warp (0-1).")
    p.add_argument("--mask-blend",       type=float, default=0.5,
                   help="EMA weight for the current frame's mask (lower = more shape inertia from prior frames).")
    p.add_argument("--anchor-ema",       type=float, default=0.3,
                   help="EMA weight for new anchor samples after median filtering (lower = smoother, more lag).")
    p.add_argument("--anchor-median",    type=int,   default=5,
                   help="Median-filter window size over recent raw blob detections.")
    p.add_argument("--boil-px",          type=float, default=20.0,
                   help="Edge-boil amplitude (source px). 0 disables the boil.")
    p.add_argument("--boil-cell",        type=float, default=32.0,
                   help="Edge-boil noise wavelength (source px) of the coarsest "
                        "octave. Wider lobes the disc into a blob; tighter just "
                        "reads as fuzz.")
    p.add_argument("--boil-octaves",     type=int,   default=2,
                   help="Noise octaves, each half the previous cell. More than "
                        "one is what makes the outline vary rather than slide.")
    p.add_argument("--boil-gain",        type=float, default=0.75,
                   help="Amplitude ratio between successive octaves. Near 1.0 "
                        "keeps the fine detail that bends the boundary.")
    p.add_argument("--boil-hold",        type=int,   default=2,
                   help="Frames each boil field is held for — 2 is 'on twos'.")
    p.add_argument("--smooth",           type=float, default=0.65)
    p.add_argument("--smooth-knots",     type=int,   default=15,
                   help="Number of cubic-spline knots sampled uniformly across the trajectory.")
    p.add_argument("--skip-smooth",      action="store_true",
                   help="Disable the trajectory-smoothing post-processing pass.")
    p.add_argument("--debug",            action="store_true")
    args = p.parse_args()
    process(args)

if __name__ == "__main__":
    main()