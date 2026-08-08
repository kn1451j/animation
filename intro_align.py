#!/usr/bin/env python3
"""Align the intro's video cuts to its audio, leaving the audio untouched.

`vids/splice/new_intro.mp4` is a single baked file: the voiceover is muxed in and
the video cuts were hand-placed to land on the audio's segment boundaries — but a
few drifted (0.1–0.4 s off). This re-times ONLY the video so each cut that belongs
on an audio boundary snaps exactly onto it, while intentional mid-segment (B-roll)
cuts are left alone. The audio stream is copied through unchanged, so global sync
is preserved and every snapped cut lands on its swap.

Method (all frame-exact, so nothing drifts across the 15 shots):
  1. Detect video cuts (ffmpeg scene detection) and audio boundaries (silencedetect
     — where speech/section starts or stops).
  2. Snap each internal cut to the nearest audio boundary within SNAP_TOL; cuts with
     no boundary in range keep their time. Endpoints (0, end) are fixed, so the
     total duration — and thus audio sync — is unchanged.
  3. Re-time each shot with trim + hold (freeze last frame): trim if its target
     window is shorter, clone-pad the last frame if longer. Concat, then mux with
     the ORIGINAL audio.

The pristine original is preserved as new_intro_orig.mp4 and used as the source on
every run, so re-running is idempotent (never re-times an already-aligned file).

Usage:  python3 intro_align.py            # detect, snap, retime, replace mp4
        python3 intro_align.py --dry-run  # print the cut->boundary mapping only
"""
import os
import re
import subprocess
import sys
from fractions import Fraction

SPLICE      = "vids/splice"
LIVE        = os.path.join(SPLICE, "new_intro.mp4")     # what main.cpp loads
ORIGINAL    = os.path.join(SPLICE, "new_intro_orig.mp4")  # pristine source (backup)
OUT_TMP     = os.path.join(SPLICE, "new_intro_aligned.mp4")

SCENE_THRESH = 0.4    # ffmpeg scene score for a hard cut
CUT_DEDUP    = 0.40   # merge cuts closer than this (same visual switch)
SILENCE_DB   = -32    # silencedetect noise floor
SILENCE_MIN  = 0.60   # ignore pauses shorter than this — keep only real swaps
SNAP_TOL     = 0.50   # a cut this close to a boundary is meant to land on it


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe(path):
    """Return (fps as Fraction, duration seconds, total frame count)."""
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,nb_frames,duration",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1", path])
    fps, nb, vdur, fdur = None, None, None, None
    for line in r.stdout.splitlines():
        k, _, v = line.partition("=")
        if k == "r_frame_rate" and "/" in v:
            fps = Fraction(v)
        elif k == "nb_frames" and v.isdigit():
            nb = int(v)
        elif k == "duration" and v not in ("", "N/A"):
            # first duration= is the stream's, second the format's; keep the last seen
            fdur = float(v)
    dur = fdur
    if nb is None and dur is not None and fps is not None:
        nb = round(dur * float(fps))
    return fps, dur, nb


def detect_cuts(path):
    """Timestamps (s) of hard video cuts, deduped."""
    r = run(["ffmpeg", "-i", path, "-filter_complex",
             f"select='gt(scene,{SCENE_THRESH})',metadata=print:file=-",
             "-an", "-f", "null", "-"])
    times = []
    for m in re.finditer(r"pts_time:([0-9.]+)", r.stderr + r.stdout):
        t = float(m.group(1))
        if not times or t - times[-1] > CUT_DEDUP:
            times.append(t)
    return times


def detect_audio_boundaries(path):
    """Timestamps (s) where the audio content changes — every silence edge
    (a section/speech start or stop) longer than SILENCE_MIN."""
    r = run(["ffmpeg", "-i", path, "-af",
             f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN}",
             "-f", "null", "-"])
    b = set()
    for m in re.finditer(r"silence_(start|end):\s*(-?[0-9.]+)", r.stderr + r.stdout):
        t = float(m.group(2))
        if t > 0.05:
            b.add(round(t, 3))
    return sorted(b)


def snap(cuts, bounds, duration):
    """Snap each cut to the nearest audio boundary within SNAP_TOL.
    Returns list of (orig, snapped, boundary_or_None, delta)."""
    out = []
    for c in cuts:
        if 0 < c < duration:
            near = min(bounds, key=lambda b: abs(b - c)) if bounds else None
            if near is not None and abs(near - c) <= SNAP_TOL:
                out.append((c, near, near, near - c))
                continue
        out.append((c, c, None, 0.0))
    return out


def main():
    dry = "--dry-run" in sys.argv
    if not os.path.exists(LIVE) and not os.path.exists(ORIGINAL):
        sys.exit(f"[intro] {LIVE} not found")

    # Always retime from the pristine original so re-runs are idempotent.
    if not os.path.exists(ORIGINAL):
        if not dry:
            import shutil
            shutil.copy2(LIVE, ORIGINAL)
            print(f"[intro] backed up original → {ORIGINAL}")
        src = LIVE
    else:
        src = ORIGINAL

    fps, dur, total = probe(src)
    if not fps or not dur or not total:
        sys.exit(f"[intro] could not probe {src} (fps={fps} dur={dur} n={total})")
    print(f"[intro] source={src}  fps={fps} ({float(fps):.3f})  "
          f"dur={dur:.3f}s  frames={total}")

    cuts   = detect_cuts(src)
    bounds = detect_audio_boundaries(src)
    print(f"[intro] {len(cuts)} video cuts, {len(bounds)} audio boundaries")

    mapping = snap(cuts, bounds, dur)
    print("\n  video cut   -> snapped     (audio boundary,  delta)")
    for orig, snapped, b, d in mapping:
        if b is None:
            print(f"   {orig:8.3f}   keep         (no boundary within {SNAP_TOL}s)")
        else:
            tag = "  (already aligned)" if abs(d) < 0.02 else ""
            print(f"   {orig:8.3f}  -> {snapped:8.3f}    ({b:8.3f}, {d:+.3f}){tag}")

    # Boundary list in frames: [0] + snapped cuts + [total]. Frame-exact so the
    # per-shot counts sum to `total` and nothing drifts.
    def f_of(t):
        return max(0, min(total, round(t * float(fps))))

    src_edges = [0] + [f_of(o)       for o, _, _, _ in mapping] + [total]
    tgt_edges = [0] + [f_of(s)       for _, s, _, _ in mapping] + [total]

    # Build the frame-exact trim + hold filtergraph.
    parts, labels = [], []
    for i in range(len(src_edges) - 1):
        s0, s1 = src_edges[i], src_edges[i + 1]
        t0, t1 = tgt_edges[i], tgt_edges[i + 1]
        src_n, tgt_n = s1 - s0, t1 - t0
        if src_n <= 0 or tgt_n <= 0:
            continue
        keep = min(src_n, tgt_n)
        hold = tgt_n - keep            # frames to freeze the last frame for
        lab = f"v{i}"
        f = (f"[0:v]trim=start_frame={s0}:end_frame={s0 + keep},"
             f"setpts=PTS-STARTPTS,"
             + (f"tpad=stop={hold}:stop_mode=clone," if hold > 0 else "")
             + f"fps={fps.numerator}/{fps.denominator},settb=AVTB[{lab}]")
        parts.append(f)
        labels.append(f"[{lab}]")

    filtergraph = ";".join(parts) + ";" + "".join(labels) + \
        f"concat=n={len(labels)}:v=1:a=0[outv]"

    if dry:
        print("\n[intro] --dry-run: no file written.")
        return

    cmd = ["ffmpeg", "-y", "-i", src,
           "-filter_complex", filtergraph,
           "-map", "[outv]", "-map", "0:a:0",
           "-r", f"{fps.numerator}/{fps.denominator}",
           "-c:v", "libx264", "-crf", "14", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-c:a", "copy", OUT_TMP]
    print("\n[intro] retiming video (audio copied through)…")
    r = run(cmd)
    if r.returncode != 0 or not os.path.exists(OUT_TMP):
        sys.stderr.write(r.stderr[-3000:])
        sys.exit("[intro] ffmpeg failed")

    os.replace(OUT_TMP, LIVE)
    print(f"[intro] wrote {LIVE}  (original preserved at {ORIGINAL})")


if __name__ == "__main__":
    main()
