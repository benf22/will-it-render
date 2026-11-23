#!/usr/bin/env python3
"""Render a video with Grounded-SAM contour overlay from pre-computed masks.

Loads the .npz produced by gsam_segment.py — no model inference needed.

Usage:
    python tools/gsam_render.py \\
        --masks  static/videos/contours/fig1_wan281_right_kite.npz \\
        --color  "#00dc3c" \\        # contour colour (hex or R,G,B)
        --thickness 3 \\
        --halo 6 \\                  # white halo width (0 = off)
        --out    static/videos/contours/fig1_wan281_right_kite.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_color(s: str) -> tuple[int, int, int]:
    s = s.strip()
    if s.startswith("#"):
        s = s.lstrip("#")
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    else:
        parts = [int(x) for x in re.split(r"[,\s]+", s)]
        r, g, b = parts[0], parts[1], parts[2]
    return (b, g, r)   # OpenCV is BGR


def load_frames(video_path: Path, half: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if half == "right":
            frame = frame[:, frame.shape[1] // 2 :]
        elif half == "left":
            frame = frame[:, : frame.shape[1] // 2]
        frames.append(frame)
    cap.release()
    return frames, fps


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks",     required=True, nargs="+",
                    help=".npz file(s) from gsam_segment.py (one or more)")
    ap.add_argument("--color",     default="#00dc3c", help="Contour color (hex or R,G,B)")
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--halo",      type=int, default=5,
                    help="White halo thickness around contour (0 = off)")
    ap.add_argument("--fill",      type=float, default=0.0,
                    help="Mask fill alpha 0–1 (0 = off)")
    ap.add_argument("--out",       default=None, help="Output mp4 path")
    args = ap.parse_args()

    # Load all mask files; derive video source from the first one
    npz_paths = [Path(p) for p in args.masks]
    meta      = json.loads(npz_paths[0].with_suffix(".json").read_text())

    out_path = Path(args.out) if args.out else npz_paths[0].with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_mask_data = []
    for p in npz_paths:
        d = np.load(p)
        all_mask_data.append((d["masks"], d["scores"]))  # each [N, H, W], [N]

    print(f"loading source video: {meta['video']}")
    frames, fps = load_frames(Path(meta["video"]), meta["half"])
    N = len(frames)
    for masks, scores in all_mask_data:
        assert len(masks) == N, f"frame/mask count mismatch in {p}"

    color_bgr   = parse_color(args.color)
    halo_bgr    = (255, 255, 255)
    H, W        = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    for i, frame in enumerate(frames):
        canvas = frame.copy()

        for masks, scores in all_mask_data:
            if scores[i] > 0:
                contours, _ = cv2.findContours(masks[i], cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if args.fill > 0:
                    overlay = canvas.copy()
                    cv2.drawContours(overlay, contours, -1, color_bgr, cv2.FILLED)
                    cv2.addWeighted(overlay, args.fill, canvas, 1 - args.fill, 0, canvas)
                if args.halo > 0:
                    cv2.drawContours(canvas, contours, -1, halo_bgr, args.halo)
                cv2.drawContours(canvas, contours, -1, color_bgr, args.thickness)

        writer.write(canvas)

    writer.release()

    # re-mux with ffmpeg for browser-compatible h264
    import subprocess, os
    tmp = str(out_path) + "_tmp.mp4"
    os.rename(out_path, tmp)
    ffmpeg_bin = shutil.which("ffmpeg") or "/home/lab/fishmab1/miniforge3/bin/ffmpeg"
    subprocess.run([
        ffmpeg_bin, "-y", "-i", tmp,
        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path)
    ], check=True, capture_output=True)
    os.remove(tmp)

    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e3:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
