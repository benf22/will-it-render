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
    ap.add_argument("--masks",     required=True, help=".npz from gsam_segment.py")
    ap.add_argument("--color",     default="#00dc3c", help="Contour color (hex or R,G,B)")
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--halo",      type=int, default=5,
                    help="White halo thickness around contour (0 = off)")
    ap.add_argument("--fill",      type=float, default=0.0,
                    help="Mask fill alpha 0–1 (0 = off)")
    ap.add_argument("--out",       default=None, help="Output mp4 path")
    args = ap.parse_args()

    npz_path  = Path(args.masks)
    json_path = npz_path.with_suffix(".json")
    meta      = json.loads(json_path.read_text())

    out_path = Path(args.out) if args.out else npz_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data   = np.load(npz_path)
    masks  = data["masks"]   # [N, H, W] uint8 0/255
    scores = data["scores"]  # [N] float

    print(f"loading source video: {meta['video']}")
    frames, fps = load_frames(Path(meta["video"]), meta["half"])
    assert len(frames) == len(masks), "frame/mask count mismatch"

    color_bgr   = parse_color(args.color)
    halo_bgr    = (255, 255, 255)
    H, W        = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    for i, (frame, mask) in enumerate(zip(frames, masks)):
        canvas = frame.copy()

        if scores[i] > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            # optional fill
            if args.fill > 0:
                overlay = canvas.copy()
                cv2.drawContours(overlay, contours, -1, color_bgr, cv2.FILLED)
                cv2.addWeighted(overlay, args.fill, canvas, 1 - args.fill, 0, canvas)
            # halo then contour
            if args.halo > 0:
                cv2.drawContours(canvas, contours, -1, halo_bgr, args.halo)
            cv2.drawContours(canvas, contours, -1, color_bgr, args.thickness)

        writer.write(canvas)

    writer.release()

    # re-mux with ffmpeg for browser-compatible h264
    import subprocess, os
    tmp = str(out_path) + "_tmp.mp4"
    os.rename(out_path, tmp)
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp,
        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path)
    ], check=True, capture_output=True)
    os.remove(tmp)

    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e3:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
