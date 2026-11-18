#!/usr/bin/env python3
"""Run Grounded-SAM (GDINO + SAM) on a video and save per-frame masks.

Saves a .npz (masks, boxes, scores) + sidecar .json so gsam_render.py can
re-render the video with any visual config without re-running inference.

Usage:
    python tools/gsam_segment.py \\
        --video  static/videos/carousel/fig1_wan281.mp4 \\
        --noun   kite \\
        --half   right \\          # left | right | full
        --out    static/videos/contours/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_frames(video_path: Path, half: str) -> tuple[list[np.ndarray], float]:
    """Return (BGR frames, fps). Optionally crop to left/right half."""
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


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--noun", required=True, help="Text prompt for GDINO, e.g. 'kite'")
    ap.add_argument("--half", default="full", choices=["left", "right", "full"])
    ap.add_argument("--out", default="static/videos/contours")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{video_path.stem}_{args.half}_{args.noun.replace(' ', '_')}"
    out_npz  = out_dir / f"{slug}.npz"
    out_json = out_dir / f"{slug}.json"

    print(f"loading video: {video_path}")
    frames_bgr, fps = load_frames(video_path, args.half)
    H, W = frames_bgr[0].shape[:2]
    print(f"  {len(frames_bgr)} frames  {W}×{H}  {fps:.1f} fps")

    # ── GDINO ────────────────────────────────────────────────────────────────
    print("loading Grounding DINO …")
    gdino_proc  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).to(args.device)
    gdino_model.eval()

    # ── SAM ──────────────────────────────────────────────────────────────────
    print("loading SAM …")
    sam_proc  = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(args.device)
    sam_model.eval()

    text_prompt = f"{args.noun.lower()}."

    all_masks  = np.zeros((len(frames_bgr), H, W), dtype=np.uint8)
    all_boxes  = np.zeros((len(frames_bgr), 4), dtype=np.float32)
    all_scores = np.zeros(len(frames_bgr),       dtype=np.float32)

    print("running GDINO + SAM per frame …")
    for i, frame_bgr in enumerate(frames_bgr):
        pil = bgr_to_pil(frame_bgr)

        # GDINO
        with torch.no_grad():
            inputs = gdino_proc(images=pil, text=text_prompt,
                                return_tensors="pt").to(args.device)
            outputs = gdino_model(**inputs)
        results = gdino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=args.box_thr, text_threshold=args.text_thr,
            target_sizes=[pil.size[::-1]],
        )[0]

        boxes, scores = results["boxes"], results["scores"]
        if len(scores) == 0:
            print(f"  frame {i:02d}: no detection")
            continue

        best = scores.argmax()
        box   = boxes[best].cpu().tolist()
        score = scores[best].item()
        all_boxes[i]  = box
        all_scores[i] = score

        # SAM
        with torch.no_grad():
            sam_in = sam_proc(images=pil, input_boxes=[[box]],
                              return_tensors="pt").to(args.device)
            sam_out = sam_model(**sam_in)
        masks = sam_proc.post_process_masks(
            sam_out.pred_masks.cpu(),
            sam_in["original_sizes"].cpu(),
            sam_in["reshaped_input_sizes"].cpu(),
        )
        mask = masks[0][0][0].numpy().astype(np.uint8) * 255
        all_masks[i] = mask
        print(f"  frame {i:02d}: score={score:.3f}  box={[round(v,1) for v in box]}")

    np.savez_compressed(out_npz, masks=all_masks, boxes=all_boxes, scores=all_scores)

    meta = dict(
        video=str(video_path),
        noun=args.noun,
        half=args.half,
        fps=fps,
        frame_size=[W, H],
        n_frames=len(frames_bgr),
        box_thr=args.box_thr,
        text_thr=args.text_thr,
    )
    out_json.write_text(json.dumps(meta, indent=2))

    detected = int((all_scores > 0).sum())
    print(f"\nsaved {out_npz}  ({detected}/{len(frames_bgr)} frames detected)")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
