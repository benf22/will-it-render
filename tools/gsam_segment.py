#!/usr/bin/env python3
"""Run Grounded-SAM (GDINO + SAM) on a video and save per-frame masks.

Saves a .npz (masks, boxes, scores) + sidecar .json so gsam_render.py can
re-render the video with any visual config without re-running inference.

Usage:
    python tools/gsam_segment.py \\
        --video  static/videos/carousel/fig1_wan281.mp4 \\
        --noun   kite \\
        --half   right \\
        --out    static/videos/contours/ \\
        --track          # fill GDINO gaps with CSRT + smooth boxes

Tracking flags:
    --track          enable CSRT gap-filling + Gaussian box smoothing
    --smooth-sigma   Gaussian sigma for box smoothing (default 1.5)
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
from scipy.ndimage import gaussian_filter1d
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_frames(video_path: Path, half: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if half == "right":
            frame = frame[:, frame.shape[1] // 2:]
        elif half == "left":
            frame = frame[:, : frame.shape[1] // 2]
        frames.append(frame)
    cap.release()
    return frames, fps


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def xyxy_to_xywh(box: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(b) -> list[float]:
    x, y, w, h = b
    return [x, y, x + w, y + h]


def run_sam(sam_model, sam_proc, pil: Image.Image, box: list[float], device: str) -> np.ndarray:
    with torch.no_grad():
        sam_in = sam_proc(images=pil, input_boxes=[[box]],
                          return_tensors="pt").to(device)
        sam_out = sam_model(**sam_in)
    masks = sam_proc.post_process_masks(
        sam_out.pred_masks.cpu(),
        sam_in["original_sizes"].cpu(),
        sam_in["reshaped_input_sizes"].cpu(),
    )
    return masks[0][0][0].numpy().astype(np.uint8) * 255


def smooth_boxes(boxes: np.ndarray, scores: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth box coords over time, only interpolating where score > 0."""
    smoothed = boxes.copy()
    detected = scores > 0
    if detected.sum() < 2:
        return smoothed
    # fill gaps with linear interpolation before smoothing
    for coord in range(4):
        col = boxes[:, coord].copy()
        # fill undetected indices via linear interpolation between detections
        det_idx = np.where(detected)[0]
        col_filled = np.interp(np.arange(len(col)), det_idx, col[det_idx])
        smoothed[:, coord] = gaussian_filter1d(col_filled, sigma=sigma)
    return smoothed


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",        required=True)
    ap.add_argument("--noun",         required=True)
    ap.add_argument("--half",         default="full", choices=["left", "right", "full"])
    ap.add_argument("--out",          default="static/videos/contours")
    ap.add_argument("--box-thr",      type=float, default=0.25)
    ap.add_argument("--text-thr",     type=float, default=0.20)
    ap.add_argument("--track",        action="store_true",
                    help="Fill GDINO gaps with CSRT tracker + smooth boxes")
    ap.add_argument("--smooth-sigma", type=float, default=2.5,
                    help="Gaussian sigma for box smoothing (--track only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    video_path = Path(args.video)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug     = f"{video_path.stem}_{args.half}_{args.noun.replace(' ', '_')}"
    out_npz  = out_dir / f"{slug}.npz"
    out_json = out_dir / f"{slug}.json"

    print(f"loading video: {video_path}")
    frames_bgr, fps = load_frames(video_path, args.half)
    H, W = frames_bgr[0].shape[:2]
    N    = len(frames_bgr)
    print(f"  {N} frames  {W}×{H}  {fps:.1f} fps")

    # ── load models ──────────────────────────────────────────────────────────
    print("loading Grounding DINO …")
    gdino_proc  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base").to(args.device)
    gdino_model.eval()

    print("loading SAM …")
    sam_proc  = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(args.device)
    sam_model.eval()

    text_prompt = f"{args.noun.lower()}."

    all_masks  = np.zeros((N, H, W), dtype=np.uint8)
    all_boxes  = np.zeros((N, 4),    dtype=np.float32)
    all_scores = np.zeros(N,         dtype=np.float32)
    source     = ["none"] * N   # "gdino" | "tracker" | "none"

    # ── pass 1: GDINO on every frame ─────────────────────────────────────────
    print("pass 1: GDINO detection …")
    for i, frame_bgr in enumerate(frames_bgr):
        pil = bgr_to_pil(frame_bgr)
        with torch.no_grad():
            inputs  = gdino_proc(images=pil, text=text_prompt,
                                 return_tensors="pt").to(args.device)
            outputs = gdino_model(**inputs)
        results = gdino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=args.box_thr, text_threshold=args.text_thr,
            target_sizes=[pil.size[::-1]],
        )[0]
        boxes, scores = results["boxes"], results["scores"]
        if len(scores) > 0:
            best = scores.argmax()
            all_boxes[i]  = boxes[best].cpu().tolist()
            all_scores[i] = scores[best].item()
            source[i]     = "gdino"
            print(f"  frame {i:02d}: gdino  score={all_scores[i]:.3f}")
        else:
            print(f"  frame {i:02d}: no detection")

    detected = int((all_scores > 0).sum())
    print(f"  GDINO detected {detected}/{N} frames")

    # ── pass 2 (optional): interpolate boxes across all frames ──────────────
    if args.track and detected > 0:
        print("pass 2: interpolating + smoothing boxes …")
        # Interpolate box coords between GDINO detections (np.interp clamps at
        # edges, so pre/post-detection frames get nearest detection's box).
        gdino_mask = (all_scores > 0).astype(float)
        all_boxes = smooth_boxes(all_boxes, gdino_mask, args.smooth_sigma)

        # Mark all frames as having a box (interp fills the gaps)
        for i in range(N):
            if all_scores[i] == 0:
                all_scores[i] = -0.5   # interp flag
                source[i]     = "interp"
                print(f"  frame {i:02d}: interp  box={[round(v,1) for v in all_boxes[i]]}")

    # ── pass 3: SAM on every box-filled frame ────────────────────────────────
    need_sam = all_scores != 0
    print(f"pass 3: SAM on {need_sam.sum()} frames …")
    for i in range(N):
        if not need_sam[i]:
            continue
        pil  = bgr_to_pil(frames_bgr[i])
        mask = run_sam(sam_model, sam_proc, pil, all_boxes[i].tolist(), args.device)
        all_masks[i] = mask
        print(f"  frame {i:02d}: sam ok  src={source[i]}")

    np.savez_compressed(out_npz, masks=all_masks, boxes=all_boxes, scores=all_scores)

    meta = dict(
        video=str(video_path), noun=args.noun, half=args.half,
        fps=fps, frame_size=[W, H], n_frames=N,
        box_thr=args.box_thr, text_thr=args.text_thr,
        tracking=args.track, smooth_sigma=args.smooth_sigma,
        source=source,
    )
    out_json.write_text(json.dumps(meta, indent=2))

    filled = int(need_sam.sum())
    print(f"\nsaved {out_npz}  ({filled}/{N} frames with mask)")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
