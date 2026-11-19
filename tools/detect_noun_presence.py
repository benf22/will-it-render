#!/usr/bin/env python3
"""For each carousel clip, detect which nouns are present in ours (right) vs reference (left)."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THR = 0.25
TEXT_THR = 0.20
SAMPLE_N = 5   # evenly-spaced frames to sample per half

CLIPS = [
    {"id": "fig1_ltx375",   "nouns": ["snowboard", "car", "refrigerator"]},
    {"id": "fig1_ltx259",   "nouns": ["tie", "cow", "dog", "boat"]},
    {"id": "fig1_wan281",   "nouns": ["kite", "baseball bat"]},
    {"id": "fig1_wan318",   "nouns": ["bus", "teddy bear", "pizza", "broccoli"]},
    {"id": "fig1_wan231",   "nouns": ["tie", "book", "dog", "frisbee", "fork"]},
    {"id": "fig6_wan145",   "nouns": ["bicycle", "vase", "cup", "toaster", "microwave", "horse"]},
    {"id": "fig6_wan52",    "nouns": ["donut", "umbrella", "sink"]},
    {"id": "fig6_ltx4",     "nouns": ["sandwich", "book", "stop sign", "truck", "banana"]},
    {"id": "fig6_ltx42",    "nouns": ["oven", "skis", "toilet", "chair", "person", "tie", "cell phone"]},
    {"id": "fig6_cogx144",  "nouns": ["zebra", "elephant", "fire hydrant", "skis"]},
    {"id": "fig6_cogx168",  "nouns": ["refrigerator", "elephant", "kite"]},
    {"id": "fig12_so68",    "nouns": ["oven"]},
    {"id": "fig12_so17",    "nouns": ["horse"]},
    {"id": "fig12_so49",    "nouns": ["orange"]},
]


def load_half_frames(video_path: Path, half: str, n: int = SAMPLE_N):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [max(0, int(total * i / n)) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if half == "right":
            frame = frame[:, frame.shape[1] // 2:]
        else:
            frame = frame[:, :frame.shape[1] // 2]
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def detect_noun(proc, model, frames, noun: str) -> bool:
    text = f"{noun}."
    for pil in frames:
        inputs = proc(images=pil, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        results = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=BOX_THR, text_threshold=TEXT_THR,
            target_sizes=[pil.size[::-1]],
        )[0]
        if len(results["scores"]) > 0:
            return True
    return False


def main():
    video_dir = Path("static/videos/carousel")

    print("loading Grounding DINO …")
    proc  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base").to(DEVICE).eval()

    all_results = {}
    for clip in CLIPS:
        vid = video_dir / f"{clip['id']}.mp4"
        if not vid.exists():
            print(f"  {clip['id']}: NOT FOUND — skip")
            continue
        print(f"\n{clip['id']}")
        left_frames  = load_half_frames(vid, "left")
        right_frames = load_half_frames(vid, "right")

        ours_only = []
        for noun in clip["nouns"]:
            in_ref  = detect_noun(proc, model, left_frames,  noun)
            in_ours = detect_noun(proc, model, right_frames, noun)
            tag = ("both" if (in_ref and in_ours) else
                   "OURS_ONLY" if in_ours else
                   "ref_only"  if in_ref  else "neither")
            print(f"  {noun:22s}  ref={int(in_ref)}  ours={int(in_ours)}  → {tag}")
            if tag == "OURS_ONLY":
                ours_only.append(noun)

        all_results[clip["id"]] = {"ours_only": ours_only}

    out = Path("tools/noun_presence.json")
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nsaved {out}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
