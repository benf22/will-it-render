#!/usr/bin/env python3
"""Build clean temporal-control MP4s for the website.

Creates two files per example in static/videos/temporal/:
  temporal_web2_{slug}.mp4  — 2-arm: TempoControl (left) | Ours (right)
  temporal_web3_{slug}.mp4  — 3-arm: Base model (left) | TempoControl | Ours (right)

No text is baked into the video — titles and timelines come from the HTML.

Run on the cluster (needs /cortex data access):
  cd ~/repos/will-it-render-project-page
  source ~/repos/LTX-2/.venv/bin/activate
  python tools/build_temporal_web.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import imageio
import decord
from PIL import Image
import pandas as pd

HERE   = Path(__file__).resolve().parent.parent   # repo root
OUT    = HERE / "static" / "videos" / "temporal"
OUT.mkdir(parents=True, exist_ok=True)

REPORT = Path.home() / "repos/video_probing/experiment_reports/2026_08_29_STEER_EVAL_temporal_qualitative_beat_tempo"
sys.path.insert(0, str(REPORT))
import build_report as B       # noqa — needed as transitive dep of build_selected
import build_selected as bs    # provides unsteer_for() / wan_unsteer_index()

PANEL_H = 320   # output panel height px; width auto-scales to preserve aspect ratio
STRIDE  = 2     # keep every 2nd source frame (halves fps, same as paper builds)

# Paper arm for LTX temporal control (LTX_perframe_v3 / bce_huber_m2).
# Used as ours override for examples where paired.csv references an older run.
LTX_PAPER_ARM = Path("/cortex/users/benfish/intermediate_results/video_neglacte_steering"
                     "/baselines/tempocontrol_ltx/LTX_bceHuberM2_allBench"
                     "/20260821_170259/frame_reducebce_huber_huber_m2_grad_balanceunitnorm")

# 6 examples: (backbone, benchmark, csv_query, slug, ours_override_or_None)
# csv_query matches 'objects' column exactly (LTX) or substring of 'prompt' (WAN).
# ours_override: explicit Path to the ours MP4, bypassing paired.csv (paper arm only).
WEB_EXAMPLES = [
    ("LTX", "single",         "car",          "car",          None),
    ("WAN", "single",         "teddy bear",   "teddy_bear",   None),
    ("LTX", "single",         "fire hydrant", "fire_hydrant", LTX_PAPER_ARM / "prompt_10__seed_42"  / "prompt_10__seed_42.mp4"),
    ("LTX", "static_dynamic", "airplane,boat","boat_airplane",LTX_PAPER_ARM / "prompt_121__seed_42" / "prompt_121__seed_42.mp4"),
    ("LTX", "single",         "giraffe",      "giraffe",      None),
    ("LTX", "static_dynamic", "car,bicycle",  "bicycle_car",  LTX_PAPER_ARM / "prompt_122__seed_42" / "prompt_122__seed_42.mp4"),
]


def read_frames(mp4: Path) -> tuple[list, float]:
    vr  = decord.VideoReader(str(mp4))
    fps = vr.get_avg_fps()
    raw = vr.get_batch(list(range(len(vr)))).asnumpy()
    out = []
    for f in raw:
        im   = Image.fromarray(f)
        w, h = im.size
        nw   = int(round(w * PANEL_H / h)) // 2 * 2
        out.append(im.resize((nw, PANEL_H), Image.LANCZOS))
    return out, fps


def vstack(frame_lists: list[list]) -> list[np.ndarray]:
    n = min(len(fl) for fl in frame_lists)
    out = []
    for i in range(n):
        total_h = sum(fl[i].height for fl in frame_lists)
        canvas  = Image.new("RGB", (frame_lists[0][i].width, total_h))
        y = 0
        for fl in frame_lists:
            canvas.paste(fl[i], (0, y)); y += fl[i].height
        out.append(np.array(canvas))
    return out


def hstack(frame_lists: list[list]) -> list[np.ndarray]:
    n = min(len(fl) for fl in frame_lists)
    out = []
    for i in range(n):
        total_w = sum(fl[i].width for fl in frame_lists)
        canvas  = Image.new("RGB", (total_w, frame_lists[0][i].height))
        x = 0
        for fl in frame_lists:
            canvas.paste(fl[i], (x, 0)); x += fl[i].width
        out.append(np.array(canvas))
    return out


def save(frames: list[np.ndarray], path: Path, fps: float) -> None:
    imageio.mimsave(str(path), frames, fps=fps, format="FFMPEG",
                    output_params=["-crf", "23", "-pix_fmt", "yuv420p",
                                   "-movflags", "+faststart"])
    print(f"  → {path.name}  {len(frames)} frames @ {fps:.1f} fps  "
          f"({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    df      = pd.read_csv(REPORT / "paired.csv")
    wan_idx = bs.wan_unsteer_index()
    B.PF    = {}   # not needed for unsteer lookup; skip slow _perframe_index()

    for bb, bench, q, slug, ours_override in WEB_EXAMPLES:
        print(f"\n── {slug} ──")
        g = df[(df.backbone == bb) & (df.benchmark == bench)]
        m = (g[g.objects.astype(str).str.lower() == q.lower()]
             if bb == "LTX"
             else g[g.prompt.astype(str).str.lower().str.contains(q.lower(), regex=False)])
        if len(m) == 0:
            print("  NOT FOUND in paired.csv — skip"); continue
        r = m.iloc[0]

        tc_p   = Path(r.mp4_tempo)
        ours_p = ours_override if ours_override is not None else Path(r.mp4_ours)
        uns_raw = bs.unsteer_for(r, wan_idx)
        uns_p   = Path(uns_raw) if uns_raw else None

        for label, p in [("TC", tc_p), ("Ours", ours_p)]:
            if not p.is_file():
                print(f"  {label} missing: {p}"); break
        else:
            tc_frames,   fps = read_frames(tc_p)
            ours_frames, _   = read_frames(ours_p)
            out_fps = fps / STRIDE

            # 2-arm: TempoControl | Ours
            frames2 = hstack([tc_frames, ours_frames])[::STRIDE]
            save(frames2, OUT / f"temporal_web2_{slug}.mp4", out_fps)

            # Mobile stacked: TempoControl (top) | Ours (bottom)
            frames_mob = vstack([tc_frames, ours_frames])[::STRIDE]
            save(frames_mob, OUT / f"temporal_mob_{slug}.mp4", out_fps)

            # 3-arm: Base model | TempoControl | Ours
            if uns_p and uns_p.is_file():
                uns_frames, _ = read_frames(uns_p)
                frames3 = hstack([uns_frames, tc_frames, ours_frames])[::STRIDE]
                save(frames3, OUT / f"temporal_web3_{slug}.mp4", out_fps)
            else:
                print(f"  no unsteer clip at {uns_p} — 3-arm skipped")


if __name__ == "__main__":
    main()
