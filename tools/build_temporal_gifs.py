#!/usr/bin/env python3
"""Temporal-control example GIFs: TempoControl | Ours, synced, with a control timeline.

One GIF per example. Above the two clips is a timeline of the requested control
signal (green = the object must be present in that frame, grey = it must be absent)
with a marker that follows the current frame. Clip matching and control signals
reuse the paper's figures/build_temporal_qual.py (matched by prompt text; control
bits from the scored protocol JSONs).

    python tools/build_temporal_gifs.py [--panel 300] [--stride 2]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import decord
import numpy as np
from PIL import Image, ImageDraw, ImageFont

B = "/cortex/users/benfish/intermediate_results/video_neglacte_steering/baselines/tempocontrol_ltx"
RUNS = {
    1: {"TempoControl": f"{B}/tempo_one_object/20260818_192059"},
    2: {"TempoControl": f"{B}/tempo_two_objects"},
}
OURS = (f"{B}/LTX_bceHuberM2_allBench/20260821_170259/"
        "frame_reducebce_huber_huber_m2_grad_balanceunitnorm")
SCORED = os.path.expanduser("~/repos/video_probing/.claude/worktrees/feat+timeline-steering/_scratch")
NF = 13
OUT = Path(__file__).resolve().parent.parent / "static" / "gifs"

# (ours prompt dir, benchmark level, slug, caption, objects)
CASES = [
    ("prompt_77", 1, "teddy_bear", "teddy bear", ["teddy bear"]),
    ("prompt_23", 1, "giraffe", "giraffe", ["giraffe"]),
    ("prompt_65", 1, "remote", "remote", ["remote"]),
    ("prompt_0", 1, "person", "person", ["person"]),
    ("prompt_10", 1, "fire_hydrant", "fire hydrant", ["fire hydrant"]),
    ("prompt_121", 2, "boat_airplane", "boat + airplane", ["boat", "airplane"]),
    ("prompt_107", 2, "teddy_bear_frisbee", "teddy bear + frisbee", ["teddy bear", "frisbee"]),
    ("prompt_122", 2, "bicycle_car", "bicycle + car", ["bicycle", "car"]),
]

GREEN = (47, 125, 50)
GREY = (225, 225, 225)
DARK = (34, 34, 34)
MUTED = (110, 110, 110)
BLUE = (53, 125, 189)
_FONTDIRS = ["/usr/share/fonts/dejavu-sans-fonts", "/usr/share/fonts/truetype/dejavu",
             "/usr/share/fonts/dejavu"]


def font(sz, bold=False):
    name = f"DejaVuSans{'-Bold' if bold else ''}.ttf"
    for d in _FONTDIRS:
        p = Path(d) / name
        if p.is_file():
            return ImageFont.truetype(str(p), sz)
    return ImageFont.load_default()


def index_by_prompt(run):
    m = {}
    for pj in glob.glob(f"{run}/**/prompt_info.json", recursive=True):
        j = json.load(open(pj))
        v = glob.glob(pj.rsplit("/", 1)[0] + "/*.mp4")
        if v:
            m[j["prompt"].strip()] = v[0]
    return m


def detections():
    """video_path -> {object: control_bit_string} from the scored protocol JSONs."""
    S = lambda v: "".join(str(int(x)) for x in v)
    out = {}
    for f in glob.glob(f"{SCORED}/**/*.json", recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "per_detector" not in d:
            continue
        for r in d["per_detector"].get("majority_ours", {}).get("per_video", []):
            pf = r.get("per_frame") or {}
            if pf and all(len(v["control"]) == NF for v in pf.values()):
                out[r["video_path"]] = {w: S(v["control"]) for w, v in pf.items()}
    return out


def read_all(mp4, panel):
    vr = decord.VideoReader(mp4)
    fps = vr.get_avg_fps()
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    out = []
    for f in frames:
        im = Image.fromarray(f)
        w, h = im.size
        out.append(im.resize((int(round(w * panel / h)) // 2 * 2, panel), Image.LANCZOS))
    return out, fps


def global_palette(frames, colors=256):
    """One shared palette per GIF (no flicker); UI colours get their own block so they never drift."""
    step = max(1, len(frames) // 8)
    sample = [f.resize((f.width // 2, f.height // 2)) for f in frames[::step]]
    ui = Image.new("RGB", (len(sample[0].size) * 0 + sample[0].width, 40))
    d = ImageDraw.Draw(ui)
    for i, col in enumerate([GREEN, GREY, DARK, MUTED, BLUE, (255, 255, 255)]):
        d.rectangle([i * 40, 0, i * 40 + 39, 39], fill=col)
    mosaic = Image.new("RGB", (sample[0].width, sample[0].height * len(sample) + 40))
    for i, f in enumerate(sample):
        mosaic.paste(f, (0, i * f.height))
    mosaic.paste(ui, (0, sample[0].height * len(sample)))
    return mosaic.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def centered(d, x0, x1, y, s, f, fill):
    d.text((x0 + (x1 - x0 - d.textlength(s, font=f)) / 2, y), s, font=f, fill=fill)


def build(slug, caption, ctrl, ctrl_label, tempo_mp4, ours_mp4, panel, stride):
    tempo, fps_src = read_all(tempo_mp4, panel)
    ours, _ = read_all(ours_mp4, panel)
    n = min(len(tempo), len(ours))
    pw, pw2 = tempo[0].size[0], ours[0].size[0]
    NF = len(ctrl)
    margin, gap = 12, 8
    W = margin * 2 + pw + pw2 + gap
    title_h, tl_h, arm_h = 30, 52, 28
    H = margin + title_h + tl_h + arm_h + panel + margin
    H += H % 2
    tl_x0, tl_x1 = margin, W - margin
    tl_y = margin + title_h + 14
    seg = (tl_x1 - tl_x0) / NF
    ft, fa, fs = font(20, True), font(18, True), font(12)
    frames = []
    for t in range(0, n, stride):
        c = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(c)
        d.text((margin, margin), caption, font=ft, fill=DARK)
        d.text((W - margin - d.textlength(ctrl_label, font=fs), margin + 6), ctrl_label, font=fs, fill=MUTED)
        for i, bit in enumerate(ctrl):
            x0 = tl_x0 + i * seg
            d.rectangle([x0 + 1, tl_y, x0 + seg - 1, tl_y + 18], fill=GREEN if bit == "1" else GREY)
        ly = tl_y + 24
        d.rectangle([tl_x0, ly + 2, tl_x0 + 12, ly + 12], fill=GREEN)
        d.text((tl_x0 + 18, ly), "object should appear", font=fs, fill=MUTED)
        lx = tl_x0 + 18 + d.textlength("object should appear", font=fs) + 16
        d.rectangle([lx, ly + 2, lx + 12, ly + 12], fill=GREY)
        d.text((lx + 18, ly), "object should be absent", font=fs, fill=MUTED)
        mx = tl_x0 + (t / max(n - 1, 1)) * (tl_x1 - tl_x0)
        d.line([mx, tl_y - 6, mx, tl_y + 24], fill=DARK, width=3)
        d.polygon([(mx - 6, tl_y - 10), (mx + 6, tl_y - 10), (mx, tl_y - 2)], fill=DARK)
        y = margin + title_h + tl_h
        centered(d, margin, margin + pw, y + 2, "TempoControl", fa, DARK)
        centered(d, margin + pw + gap, margin + pw + pw2 + gap, y + 2, "Ours", fa, BLUE)
        c.paste(tempo[t], (margin, y + arm_h))
        c.paste(ours[t], (margin + pw + gap, y + arm_h))
        frames.append(c)
    pal = global_palette(frames)
    frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"temporal_{slug}.gif"
    fps = fps_src / stride
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(round(1000 / fps)),
                   loop=0, optimize=True, disposal=1)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames, {W}x{H})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=int, default=300)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    det = detections()
    print(f"scored videos with per-frame control: {len(det)}")
    ix_tempo = {lvl: index_by_prompt(r["TempoControl"]) for lvl, r in RUNS.items()}
    ix_ours = index_by_prompt(OURS)
    for pdir, lvl, slug, caption, objs in CASES:
        info = json.load(open(f"{OURS}/{pdir}__seed_42/prompt_info.json"))
        ptxt = info["prompt"].strip()
        ours_mp4, tempo_mp4 = ix_ours.get(ptxt), ix_tempo[lvl].get(ptxt)
        if not ours_mp4 or not tempo_mp4 or ours_mp4 not in det:
            print(f"  ! skip {slug}: ours={bool(ours_mp4)} tempo={bool(tempo_mp4)} ctrl={ours_mp4 in det}")
            continue
        cands = [(w, det[ours_mp4][w]) for w in objs if w in det[ours_mp4]]
        dyn = next((w for w, c in cands if len(set(c)) > 1), cands[0][0])
        label = "control" if len(objs) == 1 else f"control ({dyn})"
        build(slug, caption, det[ours_mp4][dyn], label, tempo_mp4, ours_mp4, a.panel, a.stride)


if __name__ == "__main__":
    main()
