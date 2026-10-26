#!/usr/bin/env python3
"""Per-k example GIFs: un-steered | AttnSteer side by side, for LTX-2, Wan 2.1 and CogVideoX-5B, k = 2..7.

Examples for k that appear in the paper teaser / poster are reused; the other cells use the clip where
the un-steered video misses the most objects, AttnSteer detects all of them and A&E still fails (then checked by eye).
Missing objects (per arm, n=32 detector labels used in the paper; -1 = uncertain is not counted) are shown in red.

    python tools/build_k_examples_gifs.py            # build static/gifs/kex_*.gif
    python tools/build_k_examples_gifs.py --sheet    # candidate contact sheets -> /tmp/kex_sheets
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / "repos/video_probing/_scratch/nightrun_1019"))
import build_videos as V  # noqa: E402
from build_videos import VN, VS, centered, draw_objects, font, read_frames, write_gif  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "static" / "gifs"
CROP_WAN = 0.07
NAMES = {"LTX": "LTX-2", "WAN": "Wan 2.1", "COGX": "CogVideoX-5B"}

LAB = {
    "LTX": {"un": f"{VN}/eval/2026_08_09_gennosteer_short473/20260809_171558/object_labels_n32/labels.csv",
            "ours": f"{VS}/2026_08_23_ltx_mutual_noreg_lr15_span_unitnorm_473/object_labels_n32/labels.csv"},
    "WAN": {"un": f"{VS}/2026_08_20_wan_unsteer_473/20260820_234536/obj_labels_n32.csv",
            "ours": f"{VS}/2026_08_20_wan_B_short_473/20260820_233004/ours_obj_labels_n32.csv"},
}

# (backbone, k) -> prompt idx
SELECT = {
    ("LTX", 2): 244, ("LTX", 3): 17, ("LTX", 4): 259, ("LTX", 5): 159, ("LTX", 6): 234, ("LTX", 7): 101,
    ("WAN", 2): 213, ("WAN", 3): 141, ("WAN", 4): 380, ("WAN", 5): 377, ("WAN", 6): 119, ("WAN", 7): 221,
    ("COGX", 2): 9, ("COGX", 3): 17, ("COGX", 4): 22, ("COGX", 5): 64, ("COGX", 6): 446, ("COGX", 7): 218,
}
CANDS = {
    ("LTX", 5): [159, 420, 284], ("LTX", 6): [234, 48], ("WAN", 4): [236, 380], ("WAN", 6): [119, 306],
    ("COGX", 2): [9, 96, 192, 335], ("COGX", 3): [17, 103, 121, 143], ("COGX", 4): [22, 128, 163],
    ("COGX", 5): [64, 81, 357], ("COGX", 6): [446, 33, 169], ("COGX", 7): [218, 340, 241],
}


def _mp4(d):
    h = sorted(glob.glob(f"{d}/*.mp4"))
    return h[0] if h else None


def _cogx_arms(k):
    import cogx_results as C
    d, _ = C.arms(k)
    return d


_cache = {}


def labels(bb, arm, i, k):
    """(objects in prompt order, set of missing objects) for one arm."""
    if bb == "COGX":
        d = _cache.setdefault(k, _cogx_arms(k))[{"un": "Unsteered", "ours": "Ours"}[arm]]
        g = d[d.p == i]
        return list(g.object_name), set(g[g.label != 1].object_name)
    d = _cache.setdefault((bb, arm), pd.read_csv(LAB[bb][arm]))
    g = d[(d.prompt_idx == i) & (d.label.astype(float) != -1)]
    return list(g.object_name), set(g[g.label.astype(float) != 1].object_name)


def clips(bb, i, k):
    if bb == "LTX":
        s = V.ltx(i)
    elif bb == "WAN":
        s = V.wan(i)
    else:
        d = _cache.setdefault(k, _cogx_arms(k))
        s = {a: _mp4(d[n][d[n].p == i].extraction_dir.iloc[0]) for a, n in (("Un-steered", "Unsteered"), ("+ AttnSteer", "Ours"))}
    return s["Un-steered"], s["+ AttnSteer"]


def build(bb, k, i, panel=300, n_out=96, stride=2, tag="kex"):
    un, ours = clips(bb, i, k)
    crop = CROP_WAN if bb == "WAN" else 0.0
    fu, fo = read_frames(un, n_out, crop, panel), read_frames(ours, n_out, crop, panel)
    objs_u, miss_u = labels(bb, "un", i, k)
    objs_o, miss_o = labels(bb, "ours", i, k)
    objs = objs_u if len(objs_u) >= len(objs_o) else objs_o
    pw = fu[0].size[0]
    gap, margin, title_h, arm_h, text_h = 8, 10, 34, 30, 90
    W = margin * 2 + pw * 2 + gap
    H = margin + title_h + arm_h + panel + text_h + margin // 2
    H += H % 2
    frames = []
    for t in range(n_out):
        c = Image.new("RGB", (W, H), V.BG)
        centered(c, margin, W - margin, margin, f"{NAMES[bb]}  ·  k = {k}", 22)
        for j, (name, fr, miss) in enumerate((("Un-steered", fu, miss_u), ("AttnSteer", fo, miss_o))):
            x = margin + j * (pw + gap)
            centered(c, x, x + pw, margin + title_h, name, 20, fill=(53, 125, 189) if j else V.PRESENT_C)
            c.paste(fr[t], (x, margin + title_h + arm_h))
            draw_objects(c, x, x + pw, margin + title_h + arm_h + panel + 6, objs, miss, 15)
        frames.append(np.asarray(c))
    write_gif(frames, OUT / f"{tag}_{bb.lower()}_k{k}.gif", stride)


def sheets():
    d = Path("/tmp/kex_sheets")
    d.mkdir(exist_ok=True)
    for (bb, k), ids in CANDS.items():
        rows = []
        for i in ids:
            un, ours = clips(bb, i, k)
            crop = CROP_WAN if bb == "WAN" else 0.0
            ims = []
            for m in (un, ours):
                fr = read_frames(m, 3, crop, 240)[1:2]
                ims += fr
            ou, mu = labels(bb, "un", i, k)
            oo, mo = labels(bb, "ours", i, k)
            row = Image.new("RGB", (sum(x.width for x in ims) + 8 * len(ims), 240 + 22), (255, 255, 255))
            x = 0
            for im in ims:
                row.paste(im, (x, 22))
                x += im.width + 8
            ImageDraw.Draw(row).text((2, 2), f"{bb} k{k} p{i}  unMiss={sorted(mu)}  oursMiss={sorted(mo)}", font=font(13, True), fill=(0, 0, 0))
            rows.append(row)
        W = max(r.width for r in rows)
        sh = Image.new("RGB", (W, sum(r.height for r in rows)), (255, 255, 255))
        y = 0
        for r in rows:
            sh.paste(r, (0, y))
            y += r.height
        sh.save(d / f"{bb}_k{k}.png")
        print("sheet", d / f"{bb}_k{k}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.sheet:
        sheets()
    else:
        for (bb, k), i in SELECT.items():
            if a.only and a.only.lower() not in f"{bb}_k{k}".lower():
                continue
            build(bb, k, i)
