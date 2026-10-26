#!/usr/bin/env python3
"""Build the project-page result videos from the original generated mp4s.

Mirrors the clip choices of the paper figures (figures/build_teaser.py,
build_method_compare.py, build_fig_singleobj.py in the paper repo):
  teaser.mp4            un-steered (top) vs ours (bottom), 5 prompts side by side
  multi_wan_*.mp4       Wan 2.1: un-steered | +A&E | +Ours
  multi_ltx_*.mp4       LTX-2:   un-steered | +A&E | +Ours
  single_*.mp4          LTX-2 single object: un-steered | +A&E | +Ours

    python tools/build_videos.py [--only teaser|multi|single]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import decord
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VN = "/cortex/users/benfish/intermediate_results/video_neglacte_steering"
VS = "/cortex/users/benfish/intermediate_results/video_steering"
OUT = Path(__file__).resolve().parent.parent / "static" / "videos"

MISSING_C = (193, 39, 45)
PRESENT_C = (34, 34, 34)
BG = (255, 255, 255)
FPS = 24

_FONTDIRS = ["/usr/share/fonts/dejavu-sans-fonts", "/usr/share/fonts/truetype/dejavu",
             "/usr/share/fonts/dejavu"]


def font(sz: int, bold: bool = False) -> ImageFont.ImageFont:
    name = f"DejaVuSans{'-Bold' if bold else ''}.ttf"
    for d in _FONTDIRS:
        p = Path(d) / name
        if p.is_file():
            return ImageFont.truetype(str(p), sz)
    return ImageFont.load_default()


def read_frames(mp4: str, n_out: int, crop_lr: float = 0.0, height: int | None = None):
    """n_out frames sampled uniformly over the clip's normalised timeline."""
    vr = decord.VideoReader(mp4)
    n = len(vr)
    idx = np.linspace(0, n - 1, n_out).round().astype(int)
    frames = vr.get_batch(idx.tolist()).asnumpy()
    out = []
    for f in frames:
        im = Image.fromarray(f)
        if crop_lr > 0:
            w, h = im.size
            dx = int(round(w * crop_lr))
            im = im.crop((dx, 0, w - dx, h))
        if height:
            w, h = im.size
            im = im.resize((int(round(w * height / h)) // 2 * 2, height), Image.LANCZOS)
        out.append(im)
    return out


def text_w(draw, s, f):
    return draw.textlength(s, font=f)


def draw_objects(canvas: Image.Image, x0: int, x1: int, y: int, objects, missing, sz: int):
    """Wrapped, centred object list; missing objects bold red. Returns y after block."""
    d = ImageDraw.Draw(canvas)
    fr, fb = font(sz), font(sz, True)
    toks = []
    for i, o in enumerate(objects):
        s = o + ("," if i < len(objects) - 1 else "")
        f = fb if o in missing else fr
        toks.append((s, o in missing, f, text_w(d, s, f)))
    space = sz * 0.35
    maxw = x1 - x0 - 8
    lines, cur, w = [], [], 0.0
    for t in toks:
        if cur and w + t[3] > maxw:
            lines.append(cur)
            cur, w = [], 0.0
        cur.append(t)
        w += t[3] + space
    if cur:
        lines.append(cur)
    for line in lines:
        total = sum(t[3] for t in line) + space * (len(line) - 1)
        x = x0 + (x1 - x0 - total) / 2
        for s, miss, f, tw in line:
            d.text((x, y), s, font=f, fill=MISSING_C if miss else PRESENT_C)
            x += tw + space
        y += int(sz * 1.35)
    return y


def centered(canvas, x0, x1, y, s, sz, bold=True, fill=PRESENT_C):
    d = ImageDraw.Draw(canvas)
    f = font(sz, bold)
    d.text((x0 + (x1 - x0 - text_w(d, s, f)) / 2, y), s, font=f, fill=fill)


def write(frames: list[np.ndarray], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=None,
                           pixelformat="yuv420p", macro_block_size=2,
                           ffmpeg_params=["-crf", "27", "-preset", "slow",
                                          "-movflags", "+faststart"])
    for f in frames:
        w.append_data(f)
    w.close()
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


def write_gif(frames: list[np.ndarray], path: Path, stride: int = 2):
    """Animated GIF: every `stride`-th frame, one shared 256-colour palette (no flicker, text colours kept)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ims = [Image.fromarray(f) for f in frames[::stride]]
    sample = [im.resize((im.width // 2, im.height // 2)) for im in ims[:: max(1, len(ims) // 8)]]
    mosaic = Image.new("RGB", (sample[0].width, sample[0].height * len(sample) + 40), BG)
    for i, im in enumerate(sample):
        mosaic.paste(im, (0, i * im.height))
    d = ImageDraw.Draw(mosaic)
    for i, col in enumerate([MISSING_C, PRESENT_C, BG, (53, 125, 189), (47, 125, 50)]):
        d.rectangle([i * 40, mosaic.height - 40, i * 40 + 39, mosaic.height - 1], fill=col)
    pal = mosaic.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(round(1000 * stride / FPS)),
              loop=0, optimize=True, disposal=1)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(q)} frames, {ims[0].width}x{ims[0].height})")


# ---- clip sources -----------------------------------------------------------

def wan(i):
    return {
        "Un-steered": f"{VS}/2026_08_20_wan_unsteer_473/20260820_234536/n_iters0/prompt_{i}/prompt_{i}.mp4",
        "+ A&E": f"{VS}/2026_08_20_wan_ae_473/20260820_113016/lr500.0/prompt_{i}/prompt_{i}.mp4",
        "+ AttnSteer": f"{VS}/2026_08_20_wan_B_short_473/20260820_233004/"
                       f"nearest_probeTrue_lr5_n_iters16_steps_to_steer0-1-2-3-4-5-6-7-8-9-10/"
                       f"prompt_{i}/prompt_{i}.mp4",
    }


def ltx(i):
    return {
        "Un-steered": f"{VN}/eval/2026_08_09_gennosteer_short473/20260809_171558/n_iters0/prompt_{i}/prompt_{i}.mp4",
        "+ A&E": f"{VN}/baselines/2026_08_13_ae_473_short/20260813_002259/lr500.0/prompt_{i}/prompt_{i}.mp4",
        "+ AttnSteer": f"{VS}/2026_08_23_ltx_mutual_noreg_lr15_span_unitnorm_473/gen/"
                       f"prompt_{i}__seed_42/prompt_{i}__seed_42.mp4",
    }


WAN_ROWS = [
    ("multi_wan_motorcycle_boat", ["motorcycle", "boat"], {"motorcycle"}, wan(213)),
    ("multi_wan_cow_banana_broccoli", ["cow", "banana", "broccoli"], {"broccoli"}, wan(315)),
    ("multi_wan_fridge_banana_cat", ["refrigerator", "banana", "cat"], {"refrigerator", "cat"}, wan(369)),
    ("multi_wan_seven_objects",
     ["sheep", "oven", "carrot", "orange", "couch", "knife", "sandwich"],
     {"oven", "carrot", "couch", "knife"}, wan(221)),
]
LTX_ROWS = [
    ("multi_ltx_fridge_skis", ["refrigerator", "skis"], {"skis"}, ltx(335)),
    ("multi_ltx_pizza_book_bus", ["pizza", "book", "bus"], {"book", "bus"}, ltx(17)),
    ("multi_ltx_tie_cow_dog_boat", ["tie", "cow", "dog", "boat"], {"cow", "dog", "boat"}, ltx(259)),
    ("multi_ltx_bus_teddy_pizza_broccoli", ["bus", "teddy bear", "pizza", "broccoli"],
     {"bus", "teddy bear"}, ltx(318)),
]

_E_LO = Path(VN) / "eval/2026_08_22_vbench_singleobj_ltx_ours/20260822_111516"
_E_LAE = Path(VN) / "eval/2026_08_22_vbench_singleobj_ltx_ae/20260822_111521"
_OURS = "loss_typemean_anchor_reg5.0_kl_reg0.0_lr20.0_magnitude_reg0.0001_optimizersimple_pred_reg0.03"


def single(idx):
    return {
        "Un-steered": str(_E_LO / "n_iters0" / f"prompt_{idx}" / f"prompt_{idx}.mp4"),
        "+ A&E": str(_E_LAE / "lr500.0" / f"prompt_{idx}" / f"prompt_{idx}.mp4"),
        "+ AttnSteer": str(_E_LO / _OURS / f"prompt_{idx}" / f"prompt_{idx}.mp4"),
    }


SINGLE_ROWS = [("single_oven", "a oven", 68), ("single_horse", "a horse", 17),
               ("single_orange", "a orange", 49)]


# ---- composition ------------------------------------------------------------

def build_compare(name, objects, missing, srcs, panel_h=300, n_out=96):
    """Three panels (un-steered | +A&E | ours) with titles and a coloured object list."""
    arms = list(srcs)
    clips = {a: read_frames(srcs[a], n_out, height=panel_h) for a in arms}
    pw = clips[arms[0]][0].size[0]
    gap, margin, title_h, text_h = 8, 10, 40, 74
    W = margin * 2 + pw * 3 + gap * 2
    H = margin + title_h + panel_h + text_h + margin // 2
    H += H % 2
    out = []
    for t in range(n_out):
        c = Image.new("RGB", (W, H), BG)
        for j, a in enumerate(arms):
            x = margin + j * (pw + gap)
            centered(c, x, x + pw, margin, a, 24)
            c.paste(clips[a][t], (x, margin + title_h))
        draw_objects(c, margin, W - margin, margin + title_h + panel_h + 8, objects, missing, 24)
        out.append(np.asarray(c))
    write(out, OUT / f"{name}.mp4")


def build_single(name, phrase, idx, panel_h=300, n_out=96, gif=False):
    srcs = single(idx)
    arms = list(srcs)
    clips = {a: read_frames(srcs[a], n_out, height=panel_h) for a in arms}
    pw = clips[arms[0]][0].size[0]
    gap, margin, title_h, text_h = 8, 10, 40, 44
    W = margin * 2 + pw * 3 + gap * 2
    H = margin + title_h + panel_h + text_h + margin // 2
    H += H % 2
    out = []
    for t in range(n_out):
        c = Image.new("RGB", (W, H), BG)
        for j, a in enumerate(arms):
            x = margin + j * (pw + gap)
            centered(c, x, x + pw, margin, a, 24)
            c.paste(clips[a][t], (x, margin + title_h))
        centered(c, margin, W - margin, margin + title_h + panel_h + 8, f"“{phrase}”", 26, bold=False)
        out.append(np.asarray(c))
    if gif:
        write_gif(out, OUT.parent / "gifs" / f"{name}.gif")
    else:
        write(out, OUT / f"{name}.mp4")


def build_teaser(n_out=96, panel_h=250):
    E = f"{VN}/eval"

    def newest(pat):
        d = sorted(glob.glob(f"{E}/{pat}/2026*"))
        return d[-1] if d else None

    def first(d, pid):
        h = sorted(glob.glob(f"{d}/prompt_{pid}/*.mp4")) or sorted(glob.glob(f"{d}/prompt_{pid}__seed_*/*.mp4"))
        return h[0] if h else None

    un_ltx_old = newest("2026_06_24_v1v2classbal_vbench_ref_vbench_short")
    mat = newest("2026_08_07_optrule_matched_short_hi")
    lr50 = next((c for c in glob.glob(f"{mat}/*") if "lr50.0" in Path(c).name and Path(c).is_dir()), None) if mat else None
    un_ltx = f"{VN}/eval/2026_08_09_gennosteer_short473/20260809_171558/n_iters0"
    ours_ltx = f"{VS}/2026_08_23_ltx_mutual_noreg_lr15_span_unitnorm_473/gen"
    un_wan = f"{VS}/2026_08_20_wan_unsteer_473/20260820_234536/n_iters0"
    ours_wan = (f"{VS}/2026_08_20_wan_B_short_473/20260820_233004/"
                "nearest_probeTrue_lr5_n_iters16_steps_to_steer0-1-2-3-4-5-6-7-8-9-10")

    cols = []
    if un_ltx_old and lr50:
        cols.append(("LTX-2", 2, ["bicycle", "car"], {"bicycle"},
                     first(f"{un_ltx_old}/n_iters0", 244), first(lr50, 244), 0.0))
    cols.append(("LTX-2", 3, ["potted plant", "car", "baseball glove"], {"car"},
                 first(un_ltx, 142), first(ours_ltx, 142), 0.0))
    if un_ltx_old and lr50:
        cols.append(("LTX-2", 7, ["sheep", "clock", "person", "sink", "backpack", "hot dog", "teddy bear"],
                     {"sheep", "hot dog"}, first(f"{un_ltx_old}/n_iters0", 101), first(lr50, 101), 0.0))
    cols.append(("Wan 2.1", 3, ["umbrella", "pizza", "carrot"], {"carrot"},
                 first(un_wan, 141), first(ours_wan, 141), 0.07))
    cols.append(("Wan 2.1", 5, ["apple", "cup", "oven", "tennis racket", "bottle"], {"oven", "bottle"},
                 first(un_wan, 377), first(ours_wan, 377), 0.07))
    cols = [c for c in cols if c[4] and c[5]]
    print(f"teaser columns: {len(cols)}")

    data = [(c, read_frames(c[4], n_out, c[6], panel_h), read_frames(c[5], n_out, c[6], panel_h)) for c in cols]
    gap, margin, label_w, head_h, row_gap, text_h = 8, 10, 175, 56, 6, 96
    widths = [d[1][0].size[0] for d in data]
    W = margin * 2 + label_w + sum(widths) + gap * len(widths)
    H = margin + head_h + panel_h * 2 + row_gap + text_h + margin
    H += H % 2
    W += W % 2
    out = []
    for t in range(n_out):
        c = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(c)
        f = font(24, True)
        for r, lab in enumerate(["Un-steered", "AttnSteer"]):
            y = margin + head_h + r * (panel_h + row_gap) + panel_h // 2 - 14
            dr.text((margin, y), lab, font=f, fill=PRESENT_C)
        x = margin + label_w
        for (meta, un, ours), w in zip(data, widths):
            model, k, objs, miss, *_ = meta
            centered(c, x, x + w, margin, model, 22)
            centered(c, x, x + w, margin + 26, f"k = {k}", 22, bold=False, fill=(90, 90, 90))
            c.paste(un[t], (x, margin + head_h))
            c.paste(ours[t], (x, margin + head_h + panel_h + row_gap))
            draw_objects(c, x, x + w, margin + head_h + panel_h * 2 + row_gap + 6, objs, miss, 20)
            x += w + gap
        out.append(np.asarray(c))
    write(out, OUT / "teaser.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["teaser", "multi", "single"])
    ap.add_argument("--gif", action="store_true", help="single-object rows: write static/gifs/*.gif instead of mp4")
    a = ap.parse_args()
    if a.gif:
        for name, phrase, idx in SINGLE_ROWS:
            build_single(name, phrase, idx, gif=True)
        return
    if a.only in (None, "teaser"):
        build_teaser()
    if a.only in (None, "multi"):
        for name, objs, miss, srcs in WAN_ROWS + LTX_ROWS:
            build_compare(name, objs, miss, srcs)
    if a.only in (None, "single"):
        for name, phrase, idx in SINGLE_ROWS:
            build_single(name, phrase, idx)


if __name__ == "__main__":
    main()
