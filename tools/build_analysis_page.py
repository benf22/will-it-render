#!/usr/bin/env python3
"""Generate analysis_temporal.html — a local analysis page for picking website examples.

Loads per-frame detections from tempo_eval JSON files; outputs HTML with 4 sections:
  1. Single-object — top 20 by f1_ours
  2. Single-object — ours > TC, TC partially succeeds (not all-0 or all-1 detected)
  3. Static+dynamic — top 20 by f1_ours
  4. Static+dynamic — ours > TC, TC partially succeeds

Merged side-by-side MP4s (TC | Ours) are built into static/videos/temporal_analysis/.
Pass --rebuild to force re-encoding even when the MP4 already exists.

Run:
  cd ~/repos/will-it-render-project-page
  source ~/repos/LTX-2/.venv/bin/activate
  python tools/build_analysis_page.py [--rebuild]
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import numpy as np
import imageio
import decord
from PIL import Image
import pandas as pd

HERE      = Path(__file__).resolve().parent.parent
OUT       = HERE / "analysis_temporal.html"
VIDS_OUT  = HERE / "static" / "videos" / "temporal_analysis"
VIDS_OUT.mkdir(parents=True, exist_ok=True)

TEMPO_EVAL = Path.home() / "repos/video_probing/.claude/worktrees/feat+timeline-steering/_scratch/tempo_eval"
REPORT     = Path.home() / "repos/video_probing/experiment_reports/2026_08_29_STEER_EVAL_temporal_qualitative_beat_tempo"

DETECTOR   = "majority_ours"
N_SHOW     = 20
N_CELLS    = 13   # timeline cells in the HTML
PANEL_H    = 320  # height of each panel in the merged video
REBUILD    = "--rebuild" in sys.argv


# ── video building ────────────────────────────────────────────────────────────

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


def build_merged(tc_path: str, ours_path: str, slug: str) -> str:
    """Build TC|Ours side-by-side MP4; return relative src path for HTML."""
    out_path = VIDS_OUT / f"an_{slug}.mp4"
    if out_path.exists() and not REBUILD:
        return f"static/videos/temporal_analysis/an_{slug}.mp4"
    tc_p, ours_p = Path(tc_path), Path(ours_path)
    if not tc_p.is_file() or not ours_p.is_file():
        print(f"  MISSING video for {slug}")
        return ""
    tc_frames, fps   = read_frames(tc_p)
    ours_frames, _   = read_frames(ours_p)
    frames = hstack([tc_frames, ours_frames])
    imageio.mimsave(str(out_path), frames, fps=fps, format="FFMPEG",
                    output_params=["-crf", "23", "-pix_fmt", "yuv420p",
                                   "-movflags", "+faststart"])
    print(f"  → {out_path.name}  {len(frames)}f @ {fps:.1f}fps  "
          f"({out_path.stat().st_size/1e6:.1f}MB)")
    return f"static/videos/temporal_analysis/an_{slug}.mp4"


# ── data loading ─────────────────────────────────────────────────────────────

def load_perframe(json_path: Path) -> dict:
    """Return {video_path: {obj: {"detected": [...], "control": [...]}}}"""
    d = json.loads(json_path.read_text())
    per_det = (d.get("per_detector") or {}).get(DETECTOR) or {}
    idx: dict = {}
    for v in per_det.get("per_video", []):
        pf = v.get("per_frame") or {}
        if pf:
            idx[v["video_path"]] = pf
    return idx


def to_n(arr: list, n: int = N_CELLS) -> list[int]:
    """Downsample/resample a detection array to exactly n cells."""
    if len(arr) == n:
        return list(arr)
    return [int(arr[int(i * (len(arr) - 1) / (n - 1))]) for i in range(n)]


def compute_f1(detected: list, control: list) -> float:
    det  = [bool(d) for d in detected]
    ctrl = [bool(c) for c in control]
    n    = min(len(det), len(ctrl))
    tp   = sum(det[i] and ctrl[i] for i in range(n))
    fp   = sum(det[i] and not ctrl[i] for i in range(n))
    fn   = sum(not det[i] and ctrl[i] for i in range(n))
    p    = tp / (tp + fp) if tp + fp else 0.0
    r    = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


# ── example building ──────────────────────────────────────────────────────────

def extract_allbench_idx(video_path: str) -> int | None:
    m = re.search(r"prompt_(\d+)__seed_42", video_path)
    return int(m.group(1)) if m else None


def build_examples(bench: str, tc_pf: dict, bce_pf: dict, df: pd.DataFrame) -> list[dict]:
    sub = df[(df.backbone == "LTX") & (df.benchmark == bench)].copy()

    # Build lookup: allbench_idx -> bce entry
    bce_by_abidx: dict[int, tuple[str, dict]] = {}
    for vp, pf in bce_pf.items():
        ab = extract_allbench_idx(vp)
        if ab is not None:
            bce_by_abidx[ab] = (vp, pf)

    examples = []
    for _, row in sub.iterrows():
        pid = int(row.prompt_idx)
        ab  = pid if bench == "single" else (80 + pid)

        if ab not in bce_by_abidx:
            continue
        ours_path, bce_entry = bce_by_abidx[ab]
        tc_path   = str(row.mp4_tempo)
        tc_entry  = tc_pf.get(tc_path, {})
        if not tc_entry:
            continue

        # Collect per-object data; identify the dynamic object
        objects_str = str(row.objects)
        objects_list = [o.strip().lower() for o in objects_str.split(",")]

        obj_data: dict[str, dict] = {}
        for obj in objects_list:
            tc_o  = tc_entry.get(obj)
            bce_o = bce_entry.get(obj)
            if not tc_o or not bce_o:
                continue
            tc_det  = tc_o.get("detected", [])
            tc_ctrl = tc_o.get("control",   [])
            bce_det = bce_o.get("detected", [])

            if not tc_det or not tc_ctrl or not bce_det:
                continue

            is_dynamic = 0 < sum(tc_ctrl) < len(tc_ctrl)  # not all-0 or all-1

            obj_data[obj] = {
                "tc_detected":  to_n(tc_det),
                "bce_detected": to_n(bce_det),
                "control":      to_n(tc_ctrl),
                "f1_tc":        compute_f1(tc_det, tc_ctrl),
                "f1_ours":      compute_f1(bce_det, tc_ctrl),
                "is_dynamic":   is_dynamic,
            }

        if not obj_data:
            continue

        # Primary scoring object: dynamic if available, else first
        dyn = [o for o, d in obj_data.items() if d["is_dynamic"]]
        primary = dyn[0] if dyn else list(obj_data.keys())[0]

        f1_ours = obj_data[primary]["f1_ours"]
        f1_tc   = obj_data[primary]["f1_tc"]

        # For display: show only the controlled (dynamic) object's timeline
        display_obj = primary

        slug = f"{'s' if bench == 'single' else 'd'}_{pid}"
        examples.append({
            "pid":          pid,
            "slug":         slug,
            "bench":        bench,
            "objects":      objects_list,
            "display_obj":  display_obj,
            "tc_path":      tc_path,
            "ours_path":    ours_path,
            "control":      obj_data[display_obj]["control"],
            "tc_detected":  obj_data[display_obj]["tc_detected"],
            "bce_detected": obj_data[display_obj]["bce_detected"],
            "f1_tc":        f1_tc,
            "f1_ours":      f1_ours,
            "delta_f1":     f1_ours - f1_tc,
        })

    return examples


def tc_partial(ex: dict) -> bool:
    det = ex["tc_detected"]
    return 0 < sum(det) < len(det)


# ── HTML generation ───────────────────────────────────────────────────────────

def frames_attr(arr: list[int]) -> str:
    return ",".join(str(x) for x in arr)


def card_html(ex: dict, card_idx: int) -> str:
    objects_label = " + ".join(o.capitalize() for o in ex["objects"])
    controlled    = ex["display_obj"].capitalize()
    if len(ex["objects"]) > 1:
        objects_label += f" <span style='color:#888;font-weight:400;font-size:0.82rem;'>(controlled: {controlled})</span>"

    def f1_col(v):
        col = "#15803d" if v >= 0.7 else "#dc2626" if v < 0.4 else "#d97706"
        return f"<b style='color:{col}'>{v:.2f}</b>"

    vid_src = build_merged(ex["tc_path"], ex["ours_path"], ex["slug"])
    vid_tag = (f'<video autoplay muted loop playsinline preload="metadata" '
               f'src="{vid_src}"></video>' if vid_src else
               '<div style="background:#222;color:#aaa;padding:1rem;text-align:center">video unavailable</div>')

    return f"""
    <div class="analysis-card" id="ex-{card_idx}">
      <div class="card-meta">
        <span class="card-idx">#{ex['pid']}</span>
        <span class="card-objects">{objects_label}</span>
        <span class="card-scores">TC: {f1_col(ex['f1_tc'])} &nbsp; Ours: {f1_col(ex['f1_ours'])} &nbsp; Δ: {ex['delta_f1']:+.2f}</span>
      </div>
      <div class="card-col-headers">
        <span>TempoControl</span>
        <span class="ours-label">AttnSteer (ours)</span>
      </div>
      {vid_tag}
      <div class="timeline-block">
        <div class="timeline-rows-wrap">
          <div class="timeline-cursor"><div class="cursor-line"></div></div>
          <div class="timeline-row row-requested">
            <span class="timeline-label">Requested</span>
            <div class="timeline-track" data-frames="{frames_attr(ex['control'])}"></div>
          </div>
          <div class="timeline-row row-tc">
            <span class="timeline-label">TempoControl</span>
            <div class="timeline-track" data-frames="{frames_attr(ex['tc_detected'])}"></div>
          </div>
          <div class="timeline-row row-ours">
            <span class="timeline-label ours-label">AttnSteer (ours)</span>
            <div class="timeline-track" data-frames="{frames_attr(ex['bce_detected'])}"></div>
          </div>
        </div>
      </div>
    </div>"""


def section_html(title: str, desc: str, examples: list[dict]) -> str:
    cards = "\n".join(card_html(ex, i) for i, ex in enumerate(examples))
    return f"""
  <section class="analysis-section">
    <h2>{title}</h2>
    <p class="section-desc">{desc}</p>
    <div class="analysis-grid">
{cards}
    </div>
  </section>"""


PAGE_CSS = """
body { font-family: 'Figtree', Arial, sans-serif; background: #f0f6ff; color: #1f2937; margin: 0; }
h1 { text-align:center; padding: 1.5rem 1rem 0.5rem; font-size:1.6rem; color:#1d4ed8; }
.page-desc { text-align:center; color:#6b7280; margin-bottom:1.5rem; font-size:0.92rem; }

.analysis-section { background:#fff; border-top:3px solid #3b6fb6; border-radius:10px;
  box-shadow:0 2px 8px rgba(0,0,0,.07); max-width:900px; margin:1.5rem auto; padding:1.5rem 2rem; }
.analysis-section h2 { font-size:1.15rem; color:#1d4ed8; margin:0 0 0.3rem; }
.section-desc { color:#6b7280; font-size:0.85rem; margin:0 0 1rem; }

.analysis-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }

.analysis-card { border:1px solid #e5e7eb; border-radius:8px; overflow:hidden;
  background:#fafafa; }

.card-meta { display:flex; align-items:baseline; gap:0.75rem; padding:6px 10px 5px;
  background:#f1f5f9; border-bottom:1px solid #e5e7eb; flex-wrap:wrap; }
.card-idx { font-size:0.75rem; color:#94a3b8; font-weight:700; }
.card-objects { font-weight:700; font-size:0.9rem; }
.card-scores { font-size:0.82rem; margin-left:auto; }

.card-col-headers { display:grid; grid-template-columns:1fr 1fr;
  background:#111; padding:2px 0; }
.card-col-headers span { text-align:center; font-size:0.68rem; font-weight:700;
  letter-spacing:.06em; text-transform:uppercase; color:#9ca3af; }
.card-col-headers .ours-label { color:#60a5fa; }

.analysis-card video { display:block; width:100%; height:auto; }
"""

PAGE_JS = """
// Initialize timeline cells
document.querySelectorAll('.timeline-track[data-frames]').forEach(function(track) {
  track.getAttribute('data-frames').split(',').forEach(function(v) {
    var cell = document.createElement('div');
    cell.className = 'timeline-cell' + (parseInt(v) ? ' present' : '');
    track.appendChild(cell);
  });
});

// Sync cursor to video playback
document.querySelectorAll('.analysis-card').forEach(function(card) {
  var vid    = card.querySelector('video');
  var cursor = card.querySelector('.cursor-line');
  if (!vid || !cursor) return;
  function sync() {
    if (vid.duration)
      cursor.style.left = (vid.currentTime / vid.duration * 100).toFixed(2) + '%';
  }
  vid.addEventListener('timeupdate', sync);
  vid.addEventListener('seeked', sync);
});
"""


def build_page(cat1, cat2, cat3, cat4) -> str:
    s1 = section_html(
        "1 — Single-object: top by AttnSteer F1",
        f"LTX-2, single-object benchmark. Top {N_SHOW} examples ranked by AttnSteer F1 (↓ = best).",
        cat1)
    s2 = section_html(
        "2 — Single-object: ours > TC, TC partially succeeds",
        f"LTX-2, single-object. ours F1 > TC F1, TC detection is non-trivial "
        f"(not all-absent and not all-present). Top {N_SHOW} by Δ F1.",
        cat2)
    s3 = section_html(
        "3 — Static+dynamic: top by AttnSteer F1",
        f"LTX-2, static+dynamic benchmark. Timeline shows the controlled (dynamic) object. "
        f"Top {N_SHOW} by AttnSteer F1.",
        cat3)
    s4 = section_html(
        "4 — Static+dynamic: ours > TC, TC partially succeeds",
        f"LTX-2, static+dynamic. ours F1 > TC F1 on the dynamic object, "
        f"TC has non-trivial detection. Top {N_SHOW} by Δ F1.",
        cat4)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Temporal analysis — example picker</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="static/css/index.css">
  <style>{PAGE_CSS}</style>
</head>
<body>
  <h1>Temporal control — example picker</h1>
  <p class="page-desc">LTX-2 only · majority_ours detector · bceHuberM2 paper arm</p>
{s1}
{s2}
{s3}
{s4}
  <script>{PAGE_JS}</script>
</body>
</html>"""


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading detection JSONs …")
    tc_one  = load_perframe(TEMPO_EVAL / "tempoControl_oneObject_num32.json")
    tc_two  = load_perframe(TEMPO_EVAL / "tempoControl_twoObjects_num32.json")
    bce_one = load_perframe(TEMPO_EVAL / "bcehuberm2_oneObject_num32.json")
    bce_two = load_perframe(TEMPO_EVAL / "bcehuberm2_twoObjects_num32.json")
    print(f"  TC one={len(tc_one)}  TC two={len(tc_two)}  "
          f"bce one={len(bce_one)}  bce two={len(bce_two)}")

    df = pd.read_csv(REPORT / "paired.csv")

    print("Building examples …")
    single  = build_examples("single",         tc_one, bce_one, df)
    two_obj = build_examples("static_dynamic",  tc_two, bce_two, df)
    print(f"  single={len(single)}  static_dynamic={len(two_obj)}")

    cat1 = sorted(single,  key=lambda x: x["f1_ours"],   reverse=True)[:N_SHOW]
    cat2 = sorted([e for e in single  if e["delta_f1"] > 0 and tc_partial(e)],
                  key=lambda x: x["delta_f1"], reverse=True)[:N_SHOW]
    cat3 = sorted(two_obj, key=lambda x: x["f1_ours"],   reverse=True)[:N_SHOW]
    cat4 = sorted([e for e in two_obj if e["delta_f1"] > 0 and tc_partial(e)],
                  key=lambda x: x["delta_f1"], reverse=True)[:N_SHOW]

    print(f"  cat1={len(cat1)} cat2={len(cat2)} cat3={len(cat3)} cat4={len(cat4)}")

    # Deduplicate by slug so we don't encode the same video twice
    seen_slugs: set[str] = set()
    all_cats = [cat1, cat2, cat3, cat4]
    total = sum(1 for cat in all_cats for ex in cat if ex["slug"] not in seen_slugs
                and not seen_slugs.add(ex["slug"]))
    print(f"Building {total} unique merged videos …")

    html = build_page(cat1, cat2, cat3, cat4)
    OUT.write_text(html)
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
