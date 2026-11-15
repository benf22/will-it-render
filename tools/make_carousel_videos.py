#!/usr/bin/env python3
"""Create side-by-side (unsteer left, ours right) MP4 videos for the webpage carousel.

Encodes each pair at 480px per half (960px total width), H.264 CRF 20, yuv420p.
Output: static/videos/carousel/<id>.mp4
"""
import subprocess
import sys
from pathlib import Path

VS = "/private/schwartz-lab/benfish/intermediate_results/video_steering"
VN = "/cortex/users/benfish/intermediate_results/video_neglacte_steering"

# --- LTX (Oct 30 night seed-matched run) ---
_LTX_BASE  = f"{VS}/2026_10_30_LTX_seedmatch_night"
UN_LTX     = f"{_LTX_BASE}/q_unsteer/20261030_014524/n_iters0"
OURS_LTX   = (f"{_LTX_BASE}/q_fix/20261030_014529/"
              "grad_aggregategated0.95_mutual_mutual_orderworst_first")

# --- WAN (unsteer = Oct 30; ours = Nov 2 fullres per-k run) ---
_WAN_UN_BASE  = f"{VS}/2026_10_30_WAN_seedmatch_night"
UN_WAN        = f"{_WAN_UN_BASE}/unsteer/20261030_013122/n_iters0"
_WAN_OURS_BASE = f"{VS}/2026_11_02_WAN_fullres_ls05_mg_ppseed"
_WAN_CFG = ("grad_aggregategated0.95_mutual_zero_nearest_probeTrue_mutual_foldanchor"
            "_mutual_orderworst_first_lr5.0_n_iters8_stop_when_min_prob0.95_solver_f720862a")
_WAN_OURS_K = {
    2: f"{_WAN_OURS_BASE}/k2/20261102_061048/{_WAN_CFG}",
    3: f"{_WAN_OURS_BASE}/k3/20261102_094043/{_WAN_CFG}",
    4: f"{_WAN_OURS_BASE}/k4/20261102_094049/{_WAN_CFG}",
    5: f"{_WAN_OURS_BASE}/k5/20261102_094053/{_WAN_CFG}",
    6: f"{_WAN_OURS_BASE}/k6/20261102_094057/{_WAN_CFG}",
}

# --- CogVideoX (Oct 30 night seed-matched run) ---
_COGX_BASE = f"{VS}/2026_10_30_COGX_seedmatch_night"
UN_COGX    = f"{_COGX_BASE}/q_unsteer/20261030_014542/n_iters0"
OURS_COGX  = (f"{_COGX_BASE}/q_fix/20261030_014546/"
              "grad_aggregategated0.95_mutual_zero_mutual_orderworst_first_lr5.0")

# --- Single-object baseline (formal 473 run) ---
SO_BASE  = f"{VN}/eval/2026_08_22_vbench_singleobj_ltx_ours/20260822_111516"
OURS_SO  = (f"{SO_BASE}/loss_typemean_anchor_reg5.0_kl_reg0.0_lr20.0"
            "_magnitude_reg0.0001_optimizersimple_pred_reg0.03")


def ltx_un(pid, seed):   return f"{UN_LTX}/prompt_{pid}__seed_{seed}/prompt_{pid}__seed_{seed}.mp4"
def ltx_ours(pid, seed): return f"{OURS_LTX}/prompt_{pid}__seed_{seed}/prompt_{pid}__seed_{seed}.mp4"
def wan_un(pid, seed):   return f"{UN_WAN}/prompt_{pid}__seed_{seed}/prompt_{pid}__seed_{seed}.mp4"
def wan_ours(pid, seed, k): return f"{_WAN_OURS_K[k]}/prompt_{pid}__seed_{seed}/prompt_{pid}__seed_{seed}.mp4"
def cogx_un(pid, seed):  return f"{UN_COGX}/prompt_{pid}__seed_{seed}/prompt_{pid}__seed_{seed}.mp4"
def cogx_ours(pid, seed):return f"{OURS_COGX}/prompt_{pid}__seed_{seed}/prompt_{pid}__seed_{seed}.mp4"
def so_un(pid):          return f"{SO_BASE}/n_iters0/prompt_{pid}/prompt_{pid}.mp4"
def so_ours(pid):        return f"{OURS_SO}/prompt_{pid}/prompt_{pid}.mp4"


# (id, prompt_text, left_mp4, right_mp4, figure_ref)
PAIRS = [
    # --- Figure 1: teaser ---
    ("fig1_ltx375",  "a snowboard and a car and a refrigerator",
     ltx_un(375, 2102239827), ltx_ours(375, 2102239827), "fig1"),
    ("fig1_ltx259",  "a tie and a cow and a dog and a boat",
     ltx_un(259, 1547425198), ltx_ours(259, 1547425198), "fig1"),
    ("fig1_wan281",  "a kite and a baseball bat",
     wan_un(281, 1892721232), wan_ours(281, 1892721232, 2), "fig1"),
    ("fig1_wan318",  "a bus and a teddy bear and a pizza and broccoli",
     wan_un(318, 363949293), wan_ours(318, 363949293, 4), "fig1"),
    ("fig1_wan231",  "a tie and a book and a dog and a frisbee and a fork",
     wan_un(231, 1373273833), wan_ours(231, 1373273833, 5), "fig1"),

    # --- Figure 6: multi-object comparison ---
    ("fig6_wan145",  "a bicycle and a vase and a cup and a toaster and a microwave and a horse",
     wan_un(145, 372581924), wan_ours(145, 372581924, 6), "fig6"),
    ("fig6_wan52",   "a donut and an umbrella and a sink",
     wan_un(52, 1393641349), wan_ours(52, 1393641349, 3), "fig6"),
    ("fig6_ltx4",    "a sandwich and a book and a stop sign and a truck and a banana",
     ltx_un(4, 2105765209), ltx_ours(4, 2105765209), "fig6"),
    ("fig6_ltx42",   "an oven and skis and a toilet and a chair and a person and a tie and a cell phone",
     ltx_un(42, 1029298087), ltx_ours(42, 1029298087), "fig6"),
    ("fig6_cogx144", "a zebra and an elephant and a fire hydrant and skis",
     cogx_un(144, 437134111), cogx_ours(144, 437134111), "fig6"),
    ("fig6_cogx168", "a refrigerator and an elephant and a kite",
     cogx_un(168, 1356492204), cogx_ours(168, 1356492204), "fig6"),

    # --- Figure 12: single-object examples ---
    ("fig12_so68",   "an oven",
     so_un(68),  so_ours(68),  "fig12"),
    ("fig12_so17",   "a horse",
     so_un(17),  so_ours(17),  "fig12"),
    ("fig12_so49",   "an orange",
     so_un(49),  so_ours(49),  "fig12"),
]

OUT_DIR = Path(__file__).resolve().parent.parent / "static" / "videos" / "carousel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HALF_W = 480   # scale each side to this width; height auto (keeps aspect ratio)


def make_sbs(left: str, right: str, out: Path) -> bool:
    if out.exists():
        print(f"  skip (exists): {out.name}")
        return True
    cmd = [
        "ffmpeg", "-y",
        "-i", left, "-i", right,
        "-filter_complex",
        f"[0:v]scale={HALF_W}:-2[l];[1:v]scale={HALF_W}:-2[r];[l][r]hstack[v]",
        "-map", "[v]",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"  ERROR: {out.name}\n{r.stderr.decode()[-500:]}", file=sys.stderr)
        return False
    size_kb = out.stat().st_size // 1024
    print(f"  OK: {out.name}  ({size_kb} KB)")
    return True


if __name__ == "__main__":
    ok = err = 0
    for vid_id, prompt, left, right, fig in PAIRS:
        out = OUT_DIR / f"{vid_id}.mp4"
        print(f"[{fig}] {vid_id}")
        if make_sbs(left, right, out):
            ok += 1
        else:
            err += 1
    print(f"\nDone: {ok} OK, {err} errors")
