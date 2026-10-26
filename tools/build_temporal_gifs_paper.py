#!/usr/bin/env python3
"""GIFs for the five examples in the paper's main temporal figure
(figures/temporal_beat_tempo_5ex_noUnsteer.pdf): sandwich, car, giraffe, keyboard+phone (LTX-2),
teddy bear (Wan 2.1). Clip paths and control signals come from the report's paired.csv /
per-frame index, exactly as in that figure's builder.

    python tools/build_temporal_gifs_paper.py [--panel 300] [--stride 2]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPORT = Path.home() / "repos/video_probing/experiment_reports/2026_08_29_STEER_EVAL_temporal_qualitative_beat_tempo"
sys.path.insert(0, str(REPORT))
sys.path.insert(0, str(Path(__file__).parent))
import build_report as B  # noqa: E402
import build_selected as bs  # noqa: E402
from build_temporal_gifs import build  # noqa: E402

CAPTIONS = {"sandwich": "sandwich", "car": "car", "giraffe": "giraffe",
            "keyboard,phone": "keyboard + phone", "teddy bear": "teddy bear"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=int, default=300)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    B.PF = B._perframe_index()
    df = pd.read_csv(B.PAIRED)
    for bb, bench, q in bs.COMBINED:
        g = df[(df.backbone == bb) & (df.benchmark == bench)]
        m = (g[g.objects.astype(str).str.lower() == q.lower()] if bb == "LTX"
             else g[g.prompt.astype(str).str.lower().str.contains(q.lower(), regex=False)])
        r = m.iloc[0]
        ctrl = {o: "".join(str(int(x)) for x in s) for o, s in bs.controls_of(r).items()}
        dyn = next((o for o, c in ctrl.items() if len(set(c)) > 1), next(iter(ctrl)))
        label = "control" if len(ctrl) == 1 else f"control ({dyn})"
        slug = "paper_" + q.replace(",", "_").replace(" ", "_")
        cap = CAPTIONS[q] + (" (Wan 2.1)" if bb == "WAN" else " (LTX-2)")
        build(slug, cap, ctrl[dyn], label, r.mp4_tempo, r.mp4_ours, a.panel, a.stride)


if __name__ == "__main__":
    main()
