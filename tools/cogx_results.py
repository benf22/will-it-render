#!/usr/bin/env python3
"""CogVideoX-5B per-k results: unsteered vs A&E vs ours (formal gated0.95_mutual_zero), det-all and det-ratio.

Same definitions as _scratch/nightrun_1019/score.py: det-all = clip passes iff every non-(-1) object label is 1,
SE = sqrt(p(1-p)/n); det-ratio = mean(label==1) over non-(-1) object rows, SE clip-clustered.
Only prompts labeled in all three arms are used, per k. Mean over k: SE = sqrt(sum SE_k^2)/K.

    python tools/cogx_results.py [--html]   # --html also refreshes the table in index.html; writes static/data/cogx_results.json and prints the table
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "repos/video_probing/_scratch/nightrun_1019"))
import score  # noqa: E402

RR = score.RR
C = f"{RR}/2026_10_09_cogx_induce_fixedcfg"
KS = [2, 3, 4, 5, 6, 7]
OUT = Path(__file__).resolve().parent.parent / "static" / "data" / "cogx_results.json"


def pidx(s):
    return int(re.search(r"prompt_(\d+)", str(s)).group(1))


def arms(k):
    un = score.rd(f"{C}/labels_k{k}/det.csv")
    out = {
        "Unsteered": un[un.extraction_dir.str.contains(f"cogx_unsteer_k{k}/")],
        "A&E": un[un.extraction_dir.str.contains(f"cogx_ae_k{k}/")],
        "Ours": score.sources("COGX", k, "formal", None)[0],
    }
    res = {}
    for a, d in out.items():
        d = d.assign(p=d.extraction_dir.map(pidx), label=d.label.astype(float).astype(int))
        res[a] = d[d.label != -1]
    common = set.intersection(*[set(d.p) for d in res.values()])
    return {a: d[d.p.isin(common)] for a, d in res.items()}, len(common)


def stats(d):
    g = d.groupby("p").label.apply(lambda s: bool((s == 1).all()))
    n, p = len(g), float(g.mean())
    r = d.groupby("p").label.apply(lambda s: (s == 1).mean())
    return dict(n=n, det_all=p, det_all_se=float(np.sqrt(p * (1 - p) / n)),
                det_ratio=float((d.label == 1).mean()), det_ratio_se=float(r.std(ddof=1) / np.sqrt(n)))


def main():
    res = {a: {} for a in ("Unsteered", "A&E", "Ours")}
    for k in KS:
        d, n = arms(k)
        for a in res:
            res[a][k] = stats(d[a])
        print(f"k{k}: n={n} common prompts")
    mean = {}
    for a in res:
        mean[a] = {m: dict(v=float(np.mean([res[a][k][m] for k in KS])),
                           se=float(np.sqrt(sum(res[a][k][m + "_se"] ** 2 for k in KS)) / len(KS)))
                   for m in ("det_all", "det_ratio")}
    for m in ("det_all", "det_ratio"):
        print(f"\n{m}")
        for a in res:
            print(f"  {a:10s}", " ".join(f"{res[a][k][m]:.3f}±{res[a][k][m + '_se']:.3f}" for k in KS),
                  f"| mean {mean[a][m]['v']:.3f}±{mean[a][m]['se']:.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"ks": KS, "per_k": res, "mean": mean}, indent=1))


def html():
    """Table block for index.html (between the cogx-results markers)."""
    d = json.loads(OUT.read_text())
    ks, per_k, mean = d["ks"], d["per_k"], d["mean"]
    names = {"Unsteered": "CogVideoX-5B", "A&E": "&nbsp;&nbsp;+ A&amp;E", "Ours": "&nbsp;&nbsp;+ AttnSteer"}
    rows = []
    for m, title in (("det_all", "All objects detected (det-all, %)"), ("det_ratio", "Objects detected (det-ratio, %)")):
        rows.append(f'          <tr class="block-sep"><td colspan="{len(ks) + 2}"><b>{title}</b></td></tr>')
        best = {k: max(per_k[a][str(k)][m] for a in names) for k in ks}
        bmean = max(mean[a][m]["v"] for a in names)
        for a, label in names.items():
            cells = []
            for k in ks:
                v, se = per_k[a][str(k)][m], per_k[a][str(k)][m + "_se"]
                t = f'{100 * v:.1f} <span class="has-text-grey is-size-7">&plusmn;{100 * se:.1f}</span>'
                cells.append(f"<td><b>{t}</b></td>" if v == best[k] else f"<td>{t}</td>")
            v, se = mean[a][m]["v"], mean[a][m]["se"]
            t = f'{100 * v:.1f} <span class="has-text-grey is-size-7">&plusmn;{100 * se:.1f}</span>'
            cells.append(f"<td><b>{t}</b></td>" if v == bmean else f"<td>{t}</td>")
            rows.append(f"          <tr><td>{label}</td>{''.join(cells)}</tr>")
    head = "".join(f"<th><i>k</i>={k}</th>" for k in ks)
    return ('<div class="table-container">\n      <table class="table is-fullwidth is-narrow is-size-7 results-table">\n'
            f"        <thead><tr><th>Model</th>{head}<th>Mean</th></tr></thead>\n        <tbody>\n"
            + "\n".join(rows) + "\n        </tbody>\n      </table>\n    </div>")


if __name__ == "__main__":
    main()
    if "--html" in sys.argv:
        p = Path(__file__).resolve().parent.parent / "index.html"
        s = p.read_text()
        a, b = s.index("<!-- cogx-results:start -->"), s.index("<!-- cogx-results:end -->")
        p.write_text(s[:a] + "<!-- cogx-results:start -->\n    " + html() + "\n    " + s[b:])
