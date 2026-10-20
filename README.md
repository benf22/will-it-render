# Will It Render? — project page

Static project page for *Will It Render? Learned Attention Steering for Object Presence in Text-to-Video* (AttnSteer), built on the Academic Project Page Template layout (Bulma, hero, teaser, abstract, results, copyable BibTeX, CC BY-SA footer).

## Preview locally

```bash
python -m http.server 8000
# open http://localhost:8000
```

## Deploy

GitHub Pages serves the repo root from the default branch (`.nojekyll` is present).

## Before going public

- Fill in authors and affiliations in `index.html` (currently "Anonymous Authors" — WACV review is double-blind).
- Add the paper PDF, arXiv, and code links in the hero block.
- Replace the BibTeX with the camera-ready entry.
- Add a 1200x630 social preview at `static/images/social_preview.png` and point `og:image` at it.
- Figures in `static/images/` are rendered from the paper's PDFs (`figures/*.pdf` in the paper repo). Re-render them if the paper figures change.
- Table values are copied from the paper's `tables/` and `sections/4_experiments.tex`. The Wan 2.1 +A&E VBench cell and the A&E k=7 value (1.4) for Wan 2.1 still need checking against the reports.
