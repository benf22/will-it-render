# Will It Render? — project page

Project page for *Will It Render? Learned Attention Steering for Object Presence in Text-to-Video* (AttnSteer), built on the Academic Project Page Template layout.

Live at https://benf22.github.io/will-it-render/ (GitHub Pages, `main` / root; `.nojekyll` present).

## Preview locally

```bash
python -m http.server 8000
```

## Result videos

`static/videos/*.mp4` are built by `tools/build_videos.py` from the original generated mp4s (paths under `/cortex/...`, cluster only). Same prompts and seeds as the paper figures. Diagrams and charts in `static/images/` are rendered from the paper PDFs.

## TODO

- Fill in real author names, links and affiliations in `index.html`.
- Add paper PDF, arXiv and code links; replace the BibTeX with the camera-ready entry.
- Add a 1200x630 `static/images/social_preview.png` and point `og:image` at it.
