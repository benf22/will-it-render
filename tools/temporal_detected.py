# Per-frame object detection labels for temporal control examples.
# Marked manually using temporal_preview.html.
# 13-bit strings: frame 0 (leftmost) → frame 12 (rightmost).
# req = requested control signal; tc = TempoControl detected; ours = AttnSteer detected.
#
# Video sources:
#   Paper (MP4):        static/videos/temporal/temporal_paper_{key}.mp4
#   Supplementary (GIF): static/gifs/temporal_{key}.gif

# ── build_temporal_gifs_paper.py ─────────────────────────────────────────────
DETECTED = {
    "sandwich": {
        "tc":   "1111111111111",   # req: "0000000011111"
        "ours": "0000000111111",
    },
    "car": {
        "tc":   "0000000001111",   # req: "0001111111111"
        "ours": "0001111111111",
    },
    "giraffe": {
        "tc":   "1111111111111",   # req: "0000001111111"
        "ours": "0000001111111",
    },
    "keyboard_phone": {
        "tc":   "1111111111111",   # req: "1111111111111"
        "ours": "1111111111111",
    },
    "teddy_bear": {
        "tc":   "0000001111111",   # req: "0000000000111"
        "ours": "0000000000111",
    },
}

# ── build_temporal_gifs.py (supplementary) ───────────────────────────────────
DETECTED_SUPP = {
    "teddy_bear_supp": {
        "tc":   "0000000000000",   # req: "0000000000000"
        "ours": "0000000000000",
    },
    "giraffe_supp": {
        "tc":   "0000000000000",   # req: "0000000000000"
        "ours": "0000000000000",
    },
    "remote": {
        "tc":   "0000000000000",   # req: "0000000000000"
        "ours": "0000000000000",
    },
    "person": {
        "tc":   "0000000000000",   # req: "0000000000000"
        "ours": "0000000000000",
    },
    "fire_hydrant": {
        "tc":   "0000011111111",   # req: "0001111111111"
        "ours": "0001111111111",
    },
    "boat_airplane": {
        "tc":   "0111111111000",   # req: "0000001111111"
        "ours": "0000011111111",
    },
    "teddy_bear_frisbee": {
        "tc":   "0000000000000",   # req: "0000000000000"
        "ours": "0000000000000",
    },
    "bicycle_car": {
        "tc":   "0000000000000",   # req: "0000001111111"
        "ours": "0001111111111",
    },
}
