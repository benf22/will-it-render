#!/usr/bin/env bash
# Run all remaining GSAM segmentation + render jobs (kite already done)
set -euo pipefail

PROJ=/home/lab/fishmab1/repos/will-it-render-project-page
VENV=~/repos/LTX-2/.venv/bin/activate
COLOR="#00dc3c"
THICKNESS=2
HALO=5

source "$VENV"
cd "$PROJ"

seg() {
    local video="$1" noun="$2"
    echo "=== SEG: $video | $noun ==="
    python tools/gsam_segment.py \
        --video "static/videos/carousel/${video}.mp4" \
        --noun "$noun" \
        --half right \
        --out static/videos/contours \
        --track
}

ren() {
    local video="$1" noun="$2"
    local slug="${video}_right_${noun// /_}"
    echo "=== RENDER: $slug ==="
    python tools/gsam_render.py \
        --masks "static/videos/contours/${slug}.npz" \
        --color "$COLOR" \
        --thickness $THICKNESS \
        --halo $HALO \
        --out "static/videos/contours/${slug}.mp4"
}

run() {
    local video="$1" noun="$2"
    local slug="${video}_right_${noun// /_}"
    if [[ ! -f "static/videos/contours/${slug}.npz" ]]; then
        seg "$video" "$noun"
    else
        echo "=== SKIP SEG (exists): $slug ==="
    fi
    if [[ ! -f "static/videos/contours/${slug}.mp4" ]]; then
        ren "$video" "$noun"
    else
        echo "=== SKIP RENDER (exists): $slug ==="
    fi
}

# fig1_ltx375
run fig1_ltx375 refrigerator

# fig1_ltx259
run fig1_ltx259 cow
run fig1_ltx259 dog
run fig1_ltx259 boat

# fig1_wan318
run fig1_wan318 broccoli

# fig1_wan231
run fig1_wan231 tie
run fig1_wan231 fork

# fig6_wan145
run fig6_wan145 horse
run fig6_wan145 vase

# fig6_wan52
run fig6_wan52 umbrella

# fig6_ltx4
run fig6_ltx4 "stop sign"
run fig6_ltx4 book
run fig6_ltx4 truck
run fig6_ltx4 banana

# fig6_ltx42
run fig6_ltx42 toilet
run fig6_ltx42 "cell phone"
run fig6_ltx42 chair
run fig6_ltx42 tie

# fig6_cogx144
run fig6_cogx144 skis

# fig6_cogx168
run fig6_cogx168 refrigerator

echo "=== ALL DONE ==="
