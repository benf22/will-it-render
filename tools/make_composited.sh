#!/usr/bin/env bash
# For each carousel clip: merge all-noun contours into one right-half video,
# then ffmpeg-stitch left-half-unsteered + right-half-contour → carousel_contour/<clip>.mp4
set -euo pipefail

PROJ=/home/lab/fishmab1/repos/will-it-render-project-page
VENV=~/repos/LTX-2/.venv/bin/activate
FFMPEG=/home/lab/fishmab1/miniforge3/bin/ffmpeg
COLOR="#00dc3c"
THICKNESS=2
HALO=5

source "$VENV"
cd "$PROJ"

mkdir -p static/videos/carousel_contour

# Args: video noun1 noun2 ...
make_clip() {
    local video="$1"; shift
    local nouns=("$@")

    local out_composited="static/videos/carousel_contour/${video}.mp4"
    if [[ -f "$out_composited" ]]; then
        echo "=== SKIP (exists): $video ==="
        return
    fi

    # Build masks arg list
    local masks_args=()
    for noun in "${nouns[@]}"; do
        masks_args+=("static/videos/contours/${video}_right_${noun// /_}.npz")
    done

    local merged="static/videos/contours/${video}_right_merged.mp4"
    echo "=== MERGE contours: $video [${nouns[*]}] ==="
    python tools/gsam_render.py \
        --masks "${masks_args[@]}" \
        --color "$COLOR" --thickness $THICKNESS --halo $HALO \
        --out "$merged"

    echo "=== STITCH: $video ==="
    $FFMPEG -y \
        -i "static/videos/carousel/${video}.mp4" \
        -i "$merged" \
        -filter_complex "[0:v]crop=iw/2:ih:0:0[L];[L][1:v]hstack=inputs=2[out]" \
        -map "[out]" -crf 23 -pix_fmt yuv420p -movflags +faststart \
        "$out_composited"

    echo "    -> $out_composited"
}

# LTX-2
make_clip fig1_ltx375  refrigerator
make_clip fig1_ltx259  cow dog boat
make_clip fig6_ltx4    "stop sign" book truck banana
make_clip fig6_ltx42   toilet "cell phone" chair tie

# Wan 2.1
make_clip fig1_wan281  kite
make_clip fig1_wan318  broccoli
make_clip fig1_wan231  tie fork
make_clip fig6_wan145  horse vase
make_clip fig6_wan52   umbrella

# CogVideoX
make_clip fig6_cogx144 skis
make_clip fig6_cogx168 refrigerator

echo "=== ALL COMPOSITED ==="
