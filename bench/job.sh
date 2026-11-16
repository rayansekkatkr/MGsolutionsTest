#!/usr/bin/env bash
# The experiment currently being run on the GPU runner. Edited for each measurement.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=pose-inference:gpu
# Kept outside the checkout: actions/checkout wipes untracked files at the start of every run.
WORK="$HOME/exam-gpu"
bash tests/inference/fetch_images.sh tests/inference/images_gpu.txt "$WORK/images"
docker build -q -t "$IMAGE" inference >/dev/null
RUN=(docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$WORK:/work" -v "$PWD/bench:/bench:ro"
     -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d "$IMAGE")
for config in "fp16 8" "fp16 1" "fp32 8"; do
  set -- $config
  "${RUN[@]}" python /bench/step_timing.py --images /work/images --precision "$1" --batch "$2" 2>&1 \
    | grep -v -i "warn\|meshgrid\|Loads checkpoint"
done
