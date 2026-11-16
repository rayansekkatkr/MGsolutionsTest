#!/usr/bin/env bash
# The experiment currently being run on the GPU runner. Edited for each measurement.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=pose-inference:gpu
WORK="$PWD/tests/inference/out-gpu"
docker build -q -t "$IMAGE" inference >/dev/null
RUN=(docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$WORK:/work" -v "$PWD/bench:/bench:ro"
     -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d "$IMAGE")
for config in "fp16 8" "fp16 1" "fp32 8"; do
  set -- $config
  "${RUN[@]}" python /bench/profile.py --images /work/images --precision "$1" --batch "$2" 2>&1 \
    | grep -v -i "warn\|meshgrid\|Loads checkpoint"
done
