#!/usr/bin/env bash
# The experiment currently being run on the GPU runner. Edited for each measurement.
# This one: where the TensorRT configurations lose accuracy (person box or pose), and whether
# falling back to the exact detector for ambiguous photos recovers it.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=pose-inference:gpu
WORK="$HOME/exam-gpu"   # outside the checkout, which is wiped at the start of every run
docker build -q -t "$IMAGE" inference >/dev/null
docker run --rm --gpus all --user 0 -v "$WORK:/work:ro" -v "$PWD/bench:/bench:ro" \
  -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d "$IMAGE" bash -c '
    pip install -q --root-user-action=ignore tensorrt==8.6.1.post1 2>&1 | tail -3
    python /bench/box_study.py --images /work/practice --reference /work/practice_reference.jsonl --trt-dir /work/trt 2>&1 \
      | grep -v -i "warn\|meshgrid\|Loads checkpoint"'
echo "JOB DONE"
