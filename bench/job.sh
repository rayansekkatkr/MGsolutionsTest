#!/usr/bin/env bash
# The experiment currently being run on the GPU runner. Edited for each measurement.
# This one, with detector engines that keep more or less of the network in fp32 (TF32 off),
# the box study: how far each one's person boxes are from the exact ones, how fast it is,
# and the end-to-end share of photos within tolerance.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=pose-inference:gpu
WORK="$HOME/exam-gpu"   # outside the checkout, which is wiped at the start of every run
docker build -q -t "$IMAGE" inference >/dev/null
docker run --rm --gpus all --user 0 -v "$WORK:/work" -v "$PWD/bench:/bench:ro" -v "$PWD/inference:/src:ro" \
  -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d "$IMAGE" bash -c '
    pip install -q --root-user-action=ignore onnx==1.16.2 tensorrt==8.6.1.post1 2>&1 | tail -3
    python /bench/box_study.py --images /work/practice --reference /work/practice_reference.jsonl --trt-dir /work/trt 2>&1 \
      | grep -v -i "warn\|meshgrid\|Loads checkpoint"
    chown -R '"$(id -u):$(id -g)"' /work/trt'
echo "JOB DONE"
