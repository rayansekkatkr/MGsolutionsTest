#!/usr/bin/env bash
# The experiment currently being run on the GPU runner. Edited for each measurement.
# This one: a first look at TensorRT for the two networks (export, build time, speed per batch
# size, difference from PyTorch), plus the sources of the pose head needed to integrate it.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=pose-inference:gpu
WORK="$HOME/exam-gpu"   # outside the checkout, which is wiped at the start of every run
docker build -q -t "$IMAGE" inference >/dev/null
mkdir -p "$WORK/trt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
docker info --format 'docker default runtime: {{.DefaultRuntime}}'
docker run --rm --gpus all --user 0 -v "$WORK/trt:/work" -v "$PWD/bench:/bench:ro" \
  -e PYTHONPATH=/app:/opt/mmpose/projects/rtmpose3d "$IMAGE" bash -c '
    pip install -q --root-user-action=ignore onnx==1.16.2 tensorrt==8.6.1.post1 2>&1 | tail -3
    python /bench/trt_probe.py --work /work --show-sources 2>&1 | grep -v -i "warn\|meshgrid\|Loads checkpoint"'
ls -la "$WORK/trt" | tail -12
echo "JOB DONE"
